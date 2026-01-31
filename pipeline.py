"""
FrameDiffuser Inference Pipeline.

Dual-conditioning pipeline for G-buffer-to-image generation with:
- ControlNet for structural guidance (G-buffer)
- ControlLoRA for temporal coherence (previous frame latents)

This file is derived from control-lora-v3 (https://github.com/HighCWu/control-lora-v3)
Copyright (c) 2024 Wu Hecong, licensed under MIT License.

Modifications Copyright (c) 2025 Ole Beisswenger, Jan-Niklas Dihlmann, Hendrik Lensch
Licensed under MIT License.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from model import UNet2DConditionModelEx, ControlNetModelEx


logger = logging.get_logger(__name__)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Retrieve timesteps from scheduler."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionDualControlPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    LoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    """
    Dual-conditioning pipeline for G-buffer-to-image generation.
    
    Combines two conditioning mechanisms:
    - ControlNet: Processes 10-channel G-buffer for structural guidance
    - ControlLoRA: Conditions on previous frame latents for temporal coherence
    
    The pipeline encodes the previous frame through the VAE and concatenates it
    to the noisy latent at each denoising step, enabling autoregressive generation.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->controlnet->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder", "controlnet"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModelEx,
        controlnet: Optional[Union[ControlNetModel, ControlNetModelEx]] = None,
        scheduler: KarrasDiffusionSchedulers = None,
        safety_checker: StableDiffusionSafetyChecker = None,
        feature_extractor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__}. Ensure"
                " that you abide to the conditions of the Stable Diffusion license."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, pass `safety_checker=None`."
            )

        if controlnet is not None:
            if isinstance(controlnet, list):
                controlnet = controlnet[0]
            if hasattr(controlnet, "_orig_mod"):
                controlnet = controlnet._orig_mod
                
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def _preprocess_controlnet_input(self, controlnet_image):
        """Preprocess controlnet input to handle G-buffer format."""
        if self.controlnet is None:
            return controlnet_image
        
        is_gbuffer_controlnet = hasattr(self.controlnet, 'gbuffer_channels')
        expected_channels = getattr(self.controlnet, 'gbuffer_channels', 9)
        
        if isinstance(controlnet_image, torch.Tensor):
            channel_dim = 1 if controlnet_image.dim() == 4 else 0
            
            if controlnet_image.shape[channel_dim] == expected_channels:
                if controlnet_image.dim() == 3:
                    controlnet_image = controlnet_image.unsqueeze(0)
                return controlnet_image
            elif controlnet_image.shape[channel_dim] == 3:
                return controlnet_image
            else:
                logger.warning(f"Non-standard tensor with {controlnet_image.shape[channel_dim]} channels.")
                return controlnet_image
        
        elif isinstance(controlnet_image, list) and len(controlnet_image) > 0:
            if len(controlnet_image) == 6 and is_gbuffer_controlnet:
                from torchvision import transforms
                
                tensors = []
                for component in controlnet_image:
                    if isinstance(component, torch.Tensor):
                        if component.dim() == 3:
                            component = component.unsqueeze(0)
                        tensors.append(component)
                    elif isinstance(component, PIL.Image.Image):
                        component_tensor = transforms.ToTensor()(component).unsqueeze(0)
                        if tensors:
                            component_tensor = component_tensor.to(tensors[0].device, tensors[0].dtype)
                        tensors.append(component_tensor)
                
                # Concatenate: BaseColor(3) + Normals(3) + Depth(1) + Roughness(1) + Metallic(1) + Irradiance(1) = 10
                basecolor = tensors[0]
                normals = tensors[1]
                depth = tensors[2][:, :1] if tensors[2].shape[1] == 3 else tensors[2]
                roughness = tensors[3][:, :1] if tensors[3].shape[1] == 3 else tensors[3]
                metallic = tensors[4][:, :1] if tensors[4].shape[1] == 3 else tensors[4]
                irradiance = tensors[5][:, :1] if tensors[5].shape[1] == 3 else tensors[5]
                
                gbuffer = torch.cat([basecolor, normals, depth, roughness, metallic, irradiance], dim=1)
                return gbuffer
            else:
                return controlnet_image[0]
        
        return controlnet_image

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        """Encode text prompt into embeddings."""
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def prepare_extra_step_kwargs(self, generator, eta):
        """Prepare extra kwargs for scheduler step."""
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        lora_images,
        controlnet_images,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        extra_condition_scale=None,
        controlnet_conditioning_scale=None,
        control_guidance_start=None,
        control_guidance_end=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        """Check inputs for validity."""
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly."
                    f" Got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` must be in {self._callback_tensor_inputs}"
            )

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
    ):
        """Prepare image for pipeline."""
        if isinstance(image, torch.Tensor):
            pass
        elif isinstance(image, PIL.Image.Image):
            image = [image]
        elif isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            pass
        elif isinstance(image, list) and isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)
            
        if isinstance(image, list):
            images = []
            for img in image:
                if isinstance(img, PIL.Image.Image):
                    img = img.resize((width, height), PIL.Image.LANCZOS)
                    img = np.array(img).astype(np.float32) / 255.0
                    if img.ndim == 2:
                        img = img[..., None]
                    img = torch.from_numpy(img.transpose(2, 0, 1))
                images.append(img)
            image = torch.stack(images)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    def prepare_lora_conditions(
        self,
        lora_images,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance,
    ):
        """Prepare ControlLoRA conditions from previous frame images."""
        if not isinstance(lora_images, list):
            lora_images = [lora_images]
        
        all_latents = []
        
        for img in lora_images:
            if isinstance(img, PIL.Image.Image):
                img = img.resize((width, height), PIL.Image.LANCZOS)
                img = np.array(img).astype(np.float32) / 255.0
                img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
            elif isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    img = img.unsqueeze(0)
            
            img = img.to(device=device, dtype=dtype)
            
            # Normalize to [-1, 1] for VAE
            if img.max() > 1.0:
                img = img / 255.0
            img = img * 2.0 - 1.0
            
            # Cast to VAE dtype (float32) before encoding
            img = img.to(dtype=self.vae.dtype)
            
            # Encode to latent space
            latent = self.vae.encode(img).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
            all_latents.append(latent)
        
        # Stack latents
        if len(all_latents) == 1 and batch_size > 1:
            all_latents = all_latents * batch_size
        
        lora_latents = [lat.repeat_interleave(num_images_per_prompt, dim=0) for lat in all_latents]
        
        if do_classifier_free_guidance:
            lora_latents = [torch.cat([lat] * 2) for lat in lora_latents]
        
        return lora_latents

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        """Prepare latent variables for diffusion."""
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"Generator list length ({len(generator)}) doesn't match requested batch size ({batch_size})."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def controlnet_conditioning_scale(self):
        return self._controlnet_conditioning_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def run_safety_checker(self, image, device, dtype):
        """Run safety checker on images."""
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        lora_images: Optional[Union[PipelineImageInput, List[PipelineImageInput]]] = None,
        controlnet_images: Optional[Union[PipelineImageInput, List[PipelineImageInput]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        extra_condition_scale: Union[float, List[float]] = 1.0,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        """
        Generate images with dual conditioning.
        
        Args:
            prompt: Text prompt for generation.
            lora_images: Previous frame(s) for ControlLoRA temporal conditioning.
                         Encoded via VAE and concatenated to noisy latent.
            controlnet_images: G-buffer tensor (10 channels) or list of component images.
            height: Output height in pixels.
            width: Output width in pixels.
            num_inference_steps: Denoising steps (default: 50).
            guidance_scale: Classifier-free guidance scale (default: 7.5).
            negative_prompt: Negative prompt for guidance.
            extra_condition_scale: ControlLoRA conditioning strength.
            controlnet_conditioning_scale: ControlNet conditioning strength.
            
        Returns:
            StableDiffusionPipelineOutput with generated images.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate("callback", "1.0.0", "Use `callback_on_step_end` instead")
        if callback_steps is not None:
            deprecate("callback_steps", "1.0.0", "Use `callback_on_step_end` instead")

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        unet: UNet2DConditionModelEx = self.unet._orig_mod if is_compiled_module(self.unet) else self.unet
        num_extra_conditions = len(unet.extra_condition_names)

        # Align control guidance format
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = max(1, num_extra_conditions)
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # Check inputs
        self.check_inputs(
            prompt,
            lora_images,
            controlnet_images,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            extra_condition_scale,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._controlnet_conditioning_scale = controlnet_conditioning_scale

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if num_extra_conditions > 1 and isinstance(extra_condition_scale, float):
            extra_condition_scale = [extra_condition_scale] * num_extra_conditions

        # Default dimensions
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Encode prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Prepare ControlLoRA conditions
        lora_latents = None
        if num_extra_conditions >= 1 and lora_images is not None:
            if not isinstance(lora_images, list):
                lora_images = [lora_images] * batch_size
                
            lora_latents = self.prepare_lora_conditions(
                lora_images=lora_images,
                width=width,
                height=height,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=unet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )

        # Prepare ControlNet image
        controlnet_image = None
        if controlnet_images is not None and self.controlnet is not None:
            if not isinstance(controlnet_images, list):
                controlnet_images = [controlnet_images]
            
            is_controlnet_ex = hasattr(self.controlnet, 'gbuffer_channels')
            
            if is_controlnet_ex and len(controlnet_images) == 6:
                prepared_components = []
                for img in controlnet_images:
                    img_tensor = self.prepare_image(
                        img,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=self.controlnet.dtype,
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                    )
                    prepared_components.append(img_tensor)
                
                controlnet_image = self._preprocess_controlnet_input(prepared_components)
            else:
                img = controlnet_images[0] if isinstance(controlnet_images, list) else controlnet_images
                controlnet_image = self.prepare_image(
                    img,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=self.controlnet.dtype if self.controlnet else unet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                )

        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        self._num_timesteps = len(timesteps)

        # Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Control guidance scheduling
        extra_condition_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            extra_condition_keep.append(keeps[0] if num_extra_conditions == 1 else keeps)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_idx, timestep in enumerate(timesteps):
                if is_unet_compiled and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                    
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

                # Compute condition scale
                if isinstance(extra_condition_keep[step_idx], list):
                    if isinstance(extra_condition_scale, list):
                        cond_scale = [c * s for c, s in zip(extra_condition_scale, extra_condition_keep[step_idx])]
                    else:
                        cond_scale = [extra_condition_scale * s for s in extra_condition_keep[step_idx]]
                else:
                    if isinstance(extra_condition_scale, list):
                        cond_scale = [c * extra_condition_keep[step_idx] for c in extra_condition_scale]
                    else:
                        cond_scale = extra_condition_scale * extra_condition_keep[step_idx]

                self.unet.set_extra_condition_scale(cond_scale)

                # ControlNet forward
                controlnet_down_block_res_samples = controlnet_mid_block_res_sample = None
                if self.controlnet is not None and controlnet_image is not None:
                    # Ensure inputs match controlnet dtype
                    controlnet_input = latent_model_input.to(dtype=self.controlnet.dtype)
                    controlnet_prompt_embeds = prompt_embeds.to(dtype=self.controlnet.dtype)
                    controlnet_cond = controlnet_image.to(dtype=self.controlnet.dtype)
                    controlnet_output = self.controlnet(
                        controlnet_input,
                        timestep,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=controlnet_cond,
                        conditioning_scale=self.controlnet_conditioning_scale,
                        return_dict=False,
                    )
                    controlnet_down_block_res_samples, controlnet_mid_block_res_sample = controlnet_output
                    # Cast outputs to unet dtype for compatibility
                    if controlnet_down_block_res_samples is not None:
                        controlnet_down_block_res_samples = [x.to(dtype=self.unet.dtype) for x in controlnet_down_block_res_samples]
                    if controlnet_mid_block_res_sample is not None:
                        controlnet_mid_block_res_sample = controlnet_mid_block_res_sample.to(dtype=self.unet.dtype)

                # UNet forward with dual conditioning
                # Ensure inputs match unet dtype
                unet_input = latent_model_input.to(dtype=self.unet.dtype)
                unet_prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype)
                noise_pred = self.unet(
                    unet_input,
                    timestep,
                    encoder_hidden_states=unet_prompt_embeds,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    extra_conditions=lora_latents,
                    controlnet_down_block_res_samples=controlnet_down_block_res_samples,
                    controlnet_mid_block_res_sample=controlnet_mid_block_res_sample,
                    return_dict=False,
                )[0]

                # Classifier-free guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Scheduler step
                latents = self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]

                # Callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, step_idx, timestep, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # Progress update
                if step_idx == len(timesteps) - 1 or ((step_idx + 1) > num_warmup_steps and (step_idx + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and step_idx % callback_steps == 0:
                        callback(step_idx // getattr(self.scheduler, "order", 1), timestep, latents)

        self.unet.set_extra_condition_scale(1.0)

        # Decode latents
        if not output_type == "latent":
            # Cast latents to VAE dtype (float32 for stable decoding)
            decode_latents = latents.to(dtype=self.vae.dtype) / self.vae.config.scaling_factor
            image = self.vae.decode(decode_latents, return_dict=False, generator=generator)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
