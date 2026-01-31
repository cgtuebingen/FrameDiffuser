"""
FrameDiffuser Model Components.

Extended UNet and ControlNet models with support for:
- G-buffer conditioning (10 channels: BaseColor, Normals, Depth, Roughness, Metallic, Irradiance)
- ControlLoRA for temporal conditioning via previous frame latents

This file is derived from control-lora-v3 (https://github.com/HighCWu/control-lora-v3)
Copyright (c) 2024 Wu Hecong, licensed under MIT License.

Modifications Copyright (c) 2025 Ole Beisswenger, Jan-Niklas Dihlmann, Hendrik Lensch
Licensed under MIT License.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import copy
import os
import json
import torch
from torch import nn

from peft.tuners.lora import LoraLayer
from diffusers.configuration_utils import register_to_config
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput, UNet2DConditionModel
from diffusers.models.controlnets.controlnet import ControlNetModel, ControlNetOutput
from diffusers.utils import logging

logger = logging.get_logger(__name__)


class UNet2DConditionModelEx(UNet2DConditionModel):
    """
    UNet with ControlLoRA support for temporal conditioning.
    
    Extends the base UNet to accept additional latent channels via add_extra_conditions(),
    enabling conditioning on previous frame latents concatenated to the noisy input.
    """
    
    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
        extra_condition_names: List[str] = [],
    ):
        num_extra_conditions = len(extra_condition_names)
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels * (1 + num_extra_conditions),
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            dropout=dropout,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            attention_type=attention_type,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
        )
        
        self._internal_dict = copy.deepcopy(self._internal_dict)
        self.config.in_channels = in_channels
        self.config.extra_condition_names = extra_condition_names
    
    @property
    def extra_condition_names(self) -> List[str]:
        return self.config.extra_condition_names

    def add_extra_conditions(self, extra_condition_names: Union[str, List[str]]):
        """Add extra conditioning channels for ControlLoRA."""
        if isinstance(extra_condition_names, str):
            extra_condition_names = [extra_condition_names]
        
        conv_in_kernel = self.config.conv_in_kernel
        conv_in_weight = self.conv_in.weight
        self.config.extra_condition_names += extra_condition_names
        full_in_channels = self.config.in_channels * (1 + len(self.config.extra_condition_names))
        
        new_conv_in_weight = torch.zeros(
            conv_in_weight.shape[0], full_in_channels, conv_in_kernel, conv_in_kernel,
            dtype=conv_in_weight.dtype,
            device=conv_in_weight.device,
        )
        new_conv_in_weight[:, :conv_in_weight.shape[1]] = conv_in_weight
        self.conv_in.weight = nn.Parameter(
            new_conv_in_weight.data,
            requires_grad=conv_in_weight.requires_grad,
        )
        self.conv_in.in_channels = full_in_channels
        
        return self
    
    def activate_extra_condition_adapters(self):
        """Activate LoRA adapters for extra conditions."""
        lora_layers = [layer for layer in self.modules() if isinstance(layer, LoraLayer)]
        if len(lora_layers) > 0:
            self._hf_peft_config_loaded = True
        
        adapter_names = []
        for lora_layer in lora_layers:
            condition_adapters = [
                k for k in lora_layer.scaling.keys() 
                if k in self.config.extra_condition_names
            ]
            active_adapters = lora_layer.active_adapters
            combined_adapters = condition_adapters + active_adapters
            adapter_names.extend(combined_adapters)
        
        adapter_names = list(set(adapter_names))
        
        for lora_layer in lora_layers:
            lora_layer.set_adapter(adapter_names)
    
    def set_extra_condition_scale(self, scale: Union[float, List[float]] = 1.0):
        """Set scaling factor for extra condition LoRA adapters."""
        if isinstance(scale, float):
            scale = [scale] * len(self.config.extra_condition_names)

        lora_layers = [layer for layer in self.modules() if isinstance(layer, LoraLayer)]
        for s, n in zip(scale, self.config.extra_condition_names):
            for lora_layer in lora_layers:
                lora_layer.set_scale(n, s)
    
    @property
    def default_half_lora_target_modules(self) -> List[str]:
        module_names = []
        for name, module in self.named_modules():
            if "conv_out" in name or "up_blocks" in name:
                continue
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module_names.append(name)
        return list(set(module_names))
    
    @property
    def default_full_lora_target_modules(self) -> List[str]:
        module_names = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module_names.append(name)
        return list(set(module_names))
    
    @property
    def default_half_skip_attn_lora_target_modules(self) -> List[str]:
        return [
            module_name
            for module_name in self.default_half_lora_target_modules 
            if all(
                not module_name.endswith(attn_name) 
                for attn_name in ["to_k", "to_q", "to_v", "to_out.0"]
            )
        ]
    
    @property
    def default_full_skip_attn_lora_target_modules(self) -> List[str]:
        return [
            module_name
            for module_name in self.default_full_lora_target_modules 
            if all(
                not module_name.endswith(attn_name) 
                for attn_name in ["to_k", "to_q", "to_v", "to_out.0"]
            )
        ]

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        extra_conditions: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        controlnet_down_block_res_samples: Optional[Tuple[torch.Tensor]] = None,
        controlnet_mid_block_res_sample: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        """Forward pass with optional extra conditions and ControlNet residuals."""
        
        # Handle extra conditions (ControlLoRA previous frame latents)
        if extra_conditions is not None:
            is_classifier_free_guidance = encoder_hidden_states.shape[0] == 2 * sample.shape[0]
            
            if isinstance(extra_conditions, list):
                if len(extra_conditions) != sample.shape[0]:
                    raise ValueError(
                        f"Number of conditions ({len(extra_conditions)}) must match "
                        f"sample batch size ({sample.shape[0]})"
                    )
                stacked_conditions = torch.cat(extra_conditions, dim=0)
                sample = torch.cat([sample, stacked_conditions], dim=1)
            else:
                if extra_conditions.shape[0] != sample.shape[0]:
                    raise ValueError(
                        f"Condition batch ({extra_conditions.shape[0]}) must match "
                        f"sample batch ({sample.shape[0]})"
                    )
                sample = torch.cat([sample, extra_conditions], dim=1)
        
        # Combine ControlNet residuals with any additional residuals
        final_down_block_res = down_block_additional_residuals
        final_mid_block_res = mid_block_additional_residual
        
        if controlnet_down_block_res_samples is not None:
            if final_down_block_res is None:
                final_down_block_res = controlnet_down_block_res_samples
            else:
                final_down_block_res = [
                    r1 + r2 for r1, r2 in zip(final_down_block_res, controlnet_down_block_res_samples)
                ]
                
        if controlnet_mid_block_res_sample is not None:
            if final_mid_block_res is None:
                final_mid_block_res = controlnet_mid_block_res_sample
            else:
                final_mid_block_res = final_mid_block_res + controlnet_mid_block_res_sample
        
        return super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=final_down_block_res,
            mid_block_additional_residual=final_mid_block_res,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )


class ControlNetModelEx(ControlNetModel):
    """
    ControlNet for 10-channel G-buffer input.
    
    Processes G-buffer data (BaseColor, Normals, Depth, Roughness, Metallic, Irradiance)
    to provide structural guidance for the diffusion process. The conditioning embedding
    is replaced with a deeper network to handle the increased channel count.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 10,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (64, 96, 256, 512),
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        gbuffer_channels: int = 10,
        **kwargs
    ):
        self.gbuffer_channels = gbuffer_channels
        
        super().__init__(
            in_channels=in_channels,
            conditioning_channels=conditioning_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order="rgb",
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            **kwargs
        )
        
        # Replace conditioning embedding for multi-channel G-buffer input
        if gbuffer_channels != 3:
            self.controlnet_cond_embedding = nn.Sequential(
                nn.Conv2d(gbuffer_channels, conditioning_embedding_out_channels[0], kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(conditioning_embedding_out_channels[0], conditioning_embedding_out_channels[0], kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(conditioning_embedding_out_channels[0], conditioning_embedding_out_channels[1], kernel_size=3, padding=1, stride=2),
                nn.SiLU(),
                nn.Conv2d(conditioning_embedding_out_channels[1], conditioning_embedding_out_channels[1], kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(conditioning_embedding_out_channels[1], conditioning_embedding_out_channels[2], kernel_size=3, padding=1, stride=2),
                nn.SiLU(),
                nn.Conv2d(conditioning_embedding_out_channels[2], conditioning_embedding_out_channels[2], kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(conditioning_embedding_out_channels[2], conditioning_embedding_out_channels[3], kernel_size=3, padding=1, stride=2),
                nn.SiLU(),
                nn.Conv2d(conditioning_embedding_out_channels[3], conditioning_embedding_out_channels[3], kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(conditioning_embedding_out_channels[3], block_out_channels[0], kernel_size=3, padding=1)
            )
            
            # Zero-initialize final layer for stable training start
            nn.init.zeros_(self.controlnet_cond_embedding[-1].weight)
            nn.init.zeros_(self.controlnet_cond_embedding[-1].bias)
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        """Forward pass handling multi-channel G-buffer inputs."""
        
        # Get model weight dtype from a known layer
        model_dtype = self.time_embedding.linear_1.weight.dtype
        
        # Cast all inputs to model dtype
        sample = sample.to(dtype=model_dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=model_dtype)
        controlnet_cond = controlnet_cond.to(dtype=model_dtype)
        
        # For multi-channel G-buffer, skip channel order processing
        if controlnet_cond.shape[1] > 3:
            if attention_mask is not None:
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # Time embedding
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            timesteps = timesteps.expand(sample.shape[0])
            t_emb = self.time_proj(timesteps)
            # Cast to match model weight dtype
            t_emb = t_emb.to(dtype=model_dtype)
            emb = self.time_embedding(t_emb, timestep_cond)
            aug_emb = None

            if self.class_embedding is not None:
                if class_labels is None:
                    raise ValueError("class_labels should be provided when num_class_embeds > 0")
                if self.config.class_embed_type == "timestep":
                    class_labels = self.time_proj(class_labels)
                class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
                emb = emb + class_emb

            if self.config.addition_embed_type is not None:
                if self.config.addition_embed_type == "text":
                    aug_emb = self.add_embedding(encoder_hidden_states)
                elif self.config.addition_embed_type == "text_time":
                    text_embeds = added_cond_kwargs.get("text_embeds")
                    time_ids = added_cond_kwargs.get("time_ids")
                    time_embeds = self.add_time_proj(time_ids.flatten())
                    time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                    add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                    add_embeds = add_embeds.to(emb.dtype)
                    aug_emb = self.add_embedding(add_embeds)

            emb = emb + aug_emb if aug_emb is not None else emb

            # Pre-process
            sample = self.conv_in(sample)
            controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
            sample = sample + controlnet_cond

            # Down blocks
            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                down_block_res_samples += res_samples

            # Mid block
            if self.mid_block is not None:
                if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                else:
                    sample = self.mid_block(sample, emb)

            # ControlNet blocks
            controlnet_down_block_res_samples = ()
            for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
                down_block_res_sample = controlnet_block(down_block_res_sample)
                controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = controlnet_down_block_res_samples
            mid_block_res_sample = self.controlnet_mid_block(sample)

            # Scaling
            if guess_mode and not self.config.global_pool_conditions:
                scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)
                scales = scales * conditioning_scale
                down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
                mid_block_res_sample = mid_block_res_sample * scales[-1]
            else:
                down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
                mid_block_res_sample = mid_block_res_sample * conditioning_scale

            if self.config.global_pool_conditions:
                down_block_res_samples = [
                    torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
                ]
                mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

            if not return_dict:
                return (down_block_res_samples, mid_block_res_sample)

            return ControlNetOutput(
                down_block_res_samples=down_block_res_samples,
                mid_block_res_sample=mid_block_res_sample
            )
        else:
            # Standard 3-channel image - use parent class method
            return super().forward(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                conditioning_scale=conditioning_scale,
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs=cross_attention_kwargs,
                guess_mode=guess_mode,
                return_dict=return_dict,
            )
    
    @classmethod
    def from_unet(cls, unet, **kwargs):
        """Create ControlNetModelEx from UNet."""
        gbuffer_channels = kwargs.pop("gbuffer_channels", 10)
        
        kwargs["conditioning_channels"] = gbuffer_channels
        kwargs["controlnet_conditioning_channel_order"] = "rgb"
        
        controlnet = super().from_unet(unet, **kwargs)
        controlnet.gbuffer_channels = gbuffer_channels
        
        if controlnet.config.conditioning_channels != 3:
            out_channels = controlnet.config.block_out_channels[0]
            embedding_out_channels = getattr(controlnet.config, "conditioning_embedding_out_channels", (64, 96, 256, 512))
            
            controlnet.controlnet_cond_embedding = nn.Sequential(
                nn.Conv2d(gbuffer_channels, embedding_out_channels[0], kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(embedding_out_channels[0], embedding_out_channels[0], kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(embedding_out_channels[0], embedding_out_channels[1], kernel_size=3, padding=1, stride=2),
                nn.SiLU(),
                nn.Conv2d(embedding_out_channels[1], embedding_out_channels[1], kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(embedding_out_channels[1], embedding_out_channels[2], kernel_size=3, padding=1, stride=2),
                nn.SiLU(),
                nn.Conv2d(embedding_out_channels[2], embedding_out_channels[2], kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(embedding_out_channels[2], embedding_out_channels[3], kernel_size=3, padding=1, stride=2),
                nn.SiLU(),
                nn.Conv2d(embedding_out_channels[3], embedding_out_channels[3], kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(embedding_out_channels[3], out_channels, kernel_size=3, padding=1)
            )
            
            nn.init.zeros_(controlnet.controlnet_cond_embedding[-1].weight)
            nn.init.zeros_(controlnet.controlnet_cond_embedding[-1].bias)
            controlnet.config.conditioning_channels = gbuffer_channels
        
        return controlnet
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load pretrained ControlNetModelEx."""
        config_dict = {}
        is_local = os.path.isdir(pretrained_model_name_or_path)
        
        if is_local:
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                    
                gbuffer_channels = config_dict.get("gbuffer_channels", 10)
                conditioning_channels = config_dict.get("conditioning_channels", gbuffer_channels)
                
                kwargs["conditioning_channels"] = conditioning_channels
                kwargs["controlnet_conditioning_channel_order"] = "rgb"
        else:
            gbuffer_channels = kwargs.pop("gbuffer_channels", 10)
            kwargs["conditioning_channels"] = gbuffer_channels
            kwargs["controlnet_conditioning_channel_order"] = "rgb"
        
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.gbuffer_channels = kwargs["conditioning_channels"]
        
        return model
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save model with G-buffer config."""
        os.makedirs(save_directory, exist_ok=True)
        
        state_dict = self.state_dict()
        total_params = sum(p.numel() for p in self.parameters())
        
        weight_name = "diffusion_pytorch_model.safetensors" if kwargs.get("safe_serialization", True) else "diffusion_pytorch_model.bin"
        output_model_file = os.path.join(save_directory, weight_name)
        
        if kwargs.get("safe_serialization", True):
            from safetensors.torch import save_file as safe_save_file
            safe_save_file(state_dict, output_model_file, metadata={"format": "pt"})
        else:
            torch.save(state_dict, output_model_file)
        
        if hasattr(self.config, 'to_dict'):
            config_dict = self.config.to_dict()
        else:
            config_dict = dict(self.config)
        
        config_dict["_class_name"] = "ControlNetModelEx"
        config_dict["gbuffer_channels"] = self.gbuffer_channels
        config_dict["total_parameters"] = total_params
        
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)



