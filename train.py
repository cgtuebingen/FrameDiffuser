#!/usr/bin/env python
# coding=utf-8
"""
FrameDiffuser Training Script.

Three-stage training pipeline for G-buffer conditioned neural rendering:
- Stage 1: ControlNet training (G-buffer encoding)
- Stage 2: ControlLoRA training (temporal conditioning)
- Stage 3: Joint fine-tuning

This file is derived from control-lora-v3 (https://github.com/HighCWu/control-lora-v3)
Copyright (c) 2024 Wu Hecong, licensed under MIT License.

Modifications Copyright (c) 2025 Ole Beisswenger, Jan-Niklas Dihlmann, Hendrik Lensch
Licensed under MIT License.
"""

import argparse
import contextlib
import gc
import glob
import logging
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import sys
import random
import shutil
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder, get_token
from packaging import version
from PIL import Image, ImageDraw
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import DPMSolverMultistepScheduler
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
    ControlNetModel,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.loaders import LoraLoaderMixin

from model import UNet2DConditionModelEx, ControlNetModelEx
from pipeline import StableDiffusionDualControlPipeline
from GBufferDataset import GBufferDataset
from torch.utils.data import DataLoader
from GradientMonitor import GradientMonitor

if is_wandb_available():
    import wandb


def compute_irradiance_map(final_img, albedo_img, smoothing_iterations=15):
    """Compute irradiance map matching training method."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    
    if isinstance(final_img, Image.Image):
        final_tensor = torch.from_numpy(np.array(final_img)).float() / 255.0
    else:
        final_tensor = torch.from_numpy(final_img).float() / 255.0
        
    if isinstance(albedo_img, Image.Image):
        albedo_tensor = torch.from_numpy(np.array(albedo_img)).float() / 255.0
    else:
        albedo_tensor = torch.from_numpy(albedo_img).float() / 255.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_tensor = final_tensor.to(device)
    albedo_tensor = albedo_tensor.to(device)
    
    if final_tensor.dim() == 3 and final_tensor.shape[2] == 3:
        final_gray = 0.299 * final_tensor[:, :, 0] + 0.587 * final_tensor[:, :, 1] + 0.114 * final_tensor[:, :, 2]
    else:
        final_gray = final_tensor
        
    if albedo_tensor.dim() == 3 and albedo_tensor.shape[2] == 3:
        albedo_gray = 0.299 * albedo_tensor[:, :, 0] + 0.587 * albedo_tensor[:, :, 1] + 0.114 * albedo_tensor[:, :, 2]
    else:
        albedo_gray = albedo_tensor
    
    valid_mask = albedo_gray > 0.01
    
    irradiance = torch.ones_like(albedo_gray)
    irradiance[valid_mask] = final_gray[valid_mask] / (albedo_gray[valid_mask] + 1e-6)
    irradiance[~valid_mask] = 2.0
    
    if smoothing_iterations > 0:
        irradiance = irradiance.unsqueeze(0).unsqueeze(0)
        
        for _ in range(smoothing_iterations):
            kernel_size = 5
            sigma = 1.0
            channels = 1
            kernel = torch.arange(kernel_size).float() - kernel_size // 2
            kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
            kernel = (kernel / kernel.sum()).view(1, 1, kernel_size, 1)
            kernel = kernel * kernel.transpose(-1, -2)
            kernel = kernel.expand(channels, 1, kernel_size, kernel_size).to(device)
            
            irradiance = F.conv2d(F.pad(irradiance, (2, 2, 2, 2), mode='reflect'), kernel, groups=channels)
        
        irradiance = irradiance.squeeze(0).squeeze(0)
    
    result = torch.clamp(irradiance, 0, 2) / 2.0
    
    return result.cpu().numpy()

def auto_detect_validation_from_folder(folder_path):
    """
    Auto-detect validation files from a folder.
    
    Supports following naming conventions:
    - Simple: FinalImage/FinalImage_0001.png
    - Prefix: FinalImage/SequenceName_FinalImage_0001.png
    
    Returns dict with paths or None if detection fails.
    Picks a frame with frame_idx > 0 so there's a valid previous frame.
    """
    if not os.path.isdir(folder_path):
        return None
    
    # Check for FinalImage folder
    final_dir = os.path.join(folder_path, "FinalImage")
    if not os.path.isdir(final_dir):
        return None
    
    # Find all FinalImage files and detect naming pattern
    files = sorted([f for f in os.listdir(final_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not files:
        return None
    
    # Detect naming pattern and collect frame numbers
    frames = []
    prefix = None
    simple_naming = False
    
    for f in files:
        # Try simple naming: FinalImage_0001.png
        match = re.match(r'^FinalImage_(\d+)\.(png|jpg|jpeg)$', f)
        if match:
            simple_naming = True
            frames.append((int(match.group(1)), match.group(1), f))
            continue
        
        # Try prefix naming: Prefix_FinalImage_0001.png
        match = re.match(r'^(.+)_FinalImage_(\d+)\.(png|jpg|jpeg)$', f)
        if match:
            if prefix is None:
                prefix = match.group(1)
            frames.append((int(match.group(2)), match.group(2), f))
    
    if not frames:
        return None
    
    # Sort by frame number and find first frame with idx > 0 (has prev frame)
    frames.sort(key=lambda x: x[0])
    
    selected_frame = None
    for frame_num, frame_str, filename in frames:
        if frame_num > 0:
            # Check if previous frame exists
            prev_num = frame_num - 1
            prev_str = f"{prev_num:0{len(frame_str)}d}"
            if simple_naming:
                prev_filename = f"FinalImage_{prev_str}.png"
            else:
                prev_filename = f"{prefix}_FinalImage_{prev_str}.png"
            
            prev_path = os.path.join(final_dir, prev_filename)
            # Also check jpg
            if not os.path.exists(prev_path):
                prev_path = prev_path.replace('.png', '.jpg')
            
            if os.path.exists(prev_path):
                selected_frame = (frame_num, frame_str, filename, prev_str)
                break
    
    if selected_frame is None:
        # Fallback to first frame (will use same frame as prev)
        frame_num, frame_str, filename = frames[0]
        selected_frame = (frame_num, frame_str, filename, frame_str)
    
    frame_num, frame_str, filename, prev_str = selected_frame
    
    # Build component paths
    required = ["BaseColor", "Normals", "Depth", "Roughness"]
    
    def find_component_file(component, frame_str):
        component_dir = os.path.join(folder_path, component)
        if not os.path.isdir(component_dir):
            return None
        
        # Try simple naming
        for ext in ['.png', '.jpg', '.jpeg']:
            simple_path = os.path.join(component_dir, f"{component}_{frame_str}{ext}")
            if os.path.exists(simple_path):
                return simple_path
        
        # Try prefix naming
        if prefix:
            for ext in ['.png', '.jpg', '.jpeg']:
                prefix_path = os.path.join(component_dir, f"{prefix}_{component}_{frame_str}{ext}")
                if os.path.exists(prefix_path):
                    return prefix_path
        
        return None
    
    # Collect all component paths for current frame
    result = {
        'controlnet_validation_image': [],
        'validation_image': None,
        'final_validation_image': None,
        'validation_prev_final_image': None,
        'validation_prev_albedo_image': None,
    }
    
    # G-buffer components
    all_found = True
    for component in required:
        path = find_component_file(component, frame_str)
        if path:
            result['controlnet_validation_image'].append(path)
        else:
            all_found = False
            break
    
    if not all_found:
        return None
    
    # Optional metallic
    metallic_path = find_component_file("Metallic", frame_str)
    if metallic_path:
        result['controlnet_validation_image'].append(metallic_path)
    
    # FinalImage for current frame
    final_path = os.path.join(final_dir, filename)
    result['final_validation_image'] = final_path
    
    # Previous frame for ControlLoRA conditioning
    if simple_naming:
        prev_filename = f"FinalImage_{prev_str}.png"
    else:
        prev_filename = f"{prefix}_FinalImage_{prev_str}.png"
    prev_path = os.path.join(final_dir, prev_filename)
    if not os.path.exists(prev_path):
        prev_path = prev_path.replace('.png', '.jpg')
    if os.path.exists(prev_path):
        result['validation_image'] = prev_path
        result['validation_prev_final_image'] = prev_path
    else:
        # Fallback to current frame
        result['validation_image'] = final_path
        result['validation_prev_final_image'] = final_path
    
    # Albedo for irradiance
    result['validation_prev_albedo_image'] = result['controlnet_validation_image'][0]  # BaseColor
    
    return result

_last_validation_img_log_time = 0
_validation_logs_skipped = 0

def prepare_controlnet_for_training(controlnet):
    """Prepare ControlNet for training with G-buffer input."""
    if not isinstance(controlnet, ControlNetModelEx):
        logger.warning(f"Expected ControlNetModelEx but got {type(controlnet).__name__}")
        if isinstance(controlnet, ControlNetModel):
            if not hasattr(controlnet, 'gbuffer_channels'):
                controlnet.gbuffer_channels = 10
    
    controlnet.train()
    
    return controlnet

def validate_gbuffer_tensor(tensor, name="G-buffer tensor"):
    """Validate G-buffer tensor has 10 channels (9 G-buffer + 1 irradiance)."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    
    if tensor.dim() == 3:
        channel_dim = 0
        if tensor.shape[channel_dim] != 10:
            raise ValueError(
                f"{name} must have 10 channels in dim {channel_dim}, got {tensor.shape[channel_dim]}. "
                f"Full shape: {tensor.shape}"
            )
    elif tensor.dim() == 4:
        channel_dim = 1
        if tensor.shape[channel_dim] != 10:
            raise ValueError(
                f"{name} must have 10 channels in dim {channel_dim}, got {tensor.shape[channel_dim]}. "
                f"Full shape: {tensor.shape}"
            )
    else:
        raise ValueError(f"{name} must be 3D [C,H,W] or 4D [B,C,H,W], got shape {tensor.shape}")
    
    return True

def prepare_validation_gbuffer(components, target_size=(512, 512), final_image=None, prev_final_image=None, prev_albedo_image=None, irradiance_smoothing_iterations=0):
    """Create properly formatted 10-channel G-buffer tensor with masking."""
    import numpy as np
    
    if len(components) != 5:
        raise ValueError(f"G-buffer requires 5 components, got {len(components)}")
    
    from gbuffer_masking_utils import fill_black_pixels_in_basecolor
    
    if len(components) > 4 and components[4] is not None:
        metallic_img = components[4]
        if not isinstance(metallic_img, Image.Image):
            metallic_img = Image.fromarray(metallic_img)
        components[4] = metallic_img
    else:
        original_size = components[0].size
        metallic_black = Image.new('L', original_size, 0)
        if len(components) > 4:
            components[4] = metallic_black
        else:
            components.append(metallic_black)
    
    basecolor = components[0]
    depth_pil = components[2]
    if final_image is not None:
        component_size = basecolor.size
        if final_image.size != component_size:
            final_image = final_image.resize(component_size, Image.LANCZOS)
        
        basecolor = fill_black_pixels_in_basecolor(basecolor, depth_pil, final_image)
        components[0] = basecolor
    
    if prev_final_image is not None and prev_albedo_image is not None:
        component_size = basecolor.size
        if prev_final_image.size != component_size:
            prev_final_image = prev_final_image.resize(component_size, Image.LANCZOS)
        if prev_albedo_image.size != component_size:
            prev_albedo_image = prev_albedo_image.resize(component_size, Image.LANCZOS)
        
        prev_irradiance = compute_irradiance_map(prev_final_image, prev_albedo_image, 0)
        irradiance_uint8 = (prev_irradiance * 255).astype(np.uint8)
        irradiance_pil = Image.fromarray(irradiance_uint8, mode='L')
    else:
        component_size = basecolor.size
        irradiance_pil = Image.new('L', component_size, 128)
    
    components = [img.resize(target_size, Image.LANCZOS) for img in components]
    irradiance_pil = irradiance_pil.resize(target_size, Image.LANCZOS)
    
    rgb_tensors = []
    
    basecolor_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])(components[0])
    rgb_tensors.append(basecolor_tensor)
    
    normals_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])(components[1])
    rgb_tensors.append(normals_tensor)
    
    gray_tensors = []
    for i in range(2, 5):
        gray_img = components[i].convert("L")
        tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])(gray_img)
        gray_tensors.append(tensor)
    
    irradiance_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])(irradiance_pil)
    gray_tensors.append(irradiance_tensor)
    
    tensor = torch.cat(rgb_tensors + gray_tensors, dim=0)
    
    if tensor.shape[0] != 10:
        raise ValueError(f"Expected 10-channel tensor, got {tensor.shape[0]} channels")
    
    
    return tensor

def generate_conditioning_images(
    pipeline, 
    args, 
    accelerator, 
    tokenizer, 
    weight_dtype, 
    dataset,
    step,
    batch_size=4,
    previous_generated=None
):
    """
    Generate conditioning images using the current model for self-conditioning training.
    
    Args:
        pipeline: StableDiffusionDualControlPipeline instance
        args: Training arguments namespace
        previous_generated: Results from last generation run (for chaining)
        
    Returns:
        dict with 'version' (step number) and 'generated_images' (seq_id -> list of gen info)
    """
    
    if not accelerator.is_main_process:
        return {}
    
    chaining_ratio = getattr(args, 'generated_chaining_ratio', 0.0)
    print(f"Starting generation at step {step} (chaining ratio: {chaining_ratio*100:.0f}%)...")
    
    unet_training = pipeline.unet.training
    vae_training = pipeline.vae.training
    text_encoder_training = pipeline.text_encoder.training
    controlnet_training = pipeline.controlnet.training if pipeline.controlnet is not None else False
    
    pipeline.unet.eval()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    if pipeline.controlnet is not None:
        pipeline.controlnet.eval()
    
    with torch.no_grad():
        for param in pipeline.unet.parameters():
            param.grad = None
        if pipeline.controlnet is not None:
            for param in pipeline.controlnet.parameters():
                param.grad = None
    
    torch.cuda.empty_cache()
    
    try:
        output_dir = os.path.abspath(os.path.join(args.output_dir, args.generated_conditioning_dir))
        os.makedirs(output_dir, exist_ok=True)
        
        results = {"version": step, "generated_images": {}}
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + step) if args.seed else None
        
        sequences = dataset.sequence_frames
        total_to_generate = min(args.generated_conditioning_count, 1000)
        
        # Build sample list
        all_samples = []
        chained_count = 0
        fresh_count = 0
        
        # First: Add chained samples from previous generation
        if previous_generated is not None and chaining_ratio > 0:
            prev_images = previous_generated.get("generated_images", {})
            
            # Collect ALL chainable candidates first
            chainable_candidates = []
            
            for seq_id, gen_list in prev_images.items():
                if seq_id not in sequences:
                    continue
                    
                seq_frames = sequences[seq_id]
                # Build frame_idx -> dataset_idx lookup
                frame_to_dataset = {fidx: didx for didx, fidx in seq_frames}
                
                for gen_info in gen_list:
                    prev_frame_idx = gen_info['frame_idx']
                    prev_image_path = gen_info['path']
                    
                    # Find next frame(s) to generate
                    for offset in [1, 2]:
                        next_frame_idx = prev_frame_idx + offset
                        if next_frame_idx in frame_to_dataset:
                            chainable_candidates.append({
                                'dataset_idx': frame_to_dataset[next_frame_idx],
                                'frame_idx': next_frame_idx,
                                'seq_id': seq_id,
                                'chained_from': {
                                    'frame_idx': prev_frame_idx,
                                    'path': prev_image_path
                                }
                            })
                            break  # Only use first valid offset
            
            # Randomly select chaining_ratio of total_to_generate from candidates
            # First deduplicate by target frame (prefer offset +1 over +2)
            seen_targets = {}
            for candidate in chainable_candidates:
                key = (candidate['seq_id'], candidate['frame_idx'])
                if key not in seen_targets:
                    seen_targets[key] = candidate
                # If already seen, keep the first one (which has smaller offset due to loop order)
            
            chainable_candidates = list(seen_targets.values())
            
            num_chained = min(
                len(chainable_candidates),
                int(total_to_generate * chaining_ratio)
            )
            
            if num_chained > 0:
                random.shuffle(chainable_candidates)
                all_samples = chainable_candidates[:num_chained]
                chained_count = num_chained
        
        # Fill remaining slots with fresh (non-chained) samples
        remaining = total_to_generate - len(all_samples)
        if remaining > 0:
            # Collect frames not already in all_samples
            existing_frames = {(s['seq_id'], s['frame_idx']) for s in all_samples}
            
            fresh_candidates = []
            for seq_id, seq_frames in sequences.items():
                for dataset_idx, frame_idx in seq_frames:
                    if (seq_id, frame_idx) not in existing_frames:
                        fresh_candidates.append({
                            'dataset_idx': dataset_idx,
                            'frame_idx': frame_idx,
                            'seq_id': seq_id,
                            'chained_from': None
                        })
            
            random.shuffle(fresh_candidates)
            for sample in fresh_candidates[:remaining]:
                all_samples.append(sample)
                fresh_count += 1
        
        print(f"Generating {len(all_samples)} images ({chained_count} chained from prev gen, {fresh_count} fresh)")
        
        pipeline.set_progress_bar_config(disable=True)
        
        if hasattr(pipeline.unet, 'extra_condition_names') and len(pipeline.unet.extra_condition_names) > 0:
            pipeline.unet.activate_extra_condition_adapters()
        
        num_batches = (len(all_samples) + batch_size - 1) // batch_size
        progress_bar = tqdm(total=num_batches, desc="Generating")
        
        total_generated = 0
        
        with torch.no_grad():
            for batch_idx in range(0, len(all_samples), batch_size):
                batch_samples = all_samples[batch_idx:batch_idx + batch_size]
                
                batch_prompts = []
                batch_lora_images = []
                batch_gbuffers = []
                batch_metadata = []
                
                for sample_info in batch_samples:
                    try:
                        seq_id = sample_info['seq_id']
                        frame_idx = sample_info['frame_idx']
                        chained_from = sample_info['chained_from']
                        
                        if chained_from is not None:
                            # Chained: use previously generated image as prev frame
                            chained_image = Image.open(chained_from['path']).convert("RGB")
                            chained_frame_idx = chained_from['frame_idx']
                            
                            sample = dataset.get_clean_sample_for_generation(
                                sample_info['dataset_idx'],
                                chained_prev_image=chained_image,
                                chained_frame_idx=chained_frame_idx
                            )
                        else:
                            # Fresh: use GT prev frame
                            sample = dataset.get_clean_sample_for_generation(
                                sample_info['dataset_idx']
                            )
                        
                        input_ids = sample["input_ids"].unsqueeze(0).to(accelerator.device)
                        prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                        batch_prompts.append(prompt)
                        
                        gbuffer = sample["controlnet_pixel_values"]
                        if gbuffer.dim() == 3:
                            gbuffer = gbuffer.unsqueeze(0)
                        gbuffer = gbuffer.to(accelerator.device, dtype=weight_dtype)
                        batch_gbuffers.append(gbuffer)
                        
                        # ControlLoRA input
                        prev_frame_tensor = sample["conditioning_pixel_values"]
                        prev_frame_np = prev_frame_tensor.cpu().numpy()
                        prev_frame_np = np.transpose(prev_frame_np, (1, 2, 0))
                        prev_frame_np = (prev_frame_np * 0.5 + 0.5).clip(0, 1)
                        prev_frame_pil = Image.fromarray((prev_frame_np * 255).astype(np.uint8))
                        batch_lora_images.append(prev_frame_pil)
                        
                        batch_metadata.append({
                            'seq_id': seq_id,
                            'frame_idx': frame_idx,
                            'was_chained': chained_from is not None
                        })
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
                
                if len(batch_prompts) == 0:
                    progress_bar.update(1)
                    continue
                
                batch_gbuffer_tensor = torch.cat(batch_gbuffers, dim=0)
                
                output = pipeline(
                    batch_prompts,
                    negative_prompt=[""] * len(batch_prompts),
                    guidance_scale=1.0,
                    controlnet_images=batch_gbuffer_tensor,
                    lora_images=batch_lora_images,
                    num_inference_steps=args.generated_conditioning_steps,
                    num_images_per_prompt=1,
                    generator=generator,
                    height=args.resolution,
                    width=args.resolution,
                    controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                    output_type="pil",
                )
                
                for i, (image, metadata) in enumerate(zip(output.images, batch_metadata)):
                    seq_id = metadata['seq_id']
                    frame_idx = metadata['frame_idx']
                    was_chained = metadata['was_chained']
                    
                    filename = f"{seq_id}_frame_{frame_idx:08d}_gen{step}.png"
                    image_path = os.path.join(output_dir, filename)
                    image.save(image_path)
                    
                    if seq_id not in results["generated_images"]:
                        results["generated_images"][seq_id] = []
                    
                    results["generated_images"][seq_id].append({
                        "frame_idx": frame_idx,
                        "path": image_path,
                        "generation_step": step,
                        "was_chained": was_chained
                    })
                    
                    total_generated += 1
                
                progress_bar.update(1)
                
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        progress_bar.close()
        final_chained = sum(1 for s in all_samples if s['chained_from'] is not None)
        print(f"Generated {total_generated} images ({final_chained} chained, {total_generated - final_chained} fresh)")
        
    finally:
        if unet_training:
            pipeline.unet.train()
        if vae_training:
            pipeline.vae.train()
        if text_encoder_training:
            pipeline.text_encoder.train()
        if controlnet_training and pipeline.controlnet is not None:
            pipeline.controlnet.train()
        
        if hasattr(pipeline, '_cached_encoder_hidden_states'):
            delattr(pipeline, '_cached_encoder_hidden_states')
        
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return results

check_min_version("0.29.0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def prepare_validation_images(images_list, target_size=(512, 512)):
    """Ensure all images have the same resolution for validation"""
    if images_list is None:
        return None
    return [img.resize(target_size, Image.LANCZOS) for img in images_list]


def log_validation(
    vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step, is_final_validation=False
):
    import numpy as np
    global _last_validation_img_log_time, _validation_logs_skipped
    
    current_time = time.time()
    if current_time - _last_validation_img_log_time > 30.0:
        if _validation_logs_skipped > 0:
            logger.info(f"Processed {_validation_logs_skipped} validation runs")
            _validation_logs_skipped = 0
        logger.info(f"Running validation at step {step}...")
        _last_validation_img_log_time = current_time
    else:
        _validation_logs_skipped += 1

    unet_was_training = None
    vae_was_training = vae.training if hasattr(vae, 'training') else False
    text_encoder_was_training = text_encoder.training if hasattr(text_encoder, 'training') else False
    controlnet_was_training = None

    if not is_final_validation:
        unet_was_training = unet.training if hasattr(unet, 'training') else False
        
        unet = accelerator.unwrap_model(unet)
        if hasattr(unet, "_orig_mod"):
            unet = unet._orig_mod
            
        if args.train_controlnet and controlnet is not None:
            controlnet_was_training = controlnet.training if hasattr(controlnet, 'training') else False
            controlnet = accelerator.unwrap_model(controlnet)
            if hasattr(controlnet, "_orig_mod"):
                controlnet = controlnet._orig_mod
            if isinstance(controlnet, list) and len(controlnet) > 0:
                controlnet = controlnet[0]
    else:
        unet = UNet2DConditionModelEx.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            torch_dtype=weight_dtype)
        
        unet_was_training = False
        
        if args.train_controllora or args.use_controllora:
            unet = unet.add_extra_conditions([args.lora_adapter_name])
        
        if args.train_controlnet and args.controlnet_model_name_or_path:
            config_path = os.path.join(args.controlnet_model_name_or_path, "config.json")
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                cross_attention_dim = config.get('cross_attention_dim', 768)
                gbuffer_channels = config.get('gbuffer_channels', 10)
            except:
                cross_attention_dim = 768
                gbuffer_channels = 10
            
            controlnet = ControlNetModelEx.from_pretrained(
                args.controlnet_model_name_or_path,
                cross_attention_dim=cross_attention_dim,
                gbuffer_channels=gbuffer_channels,
                torch_dtype=weight_dtype
            )
            logger.info(f"Loaded ControlNetEx from {args.controlnet_model_name_or_path}")
        elif args.train_controlnet:
            config_path = os.path.join(args.output_dir, "controlnet", "config.json")
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                cross_attention_dim = config.get('cross_attention_dim', 768)
                gbuffer_channels = config.get('gbuffer_channels', 10)
            except:
                cross_attention_dim = 768
                gbuffer_channels = 10
            
            controlnet = ControlNetModelEx.from_pretrained(
                os.path.join(args.output_dir, "controlnet"),
                cross_attention_dim=cross_attention_dim,
                gbuffer_channels=gbuffer_channels,
                torch_dtype=weight_dtype
            )

    unet.eval()
    vae.eval()
    text_encoder.eval()
    
    if controlnet is not None:
        if controlnet_was_training is None:
            controlnet_was_training = controlnet.training if hasattr(controlnet, 'training') else False
        controlnet.eval()
    
    
    try:
        scheduler = DPMSolverMultistepScheduler.from_config(
            DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler").config
        )
        
        pipeline = StableDiffusionDualControlPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        
        args.stored_num_validation_images = args.num_validation_images
        args.num_validation_images = 1
        
        if is_final_validation:
            if args.train_controllora or args.use_controllora:
                pipeline.load_lora_weights(args.output_dir, adapter_name=args.lora_adapter_name)
        
        # For final validation, convert everything to weight_dtype for proper inference
        # For non-final validation, preserve training dtypes to avoid breaking training state
        if is_final_validation:
            pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
            # Keep VAE in float32 for stable decoding
            pipeline.vae = pipeline.vae.to(dtype=torch.float32)
            # Ensure UNet is fully in weight_dtype (LoRA layers may be fp32 from saved weights)
            pipeline.unet = pipeline.unet.to(dtype=weight_dtype)
            if pipeline.controlnet is not None:
                pipeline.controlnet = pipeline.controlnet.to(dtype=weight_dtype)
        else:
            # Move to device only, preserve dtypes for training continuity
            pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        if args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

        validation_prompts = [args.train_prompt]
        validation_lora_images = args.validation_image if (args.train_controllora or args.use_controllora) else None
        validation_controlnet_images = args.controlnet_validation_image if (args.train_controlnet or args.use_controlnet) else None

        final_validation_images = None
        if args.final_validation_image is not None:
            final_validation_images = []
            for img_path in args.final_validation_image:
                if isinstance(img_path, str):
                    final_validation_images.append(Image.open(img_path).convert("RGB"))
                else:
                    final_validation_images.append(img_path)
            logger.info(f"Loaded {len(final_validation_images)} final images for masking")
        else:
            logger.info("No final validation images: skipping masking")

        height = args.resolution
        width = args.resolution

        
        if validation_lora_images is not None and isinstance(validation_lora_images, list):
            if len(validation_lora_images) == 0:
                raise ValueError("Empty validation_lora_images list")
                
            if all(isinstance(img, str) for img in validation_lora_images):
                validation_lora_images = [Image.open(img).convert("RGB") for img in validation_lora_images]
            
            validation_lora_images = prepare_validation_images(validation_lora_images)
            
            if len(validation_lora_images) < len(validation_prompts):
                raise ValueError(
                    f"Not enough previous frames. Got {len(validation_lora_images)} frames "
                    f"but have {len(validation_prompts)} prompts"
                )
                
        
        has_controlnet_images = validation_controlnet_images is not None and (
            isinstance(validation_controlnet_images, list) and len(validation_controlnet_images) > 0 or 
            isinstance(validation_controlnet_images, torch.Tensor) and validation_controlnet_images.numel() > 0
        )
        
        gbuffer_tensor = None
        gbuffer_tensors_per_prompt = []

        if has_controlnet_images:
            if isinstance(validation_controlnet_images, list) and len(validation_controlnet_images) == 5:
                if all(isinstance(img, str) for img in validation_controlnet_images):
                    validation_controlnet_images = [Image.open(img).convert("RGB") for img in validation_controlnet_images]
                
                
                for prompt_idx in range(len(validation_prompts)):
                    final_img = None
                    if final_validation_images and prompt_idx < len(final_validation_images):
                        final_img = final_validation_images[prompt_idx]
                    else:
                        import numpy as np
                        placeholder = np.ones((args.resolution, args.resolution, 3), dtype=np.uint8) * 128
                        noise = np.random.randint(-30, 30, size=(args.resolution, args.resolution, 3))
                        placeholder = np.clip(placeholder + noise, 0, 255).astype(np.uint8)
                        final_img = Image.fromarray(placeholder, mode='RGB')
                    
                    prev_final_img = None
                    prev_albedo_img = None
                    
                    if hasattr(args, 'validation_prev_final_image') and args.validation_prev_final_image:
                        if prompt_idx < len(args.validation_prev_final_image):
                            prev_final_img = Image.open(args.validation_prev_final_image[prompt_idx]).convert("RGB")
                    
                    if hasattr(args, 'validation_prev_albedo_image') and args.validation_prev_albedo_image:
                        if prompt_idx < len(args.validation_prev_albedo_image):
                            prev_albedo_img = Image.open(args.validation_prev_albedo_image[prompt_idx]).convert("RGB")
                    
                    try:
                        gbuffer_tensor = prepare_validation_gbuffer(
                            validation_controlnet_images, 
                            target_size=(args.resolution, args.resolution),
                            final_image=final_img,
                            prev_final_image=prev_final_img,
                            prev_albedo_image=prev_albedo_img,
                            irradiance_smoothing_iterations=0
                        )
                        
                        if gbuffer_tensor.dim() == 3:
                            gbuffer_tensor = gbuffer_tensor.unsqueeze(0)
                        
                        gbuffer_tensors_per_prompt.append(gbuffer_tensor)
                    except Exception as e:
                        logger.error(f"Failed to create G-buffer for prompt {prompt_idx}: {str(e)}")
                        gbuffer_tensors_per_prompt.append(None)
                        
            elif isinstance(validation_controlnet_images, torch.Tensor):
                try:
                    if validation_controlnet_images.dim() == 3:
                        validate_gbuffer_tensor(validation_controlnet_images)
                        gbuffer_tensor = validation_controlnet_images.unsqueeze(0)
                    elif validation_controlnet_images.dim() == 4:
                        validate_gbuffer_tensor(validation_controlnet_images)
                        gbuffer_tensor = validation_controlnet_images
                    else:
                        raise ValueError(f"G-buffer has unexpected shape: {validation_controlnet_images.shape}")
                    
                    for _ in range(len(validation_prompts)):
                        gbuffer_tensors_per_prompt.append(gbuffer_tensor)
                        
                except Exception as e:
                    logger.error(f"Invalid G-buffer: {str(e)}")
                    gbuffer_tensor = None

        image_logs = []
        # Always use autocast for inference to handle dtype conversion automatically
        inference_ctx = torch.autocast("cuda") if torch.cuda.is_available() else contextlib.nullcontext()

        for i, validation_prompt in enumerate(validation_prompts):
            lora_validation_images = None
            controlnet_validation_images = None
            
            if (args.train_controllora or args.use_controllora) and validation_lora_images is not None:
                lora_image_idx = min(i, len(validation_lora_images) - 1)
                prev_frame_image = validation_lora_images[lora_image_idx]
                lora_validation_images = [prev_frame_image]
                
            if has_controlnet_images and len(gbuffer_tensors_per_prompt) > i and gbuffer_tensors_per_prompt[i] is not None:
                controlnet_validation_images = gbuffer_tensors_per_prompt[i]
            
            images = []

            try:
                with inference_ctx:
                    if (args.train_controllora or args.use_controllora) and (args.train_controlnet or args.use_controlnet) and lora_validation_images and controlnet_validation_images is not None:
                        if lora_validation_images is not None and len(lora_validation_images) < i + 1:
                            raise ValueError(f"Not enough previous frames for prompt {i+1}")
                        
                        if isinstance(controlnet_validation_images, torch.Tensor):
                            if controlnet_validation_images.dim() == 3:
                                controlnet_validation_images = controlnet_validation_images.unsqueeze(0)
                                
                            gbuffer_tensor = controlnet_validation_images.to(device=accelerator.device, dtype=weight_dtype)
                            
                            
                            image = pipeline(
                                validation_prompt, 
                                lora_images=lora_validation_images,
                                controlnet_images=gbuffer_tensor,
                                num_inference_steps=args.generated_conditioning_steps,
                                num_images_per_prompt=1, 
                                generator=generator,
                                guidance_scale=1.0,
                                height=height,  
                                width=width,    
                                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                            ).images[0]
                        else:
                            image = pipeline(
                                validation_prompt, 
                                lora_images=lora_validation_images,
                                controlnet_images=controlnet_validation_images,
                                num_inference_steps=args.generated_conditioning_steps,
                                num_images_per_prompt=1,  
                                generator=generator,
                                guidance_scale=1.0,
                                height=height,  
                                width=width,    
                                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                            ).images[0]
                    elif (args.train_controllora or args.use_controllora) and lora_validation_images:
                        image = pipeline(
                            validation_prompt, 
                            lora_images=lora_validation_images,
                            num_inference_steps=args.generated_conditioning_steps,
                            num_images_per_prompt=1,
                            generator=generator,
                            guidance_scale=1.0,
                            height=height,  
                            width=width,    
                        ).images[0]
                    elif (args.train_controlnet or args.use_controlnet) and controlnet_validation_images is not None:
                        
                        if isinstance(controlnet_validation_images, torch.Tensor):
                            if controlnet_validation_images.dim() == 3:
                                controlnet_validation_images = controlnet_validation_images.unsqueeze(0)
                                
                            gbuffer_tensor = controlnet_validation_images.to(device=accelerator.device, dtype=weight_dtype)
                            
                            image = pipeline(
                                validation_prompt, 
                                controlnet_images=gbuffer_tensor,
                                num_inference_steps=args.generated_conditioning_steps,
                                num_images_per_prompt=1,  
                                generator=generator,
                                guidance_scale=1.0,
                                height=height,  
                                width=width,    
                                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                            ).images[0]
                        else:
                            image = pipeline(
                                validation_prompt, 
                                controlnet_images=controlnet_validation_images,
                                num_inference_steps=args.generated_conditioning_steps,
                                num_images_per_prompt=1,
                                generator=generator,
                                guidance_scale=1.0,
                                height=height,  
                                width=width,    
                                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                            ).images[0]
                images.append(image)
            except Exception as e:
                logger.error(f"Error during validation: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Create simple error placeholder image
                error_img = Image.new('RGB', (512, 512), color=(128, 0, 0))
                images.append(error_img)

            if (args.train_controlnet or args.use_controlnet):
                if controlnet_validation_images is None:
                    raise ValueError(f"ControlNet in use but no controlnet_validation_images for prompt {i}!")
            
            image_logs.append({
                "validation_prompt": validation_prompt,
                "lora_validation_image": lora_validation_images,
                "controlnet_validation_image": controlnet_validation_images,
                "final_validation_image": final_validation_images[i] if final_validation_images and i < len(final_validation_images) else None,
                "images": images
            })

        if current_time - _last_validation_img_log_time > 30.0:
            logger.info(f"Generated {len(image_logs)} validation images")

        tracker_key = "test" if is_final_validation else "validation"
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                from PIL import ImageFont
                
                def add_label_to_image(img_array, label, font_size=16):
                    """Add a text label to the top-left corner of an image."""
                    # Convert to uint8 if needed
                    if img_array.max() <= 1.0:
                        img_uint8 = (img_array * 255).astype(np.uint8)
                    else:
                        img_uint8 = img_array.astype(np.uint8)
                    
                    # Handle grayscale
                    if len(img_uint8.shape) == 2:
                        img_uint8 = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)
                    
                    img_pil = Image.fromarray(img_uint8)
                    draw = ImageDraw.Draw(img_pil)
                    
                    # Use default font
                    try:
                        # Try common font paths
                        font_paths = [
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                            "C:/Windows/Fonts/arial.ttf",  # Windows
                            "/System/Library/Fonts/Helvetica.ttc",  # macOS
                        ]
                        font = None
                        for fp in font_paths:
                            if os.path.exists(fp):
                                font = ImageFont.truetype(fp, font_size)
                                break
                        if font is None:
                            font = ImageFont.load_default()
                    except:
                        font = ImageFont.load_default()
                    
                    # Draw black background for text
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    draw.rectangle([0, 0, text_width + 8, text_height + 8], fill=(0, 0, 0))
                    draw.text((4, 4), label, fill=(255, 255, 255), font=font)
                    
                    return np.asarray(img_pil) / 255.0
                
                for log in image_logs:
                    images = log["images"]
                    validation_prompt = log["validation_prompt"]
                    lora_validation_image = log["lora_validation_image"]
                    controlnet_validation_image = log["controlnet_validation_image"]
                    final_validation_image = log.get("final_validation_image")

                    formatted_images = []
                    image_labels = []
                    
                    # Add GT (ground truth) if available
                    if final_validation_image is not None:
                        if isinstance(final_validation_image, Image.Image):
                            gt_img = final_validation_image.resize((args.resolution, args.resolution), Image.LANCZOS)
                            gt_img = np.asarray(gt_img)
                        else:
                            gt_img = np.asarray(final_validation_image)
                        if gt_img.max() > 1.0:
                            gt_img = gt_img / 255.0
                        formatted_images.append(gt_img)
                        image_labels.append("GT")
                    
                    # Add ControlLoRA input (previous frame)
                    if lora_validation_image is not None:
                        for img in lora_validation_image[:1]:
                            if isinstance(img, torch.Tensor):
                                img = img.cpu().detach().numpy()
                                if len(img.shape) == 3 and img.shape[0] == 3:
                                    img = np.transpose(img, (1, 2, 0))
                                img = (img * 0.5 + 0.5).clip(0, 1)
                            elif isinstance(img, Image.Image):
                                if img.size != (args.resolution, args.resolution):
                                    img = img.resize((args.resolution, args.resolution), Image.LANCZOS)
                                img = np.asarray(img)
                                if img.max() > 1.0:
                                    img = img / 255.0
                            formatted_images.append(np.asarray(img))
                            image_labels.append("Prev Frame")
                    
                    # Add G-buffer components
                    if (args.train_controlnet or args.use_controlnet) and controlnet_validation_image is not None:
                        if isinstance(controlnet_validation_image, torch.Tensor):
                            if controlnet_validation_image.dim() == 3 and controlnet_validation_image.shape[0] == 10:
                                tensor = controlnet_validation_image.cpu().detach()
                            elif controlnet_validation_image.dim() == 4 and controlnet_validation_image.shape[1] == 10:
                                tensor = controlnet_validation_image[0].cpu().detach()
                            else:
                                continue
                            
                            components = [
                                (tensor[:3], "BaseColor"),
                                (tensor[3:6], "Normals"),
                                (tensor[6:7], "Depth"),
                                (tensor[7:8], "Roughness"),
                                (tensor[8:9], "Metallic"),
                                (tensor[9:10], "Irradiance"),
                            ]
                            
                            for comp, name in components:
                                if comp.shape[0] == 3:
                                    comp_np = comp.numpy()
                                    comp_np = np.transpose(comp_np, (1, 2, 0))
                                    comp_np = (comp_np * 0.5 + 0.5).clip(0, 1)
                                else:
                                    comp_np = comp.numpy()[0]
                                    comp_np = (comp_np * 0.5 + 0.5).clip(0, 1)
                                
                                formatted_images.append(comp_np)
                                image_labels.append(name)

                    # Add output images
                    for image in images:
                        if isinstance(image, Image.Image):
                            img_np = np.asarray(image)
                        else:
                            img_np = image.cpu().detach().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
                            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                                img_np = np.transpose(img_np, (1, 2, 0))
                            if img_np.max() <= 1.0 and img_np.min() < 0:
                                img_np = (img_np * 0.5 + 0.5).clip(0, 1)
                        if img_np.max() > 1.0:
                            img_np = img_np / 255.0
                        formatted_images.append(img_np)
                        image_labels.append("Output")

                    try:
                        # Add labels and stack images
                        labeled_images = []
                        for img, label in zip(formatted_images, image_labels):
                            if len(img.shape) == 2:
                                img = np.stack([img, img, img], axis=-1)
                            labeled_img = add_label_to_image(img, label)
                            labeled_images.append(labeled_img)
                        
                        stacked_images = np.stack(labeled_images)
                        tracker.writer.add_images(validation_prompt, stacked_images, step, dataformats="NHWC")
                        
                    except Exception as e:
                        logger.error(f"Error logging images to tensorboard: {str(e)}")
                        logger.error(f"Image shapes: {[img.shape for img in formatted_images if hasattr(img, 'shape')]}")
                        
            elif tracker.name == "wandb":
                formatted_images = []

                for log in image_logs:
                    images = log["images"]
                    validation_prompt = log["validation_prompt"]
                    lora_validation_image = log["lora_validation_image"]
                    controlnet_validation_image = log["controlnet_validation_image"]

                    # Define captions for validation components
                    captions = ["PREV FRAME (t-1)", "BaseColor (Masked)", "Normals", 
                            "Depth", "Roughness", "Metallic", "Generated Output (t)"]
                    caption_idx = 0
                    
                    # Add input images with appropriate captions
                    if lora_validation_image is not None:
                        for i, img in enumerate(lora_validation_image[:1]):  # Show only first previous frame
                            # Process tensor if needed
                            if isinstance(img, torch.Tensor):
                                img = img.cpu().detach().numpy()
                                if len(img.shape) == 3 and img.shape[0] == 3:  # CHW format
                                    img = np.transpose(img, (1, 2, 0))
                                img = (img * 0.5 + 0.5).clip(0, 1)
                                img = Image.fromarray((img * 255).astype(np.uint8))
                            elif isinstance(img, Image.Image):
                                # Resize PIL images to match output resolution
                                if img.size != (args.resolution, args.resolution):
                                    img = img.resize((args.resolution, args.resolution), Image.LANCZOS)
                                
                            formatted_images.append(wandb.Image(img, caption=captions[caption_idx]))
                            caption_idx += 1
                    
                    # Handle ControlNet validation images for wandb
                    if (args.train_controlnet or args.use_controlnet):
                        if controlnet_validation_image is None:
                            raise ValueError(
                                f"ControlNet is being trained/used but controlnet_validation_image is None! "
                                f"You must provide proper G-buffer validation images via --controlnet_validation_image"
                            )
                        
                        validation_images_to_use = controlnet_validation_image
                        if validation_images_to_use is not None:
                            if isinstance(validation_images_to_use, torch.Tensor):
                                # Handle multi-dimensional tensor formats
                                if validation_images_to_use.dim() == 3 and validation_images_to_use.shape[0] == 10:  # Changed from 9 to 10
                                    # 3D tensor with 10 channels [C, H, W]
                                    component_tensors = [
                                        validation_images_to_use[:3],  # BaseColor
                                        validation_images_to_use[3:6],  # Normals
                                        validation_images_to_use[6:7],  # Depth
                                        validation_images_to_use[7:8],  # Roughness
                                        validation_images_to_use[8:9],  # Metallic
                                        validation_images_to_use[9:10], # Irradiance
                                    ]
                                elif validation_images_to_use.dim() == 4 and validation_images_to_use.shape[1] == 10:  # Changed from 9 to 10
                                    # 4D tensor with format [B, C, H, W]
                                    tensor = validation_images_to_use[0]
                                    component_tensors = [
                                        tensor[:3],  # BaseColor
                                        tensor[3:6],  # Normals
                                        tensor[6:7],  # Depth
                                        tensor[7:8],  # Roughness
                                        tensor[8:9],  # Metallic
                                        tensor[9:10], # Irradiance
                                    ]
                                else:
                                    raise ValueError(
                                        f"ControlNet validation tensor has invalid shape {validation_images_to_use.shape}! "
                                        f"Expected 10-channel G-buffer tensor with shape [10,H,W] or [B,10,H,W]"  # Changed from 9 to 10
                                    )
                                    
                                # Process each component
                                for i, comp_tensor in enumerate(component_tensors):
                                    if i < len(captions) - 1: 
                                        # Convert tensor to numpy and denormalize
                                        comp_np = comp_tensor.cpu().detach().numpy()
                                        
                                        if comp_tensor.shape[0] == 3:  # RGB components
                                            comp_np = np.transpose(comp_np, (1, 2, 0))
                                            comp_np = (comp_np * 0.5 + 0.5).clip(0, 1)
                                            comp_img = Image.fromarray((comp_np * 255).astype(np.uint8))
                                        else:  # Single channel components
                                            comp_np = comp_np[0]  # Take first channel
                                            comp_np = (comp_np * 0.5 + 0.5).clip(0, 1)
                                            # Convert to RGB by duplicating channels
                                            comp_img = Image.fromarray((comp_np * 255).astype(np.uint8)).convert("RGB")
                                        
                                        formatted_images.append(wandb.Image(comp_img, caption=captions[caption_idx]))
                                        caption_idx += 1
                            elif isinstance(validation_images_to_use, list):
                                if len(validation_images_to_use) < 5:
                                    raise ValueError(
                                        f"ControlNet validation image list has {len(validation_images_to_use)} items! "
                                        f"Expected exactly 5 G-buffer components (BaseColor, Normals, Depth, Roughness, Metallic)"
                                    )
                                
                                # Handle list of images
                                for i, img in enumerate(validation_images_to_use[:5]):
                                    if i + 1 < len(captions):  # Skip last caption (reserved for output)
                                        # Process tensor if needed
                                        if isinstance(img, torch.Tensor):
                                            img = img.cpu().detach().numpy()
                                            if len(img.shape) == 3 and img.shape[0] == 3:  # RGB format
                                                img = np.transpose(img, (1, 2, 0))
                                            elif len(img.shape) == 3 and img.shape[0] == 1:  # Single channel
                                                img = img[0]  # Take first channel
                                            
                                            # Denormalize
                                            img = (img * 0.5 + 0.5).clip(0, 1)
                                            
                                            # Convert to proper image format
                                            if len(img.shape) == 2:  # Grayscale
                                                img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
                                            else:  # RGB
                                                img = Image.fromarray((img * 255).astype(np.uint8))
                                        elif isinstance(img, Image.Image):
                                            # Resize PIL images to match output resolution
                                            if img.size != (args.resolution, args.resolution):
                                                img = img.resize((args.resolution, args.resolution), Image.LANCZOS)
                                        
                                        formatted_images.append(wandb.Image(img, caption=captions[caption_idx]))
                                        caption_idx += 1
                            else:
                                raise ValueError(
                                    f"ControlNet validation image has invalid type {type(validation_images_to_use)}! "
                                    f"Expected torch.Tensor or list of PIL Images"
                                )
                        else:
                            # This should never happen due to the earlier check, but just in case
                            raise ValueError(
                                f"validation_images_to_use is None despite passing earlier validation! "
                                f"This indicates a serious bug in the validation logic."
                            )

                    # Add output images (last caption)
                    for image in images:
                        formatted_images.append(wandb.Image(image, caption=captions[-1]))

                tracker.log({tracker_key: formatted_images})

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if hasattr(args, 'stored_num_validation_images'):
            args.num_validation_images = args.stored_num_validation_images
            delattr(args, 'stored_num_validation_images')
        
        gc.collect()
    finally:
        if not is_final_validation:
            if unet_was_training:
                unet.train()
            if vae_was_training:
                vae.train()
            if text_encoder_was_training:
                text_encoder.train()
            if controlnet is not None and controlnet_was_training:
                controlnet.train()

    return image_logs

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            lora_validation_image = log["lora_validation_image"]
            controlnet_validation_image = log["controlnet_validation_image"]
            
            img_str += f"prompt: {validation_prompt}\n"
            
            all_images = []
            # Handle lists of conditioning images properly
            if lora_validation_image is not None:
                for j, img in enumerate(lora_validation_image):
                    img_type = "prev_frame" 
                    img.save(os.path.join(repo_folder, f"lora_{img_type}_{i}.png"))
                    all_images.append(img)
                
            if controlnet_validation_image is not None:
                gbuffer_types = ["basecolor", "normals", "depth", "roughness", "metallic"]
                if isinstance(controlnet_validation_image, torch.Tensor):
                    # Handle tensor case differently
                    if controlnet_validation_image.dim() == 3 and controlnet_validation_image.shape[0] == 9:
                        # Extract individual components
                        components = [
                            controlnet_validation_image[:3],  # BaseColor
                            controlnet_validation_image[3:6],  # Normals
                            controlnet_validation_image[6:7],  # Depth
                            controlnet_validation_image[7:8],  # Roughness
                            controlnet_validation_image[8:9],  # Metallic
                        ]
                        
                        # Convert to PIL and save
                        for j, (comp, comp_name) in enumerate(zip(components, gbuffer_types)):
                            tensor = comp.cpu()
                            if tensor.shape[0] == 3:  # RGB
                                img_np = tensor.permute(1, 2, 0).numpy()
                                img_np = (img_np * 0.5 + 0.5).clip(0, 1)
                                img = Image.fromarray((img_np * 255).astype(np.uint8))
                            else:  # Single channel
                                img_np = tensor[0].numpy()
                                img_np = (img_np * 0.5 + 0.5).clip(0, 1)
                                img = Image.fromarray((img_np * 255).astype(np.uint8)).convert("RGB")
                            
                            img.save(os.path.join(repo_folder, f"controlnet_{comp_name}_{i}.png"))
                            all_images.append(img)
                    elif controlnet_validation_image.dim() == 4 and controlnet_validation_image.shape[1] == 9:
                        # Handle 4D tensor
                        tensor = controlnet_validation_image[0]
                        components = [
                            tensor[:3],  # BaseColor
                            tensor[3:6],  # Normals
                            tensor[6:7],  # Depth
                            tensor[7:8],  # Roughness
                            tensor[8:9],  # Metallic
                        ]
                        
                        # Convert to PIL and save
                        for j, (comp, comp_name) in enumerate(zip(components, gbuffer_types)):
                            tensor = comp.cpu()
                            if tensor.shape[0] == 3:  # RGB
                                img_np = tensor.permute(1, 2, 0).numpy()
                                img_np = (img_np * 0.5 + 0.5).clip(0, 1)
                                img = Image.fromarray((img_np * 255).astype(np.uint8))
                            else:  # Single channel
                                img_np = tensor[0].numpy()
                                img_np = (img_np * 0.5 + 0.5).clip(0, 1)
                                img = Image.fromarray((img_np * 255).astype(np.uint8)).convert("RGB")
                            

                            img.save(os.path.join(repo_folder, f"controlnet_{comp_name}_{i}.png"))
                            all_images.append(img)
                else:
                    # List of images case
                    for j, img in enumerate(controlnet_validation_image[:5]):
                        if j < len(gbuffer_types):
                            img_type = gbuffer_types[j]
                        else:
                            img_type = f"input_{j}"
                        img.save(os.path.join(repo_folder, f"controlnet_{img_type}_{i}.png"))
                        all_images.append(img)
                
            all_images.extend(images)
            
            image_grid(all_images, 1, len(all_images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    conditioning_type = []
    if hasattr(args, 'train_controllora') and args.train_controllora:
        conditioning_type.append("ControlLoRA")
    if hasattr(args, 'train_controlnet') and args.train_controlnet:
        conditioning_type.append("ControlNet")
    
    conditioning_str = " and ".join(conditioning_type)

    model_description = f"""
# Dual Control Model - {repo_id.split("/")[-1] if "/" in repo_id else repo_id}

These are {conditioning_str} weights trained on {base_model}.
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "controlnet",
        "control-lora-v3",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training script for dual control (ControlNet + ControlLoRA) models.", 
                                     epilog="""
    Examples:
    # Train ControlNet only
    python train.py --train_controlnet ...
    
    # Use trained ControlNet (frozen) and train ControlLoRA
    python train.py --train_controllora --use_controlnet --controlnet_model_name_or_path path/to/trained/controlnet ...
    
    # Train both ControlNet and ControlLoRA together
    python train.py --train_controllora --train_controlnet ...
    """)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_prompt",
        type=str,
        default=None,
        help="Prompt to use for all training samples. Required.",
    )
    parser.add_argument(
        "--gbuffer_root_dir",
        type=str,
        default="./data/train",
        help="Root directory for G-buffer dataset (default: ./data/train)",
    )
    parser.add_argument(
        "--exclude_sequence_prefix",
        type=str,
        default=None,
        help="Exclude sequences starting with this prefix from training (e.g., 'Val' to exclude validation sequences)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
        ),
    )
    parser.add_argument(
        "--use_black_irradiance",
        action="store_true",
        help="Use black/neutral irradiance instead of computed irradiance in ControlNet G-buffer",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--generated_chaining_ratio",
        type=float,
        default=0.1,
        help="Ratio of generated images that should use previously generated images as input (default: 0.1 = 10%)",
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    # ControlLoRA args
    parser.add_argument(
        "--train_controllora",
        action="store_true",
        help="Whether to train the ControlLoRA model.",
    )
    parser.add_argument(
        "--use_controllora",
        action="store_true",
        help="Whether to use but not train the ControlLoRA model (loads it and keeps it frozen).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--load_lora_weights",
        type=str,
        default=None,
        help="Directory containing trained LoRA weights to load",
    )
    # ControlNet args
    parser.add_argument(
        "--train_controlnet",
        action="store_true",
        help="Whether to train the ControlNet model.",
    )
    parser.add_argument(
        "--use_controlnet",
        action="store_true",
        help="Whether to use but not train the ControlNet model (loads it and keeps it frozen).",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to the controlnet model to use. If not provided, controlnet will be initialized from unet.",
    )
    parser.add_argument(
        "--controlnet_learning_rate",
        type=float,
        default=None,
        help="Learning rate for ControlNet. If not specified, the main learning rate will be used.",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale", 
        type=float, 
        default=1.0,
        help="The scale for controlnet conditioning during inference.",
    )
    parser.add_argument(
        "--gbuffer_channels",
        type=int,
        default=10,
        help="Number of G-buffer channels (default: 10 = BaseColor[3] + Normals[3] + Depth[1] + Roughness[1] + Metallic[1] + Irradiance[1])"
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--validation_folder",
        type=str,
        default=None,
        help=(
            "Path to a folder containing validation data. Auto-detects G-buffer components "
            "and picks a frame with a valid previous frame. Supports all naming conventions."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each validation prompt/image pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps using --train_prompt.",
    )
    parser.add_argument(
        "--max_prev_frames",
        type=int,
        default=2,
        help="Maximum number of previous frames to use for temporal conditioning in GBufferDataset",
    )
    parser.add_argument(
        "--use_generated_conditioning",
        action="store_true",
        help="Use model-generated images as conditioning inputs periodically during training",
    )
    parser.add_argument(
        "--generated_conditioning_freq",
        type=int,
        default=1000,
        help="Generate new conditioning images every N steps",
    )
    parser.add_argument(
        "--generated_conditioning_count",
        type=int,
        default=1000,
        help="Number of images to generate for conditioning",
    )
    parser.add_argument(
        "--generated_conditioning_batch_size",
        type=int,
        default=4,
        help="Batch size for generating conditioning images",
    )
    parser.add_argument(
        "--generated_conditioning_dir",
        type=str,
        default="generated_conditioning",
        help="Directory to store generated conditioning images",
    )
    parser.add_argument(
        "--generated_sample_ratio",
        type=float,
        default=0.25,
        help="Ratio of samples in each batch that use generated conditioning (e.g., 0.25 = 1 sample in a batch of 4)",
    )
    parser.add_argument(
        "--generated_conditioning_start_step",
        type=int,
        default=10000,
        help="Step to start using generated conditioning images (default: 0, start immediately)",
    )
    parser.add_argument(
        "--generated_conditioning_steps",
        type=int,
        default=40,
        help="Number of denoising steps to use when generating conditioning images (default: 5)",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Set hardcoded values for removed arguments
    args.lora_adapter_name = "default"
    args.half_or_full_lora = "full_skip_attn"
    args.revision = None
    args.variant = None
    args.tokenizer_name = None
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_epsilon = 1e-8
    args.lr_num_cycles = 1
    args.lr_power = 1.0
    args.tracker_project_name = "train_dual_control"
    args.push_to_hub = False
    args.hub_token = None
    args.hub_model_id = None
    args.skip_final_validation = False
    args.scale_lr = False
    args.use_8bit_adam = False

    if not args.train_controllora and not args.train_controlnet:
        raise ValueError("At least one of `--train_controllora` or `--train_controlnet` must be specified")
        
    if args.train_controlnet and args.use_controlnet:
        raise ValueError("Cannot specify both `--train_controlnet` and `--use_controlnet` - choose one mode for ControlNet")
        
    if args.train_controllora and args.use_controllora:
        raise ValueError("Cannot specify both `--train_controllora` and `--use_controllora` - choose one mode for ControlLoRA")

    # Auto-detect validation data from --validation_folder or ./data/validation
    validation_folder = args.validation_folder if args.validation_folder else "./data/validation"
    
    # Initialize validation attributes (will be populated by auto-detection)
    args.controlnet_validation_image = None
    args.validation_image = None
    args.final_validation_image = None
    args.validation_prev_final_image = None
    args.validation_prev_albedo_image = None
    
    if os.path.isdir(validation_folder):
        detected = auto_detect_validation_from_folder(validation_folder)
        if detected:
            args.controlnet_validation_image = detected['controlnet_validation_image']
            if detected['validation_image']:
                args.validation_image = [detected['validation_image']]
            if detected['final_validation_image']:
                args.final_validation_image = [detected['final_validation_image']]
            if detected['validation_prev_final_image']:
                args.validation_prev_final_image = [detected['validation_prev_final_image']]
            if detected['validation_prev_albedo_image']:
                args.validation_prev_albedo_image = [detected['validation_prev_albedo_image']]

    # Validate train_prompt is set
    if args.train_prompt is None:
        raise ValueError("`--train_prompt` is required. Please specify the prompt to use for training.")

    # Validate validation inputs (auto-detected from --validation_folder or ./data/validation)
    if args.train_controllora and (args.validation_image is None or len(args.validation_image) == 0):
        raise ValueError(
            "No validation image found. Ensure --validation_folder points to a directory with G-buffer data, "
            "or place validation data in ./data/validation"
        )
        
    if (args.train_controlnet or args.use_controlnet) and args.controlnet_validation_image is None:
        raise ValueError(
            "No G-buffer validation images found. Ensure --validation_folder points to a directory with G-buffer data, "
            "or place validation data in ./data/validation"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the encoders."
        )

    return args


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, accelerator):
        self.args = args
        self.tokenizer = tokenizer
        
        # Initialize GBufferDataset directly
        logger.info(f"Initializing GBufferDataset from {args.gbuffer_root_dir}")
        self.dataset = GBufferDataset(args=args, tokenizer=tokenizer)
        logger.info(f"GBufferDataset initialized successfully with {len(self.dataset)} samples")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]



def collate_fn(examples):
    """
    Optimized collate function with minimized CPU-GPU transfers
    """
    output_dict = {}
    
    # Process all tensors at once
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.channels_last).float()  # Use channels_last for better performance
    output_dict["pixel_values"] = pixel_values
    
    # Always keep input_ids as integers (Long)
    input_ids = torch.stack([example["input_ids"] for example in examples])
    input_ids = input_ids.long()  # Ensure it's the correct type
    output_dict["input_ids"] = input_ids
    
    # Include previous frames (ControlLoRA conditioning) with the same optimization
    if "conditioning_pixel_values" in examples[0]:
        prev_frames = torch.stack([example["conditioning_pixel_values"] for example in examples])
        prev_frames = prev_frames.to(memory_format=torch.channels_last).float()
        output_dict["conditioning_pixel_values"] = prev_frames
    
    if "controlnet_pixel_values" in examples[0]:
        controlnet_pixel_values = torch.stack([example["controlnet_pixel_values"] for example in examples])
        controlnet_pixel_values = controlnet_pixel_values.to(memory_format=torch.channels_last).float()
        output_dict["controlnet_pixel_values"] = controlnet_pixel_values
    
    if "uses_generated_conditioning" in examples[0]:
        # Create a boolean tensor from the flags
        uses_generated = torch.tensor(
            [example["uses_generated_conditioning"] for example in examples], 
            dtype=torch.bool
        )
        output_dict["uses_generated_conditioning"] = uses_generated

    return output_dict


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    
    # Check tensorboard availability
    if args.report_to == "tensorboard":
        try:
            import tensorboard
        except ImportError:
            raise ImportError(
                "--report_to=tensorboard requires tensorboard. Install with: pip install tensorboard"
            )
    
    hub_token = args.hub_token or get_token()

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.WARNING,
    )
    logger.info(accelerator.state, main_process_only=False)
    # Suppress verbose HuggingFace warnings
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Training with seed: {args.seed}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    
    # Load the UNet
    logger.info("Loading UNet model")
    unet: UNet2DConditionModelEx = UNet2DConditionModelEx.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    
    # Initialize ControlNet model if training it or using it
    controlnet = None
    if args.train_controlnet or args.use_controlnet:
        logger.info("Loading ControlNet model with direct G-buffer input support")
        if args.controlnet_model_name_or_path:
            # Load config first to get cross_attention_dim and other parameters
            config_path = os.path.join(args.controlnet_model_name_or_path, "config.json")
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                cross_attention_dim = config.get('cross_attention_dim', 768)
            except:
                cross_attention_dim = 768
            
            controlnet = ControlNetModelEx.from_pretrained(
                args.controlnet_model_name_or_path,
                cross_attention_dim=cross_attention_dim,
                gbuffer_channels=args.gbuffer_channels
            )
        else:
            # Initialize from UNet weights
            logger.info("Initializing ControlNetEx from UNet weights with direct G-buffer input")
            controlnet = ControlNetModelEx.from_unet(
                unet,
                gbuffer_channels=args.gbuffer_channels
            )
        
        if args.train_controlnet:
            controlnet = prepare_controlnet_for_training(controlnet)
        else:
            controlnet.eval()
            controlnet.requires_grad_(False)

    if args.train_controllora or args.use_controllora:
        unet = unet.add_extra_conditions([args.lora_adapter_name])

        if args.load_lora_weights:
            pipeline = StableDiffusionDualControlPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=None,
                scheduler=noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False
            )
            pipeline.load_lora_weights(args.load_lora_weights, adapter_name=args.lora_adapter_name)
                
            lora_params_count_before = sum(p.numel() for n, p in unet.named_parameters() 
                                        if ".lora_" in n and p.requires_grad)
                
            for name, param in unet.named_parameters():
                if ".lora_" in name:
                    if args.train_controllora:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                
            lora_params_count_after = sum(p.numel() for n, p in unet.named_parameters() 
                                        if ".lora_" in n and p.requires_grad)
            
            unet._pre_wrapped_lora_A_params = [p for n, p in unet.named_parameters() 
                                            if ".lora_A." in n and p.requires_grad]
            unet._pre_wrapped_lora_B_params = [p for n, p in unet.named_parameters() 
                                            if ".lora_B." in n and p.requires_grad]
            
            del pipeline
            torch.cuda.empty_cache()
            gc.collect()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    vae.requires_grad_(False)
    vae.eval()

    unet.requires_grad_(False)
    unet.train()

    text_encoder.requires_grad_(False)
    text_encoder.eval()

    if controlnet is not None:
        if args.train_controlnet:
            controlnet.train()
            for param in controlnet.parameters():
                param.requires_grad = True
        else:
            controlnet.requires_grad_(False)
            controlnet.eval()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    for param in unet.parameters():
        param.requires_grad_(False)

    lora_A_layers = []
    lora_B_layers = []
    
    if args.train_controllora or args.use_controllora:
        adapter_already_loaded = args.load_lora_weights is not None
        
        if not adapter_already_loaded:
            if args.half_or_full_lora == "half":
                lora_target_modules = unet.default_half_lora_target_modules
            elif args.half_or_full_lora == "full":
                lora_target_modules = unet.default_full_lora_target_modules
            elif args.half_or_full_lora == "half_skip_attn":
                lora_target_modules = unet.default_half_skip_attn_lora_target_modules
            elif args.half_or_full_lora == "full_skip_attn":
                lora_target_modules = unet.default_full_skip_attn_lora_target_modules
                
            lora_config = LoraConfig(
                r=args.rank,
                lora_alpha=args.rank,
                init_lora_weights="gaussian",
                target_modules=lora_target_modules,
            )
            
            unet.add_adapter(lora_config, adapter_name=args.lora_adapter_name)
        
        lora_A_layers = [p for n, p in unet.named_parameters() if ".lora_A." in n and p.requires_grad]
        lora_B_layers = [p for n, p in unet.named_parameters() if ".lora_B." in n and p.requires_grad]
    
    if args.gradient_checkpointing:
        unet.disable_gradient_checkpointing()
        if controlnet is not None:
            controlnet.disable_gradient_checkpointing()

    # Move frozen models to weight_dtype (bf16/fp16) to save memory
    # Keep trainable models in fp32 for reliable gradients
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # UNet: if training LoRA, keep in fp32; otherwise use weight_dtype
    if args.train_controllora:
        unet = unet.to(accelerator.device, dtype=torch.float32)
    else:
        unet = unet.to(accelerator.device, dtype=weight_dtype)
    
    # ControlNet: if training, keep in fp32; otherwise use weight_dtype
    if controlnet is not None:
        if args.train_controlnet:
            controlnet = controlnet.to(accelerator.device, dtype=torch.float32)
        else:
            controlnet = controlnet.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_controlnet and controlnet is not None:
            controlnet.enable_gradient_checkpointing()

    # Note: trainable models are already in fp32, no need for cast_training_params

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None:
                controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # Collect models first, then pop all weights
            unet_model = None
            controlnet_model = None
            
            for model in models:
                unwrapped_model = unwrap_model(model)
                
                if isinstance(unwrapped_model, (ControlNetModel, ControlNetModelEx)):
                    controlnet_model = unwrapped_model
                elif isinstance(unwrapped_model, UNet2DConditionModelEx):
                    unet_model = unwrapped_model
                elif isinstance(unwrapped_model, UNet2DConditionModel):
                    # Base UNet when not using ControlLoRA
                    pass
                else:
                    raise ValueError(f"unexpected save model type: {type(unwrapped_model)}")
            
            # Pop all weights and tells accelerate we're handling all model saves
            while len(weights) > 0:
                weights.pop()
            
            # Save ControlLoRA weights
            if args.train_controllora and unet_model is not None:
                if hasattr(unet_model, '_pre_wrapped_lora_A_params'):
                    lora_A_params = unet_model._pre_wrapped_lora_A_params
                    lora_B_params = unet_model._pre_wrapped_lora_B_params
                    total_lora_params = sum(p.numel() for p in lora_A_params) + sum(p.numel() for p in lora_B_params)

                unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unet_model, adapter_name=args.lora_adapter_name)
                )

                lora_a_keys = [k for k in unet_lora_layers_to_save.keys() if ".lora_A." in k]
                lora_b_keys = [k for k in unet_lora_layers_to_save.keys() if ".lora_B." in k]

                if hasattr(unet_model, '_pre_wrapped_lora_A_params'):
                    expected_layers = len(unet_model._pre_wrapped_lora_A_params) + len(unet_model._pre_wrapped_lora_B_params)
                    actual_layers = len(lora_a_keys) + len(lora_b_keys)
                    
                    if expected_layers != actual_layers:
                        logger.warning(f"Layer count mismatch! Expected {expected_layers}, saving {actual_layers}")
                    
                StableDiffusionDualControlPipeline.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    safe_serialization=True
                )
            
            # Save ControlNet weights
            if args.train_controlnet and controlnet_model is not None:
                controlnet_dir = os.path.join(output_dir, "controlnet")
                os.makedirs(controlnet_dir, exist_ok=True)
                controlnet_model.save_pretrained(controlnet_dir)

    def load_model_hook(models, input_dir):
        # Pop all models from list, tells accelerate we're handling them
        # Must pop all models, otherwise accelerate will try to load remaining ones
        unet_model = None
        controlnet_model = None
        
        while len(models) > 0:
            model = models.pop()
            unwrapped_model = unwrap_model(model)
            
            if isinstance(unwrapped_model, (ControlNetModel, ControlNetModelEx)):
                controlnet_model = unwrapped_model
            elif isinstance(unwrapped_model, (UNet2DConditionModel, UNet2DConditionModelEx)):
                unet_model = unwrapped_model
            else:
                raise ValueError(f"unexpected load model type: {type(unwrapped_model)}")
        
        # Now load the models we care about
        if args.train_controllora and unet_model is not None:
            lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(input_dir)
            unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
            unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
            
            lora_a_keys = [k for k in unet_state_dict.keys() if ".lora_A." in k]
            lora_b_keys = [k for k in unet_state_dict.keys() if ".lora_B." in k]
            total_params = sum(v.numel() for v in unet_state_dict.values())

            incompatible_keys = set_peft_model_state_dict(unet_model, unet_state_dict, adapter_name=args.lora_adapter_name)
            if incompatible_keys is not None:
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter found {len(unexpected_keys)} unexpected keys: {unexpected_keys[:5]}..."
                    )
                
                missing_keys = getattr(incompatible_keys, "missing_keys", None)
                if missing_keys:
                    # Error on missing LoRA keys
                    lora_missing = [k for k in missing_keys if ".lora_A." in k or ".lora_B." in k]
                    
                    if lora_missing:
                        attn_missing = sum(1 for k in lora_missing if any(attn in k for attn in ["to_k", "to_q", "to_v", "to_out"]))
                        gbuffer_missing = sum(1 for k in lora_missing if ".gbuffer." in k)
                        
                        error_msg = (
                            f"ERROR: {len(lora_missing)} LoRA keys not loaded. "
                            f"Includes {attn_missing} attention keys and {gbuffer_missing} gbuffer keys. "
                            f"First few: {lora_missing[:10]}\n"
                            f"Current config: --half_or_full_lora={args.half_or_full_lora}\n"
                            f"Ensure consistent LoRA configuration across training stages."
                        )
                        raise ValueError(error_msg)
            
            for name, param in unet_model.named_parameters():
                if ".lora_" in name:
                    param.requires_grad = True
            
            unet_model._pre_wrapped_lora_A_params = [p for n, p in unet_model.named_parameters() 
                                                    if ".lora_A." in n and p.requires_grad]
            unet_model._pre_wrapped_lora_B_params = [p for n, p in unet_model.named_parameters() 
                                                    if ".lora_B." in n and p.requires_grad]

            stored_params = len(unet_model._pre_wrapped_lora_A_params) + len(unet_model._pre_wrapped_lora_B_params) 

            if stored_params != len(lora_a_keys) + len(lora_b_keys):
                logger.warning(f"Parameter count mismatch! State dict has {len(lora_a_keys) + len(lora_b_keys)}, stored {stored_params}")

            if args.mixed_precision == "fp16":
                cast_training_params([unet_model], dtype=torch.float32)
            
            # Log layer counts to verify completeness
            lora_a_layers = [n for n, p in unet_model.named_parameters() if ".lora_A." in n and p.requires_grad]
            lora_b_layers = [n for n, p in unet_model.named_parameters() if ".lora_B." in n and p.requires_grad]
            logger.info(f"Loaded {len(lora_a_layers)} LoRA A layers and {len(lora_b_layers)} LoRA B layers")
        
        if args.train_controlnet and controlnet_model is not None:
            controlnet_dir = os.path.join(input_dir, "controlnet")
            if os.path.exists(controlnet_dir):
                model_path = os.path.join(controlnet_dir, "diffusion_pytorch_model.safetensors")
                if not os.path.exists(model_path):
                    model_path = os.path.join(controlnet_dir, "diffusion_pytorch_model.bin")
                    
                if os.path.exists(model_path):
                    if model_path.endswith(".safetensors"):
                        from safetensors import safe_open
                        with safe_open(model_path, framework="pt") as f:
                            state_dict = {k: f.get_tensor(k) for k in f.keys()}
                    else:
                        state_dict = torch.load(model_path, map_location="cpu")
                    
                    controlnet_model.load_state_dict(state_dict)
                
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_controlnet and controlnet is not None:
            controlnet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer_params = []
    
    if args.train_controllora:
        lora_A_layers = [p for n, p in unet.named_parameters() if ".lora_A." in n and p.requires_grad]
        lora_B_layers = [p for n, p in unet.named_parameters() if ".lora_B." in n and p.requires_grad]
        
        if not lora_A_layers and hasattr(unet, '_pre_wrapped_lora_A_params'):
            lora_A_layers = unet._pre_wrapped_lora_A_params 
            lora_B_layers = unet._pre_wrapped_lora_B_params
        
        if not lora_A_layers and not lora_B_layers:
            raise RuntimeError("No LoRA parameters found. Training cannot proceed with ControlLoRA!")
        
        lora_params_set = set(lora_A_layers + lora_B_layers)
        total_lora_params = sum(p.numel() for p in lora_A_layers) + sum(p.numel() for p in lora_B_layers)

        optimizer_params.extend([
            {
                "params": lora_A_layers,
                "weight_decay": args.adam_weight_decay,
                "lr": args.learning_rate,
            },
            {
                "params": lora_B_layers,
                "weight_decay": args.adam_weight_decay,
                "lr": args.learning_rate,
            },
        ])
    
    if args.train_controlnet and controlnet is not None:
        controlnet_lr = args.controlnet_learning_rate or args.learning_rate
        
        controlnet_params = [p for p in controlnet.parameters() if p.requires_grad]
        
        if controlnet_params:
            optimizer_params.append({
                "params": controlnet_params,
                "weight_decay": args.adam_weight_decay,
                "lr": controlnet_lr,
            })
            
            total_controlnet_params = sum(p.numel() for p in controlnet_params)

    # Build optimizer kwargs - use foreach=False for bf16 gradient compatibility
    optimizer_kwargs = {
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
    }
    if optimizer_class == torch.optim.AdamW:
        optimizer_kwargs["foreach"] = False  # Disable fused implementation for bf16 gradients
    
    optimizer = optimizer_class(
        optimizer_params,
        **optimizer_kwargs,
    )

    def count_trainable_parameters(model, name="Model"):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"{name}: {trainable_params:,} trainable / {total_params:,} total ({trainable_params/total_params:.2%})")

    if args.train_controllora:
        visible_lora_params = sum(p.numel() for n, p in unet.named_parameters() if ".lora_" in n and p.requires_grad)
        
        total_lora_params = visible_lora_params
        if hasattr(unet, '_pre_wrapped_lora_A_params'):
            total_lora_params = sum(p.numel() for p in unet._pre_wrapped_lora_A_params)
            total_lora_params += sum(p.numel() for p in unet._pre_wrapped_lora_B_params)
        
        total_params = sum(p.numel() for p in unet.parameters())
        
        logger.info(f"UNet+LoRA: {total_lora_params:,} trainable / {total_params:,} total ({total_lora_params/total_params:.2%})")

    if args.train_controlnet and controlnet is not None:
        count_trainable_parameters(controlnet, "ControlNet")
        
    train_dataset = TrainDataset(args, tokenizer, accelerator)
    
    # Get reference to underlying dataset for step tracking
    gbuffer_dataset = getattr(train_dataset, 'dataset', None)
    if gbuffer_dataset is not None and hasattr(gbuffer_dataset, 'set_current_step'):
        logger.info("GBuffer dataset found")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True if args.dataloader_num_workers > 0 else False,
        prefetch_factor=6 if args.dataloader_num_workers > 0 else None,
        persistent_workers=True if args.dataloader_num_workers > 0 else False,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if args.train_controllora and args.train_controlnet:
        unet, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, controlnet, optimizer, train_dataloader, lr_scheduler
        )
        # Ensure trainable models stay in fp32 for stable gradients
        unwrapped_controlnet = accelerator.unwrap_model(controlnet)
        unwrapped_controlnet.to(dtype=torch.float32)
        # UNet LoRA params will be handled below
    elif args.train_controllora and args.use_controlnet:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        controlnet = controlnet.to(accelerator.device)
    elif args.train_controllora:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        if controlnet is not None:
            controlnet = controlnet.to(accelerator.device)
    elif args.train_controlnet:
        unet, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, controlnet, optimizer, train_dataloader, lr_scheduler
        )
        # Ensure trainable ControlNet stays in fp32 for stable gradients
        # accelerator.prepare() may have converted it to bf16
        unwrapped_controlnet = accelerator.unwrap_model(controlnet)
        unwrapped_controlnet.to(dtype=torch.float32)
        logger.info(f"ControlNet dtype after prepare: {next(unwrapped_controlnet.parameters()).dtype}")
    elif args.use_controlnet:
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, train_dataloader, lr_scheduler
        )
        unet = unet.to(accelerator.device)
        controlnet = controlnet.to(accelerator.device)
      
    if args.train_controllora:
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        if hasattr(unwrapped_unet, '_post_wrapped_lora_A_params'):
            delattr(unwrapped_unet, '_post_wrapped_lora_A_params')
        if hasattr(unwrapped_unet, '_post_wrapped_lora_B_params'):
            delattr(unwrapped_unet, '_post_wrapped_lora_B_params')
        
        lora_a_params = []
        lora_b_params = []
        for name, param in unwrapped_unet.named_parameters():
            if ".lora_A." in name:
                param.requires_grad = True
                lora_a_params.append(param)
            elif ".lora_B." in name:
                param.requires_grad = True
                lora_b_params.append(param)
        
        unwrapped_unet._post_wrapped_lora_A_params = lora_a_params
        unwrapped_unet._post_wrapped_lora_B_params = lora_b_params
        
        total_lora_params = len(lora_a_params) + len(lora_b_params)
        if total_lora_params == 0:
            logger.warning("No LoRA parameters found after unwrapping!")
        else:
            # Cast LoRA parameters to fp32 for stable gradients with bf16 mixed precision
            for param in lora_a_params + lora_b_params:
                param.data = param.data.to(torch.float32)
            logger.info(f"Cast {total_lora_params} LoRA parameters to fp32")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        tracker_config.pop("validation_image", None)
        tracker_config.pop("controlnet_validation_image", None)
        tracker_config.pop("final_validation_image", None)
        tracker_config.pop("validation_prev_final_image", None)
        tracker_config.pop("validation_prev_albedo_image", None)
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
    
    gradient_monitor = GradientMonitor(output_dir=os.path.join(args.output_dir, "gradient_logs"))
    gradient_monitor.register_models(unet=unet, controlnet=controlnet)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Examples: {len(train_dataset)}")
    logger.info(f"  Batches/epoch: {len(train_dataloader)}")
    logger.info(f"  Epochs: {args.num_train_epochs}")
    logger.info(f"  Batch size: {args.train_batch_size}")
    logger.info(f"  Total batch size: {total_batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  Total steps: {args.max_train_steps}")
    logger.info(f"  ControlLoRA: {args.train_controllora}")
    logger.info(f"  ControlNet: {args.train_controlnet}")
    logger.info(f"  Self-conditioning: {args.use_generated_conditioning}")
    
    if args.use_generated_conditioning:
        logger.info(f"  Gen freq: {args.generated_conditioning_freq} steps")
        logger.info(f"  Gen count: {args.generated_conditioning_count} images")
        logger.info(f"  Sample ratio: {args.generated_sample_ratio * 100:.1f}%")
    
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
        bar_format='{l_bar}{bar:5}{r_bar}',
    )
    
    torch.autograd.set_detect_anomaly(False)
    image_logs = None
    
    last_detailed_log = time.time()
    log_interval = 60.0
    
    # Track previous generation results for chaining
    previous_generation_results = None
    
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet if args.train_controllora else None, 
                                       controlnet if args.train_controlnet else None):
                compile_attempted = getattr(unet, '_compile_attempted', False)
                if hasattr(torch, 'compile') and not args.train_controlnet and not compile_attempted:
                    if sys.platform == 'win32':
                        unet._compile_attempted = True
                    else:
                        try:
                            if not hasattr(unet, '_compiled'):
                                unet_forward = torch.compile(
                                    unet.forward,
                                    mode="reduce-overhead",
                                    fullgraph=False
                                )
                                unet._original_forward = unet.forward
                                unet.forward = unet_forward
                                unet._compiled = True
                        except Exception as e:
                            if hasattr(unet, '_original_forward'):
                                unet.forward = unet._original_forward
                        finally:
                            unet._compile_attempted = True
                
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        # Keep tensors in fp32 - autocast handles bf16 conversion internally
                        batch[k] = batch[k].to(accelerator.device, non_blocking=True)

                # Track generated conditioning usage
                gen_cond_used = 0
                if "uses_generated_conditioning" in batch:
                    gen_flags = batch["uses_generated_conditioning"]
                    if isinstance(gen_flags, torch.Tensor):
                        gen_cond_used = gen_flags.sum().item()
                    elif isinstance(gen_flags, bool):
                        gen_cond_used = 1 if gen_flags else 0

                with accelerator.autocast():
                    if "uses_generated_conditioning" in batch:
                        batch_size = batch["pixel_values"].shape[0]
                        all_latents = []
                        
                        for i in range(batch_size):
                            latents = vae.encode(batch["pixel_values"][i:i+1]).latent_dist.sample()
                            all_latents.append(latents)
                        
                        latents = torch.cat(all_latents, dim=0) * vae.config.scaling_factor
                    else:
                        latents = vae.encode(batch["pixel_values"]).latent_dist.sample() * vae.config.scaling_factor

                batch_size = latents.shape[0]
                # Keep latents in fp32 for training - autocast handles bf16 conversion

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                
                lora_latents = None
                controlnet_input = None
                
                if (args.train_controllora or args.use_controllora) and "conditioning_pixel_values" in batch:
                    with torch.set_grad_enabled(True):
                        with accelerator.autocast():
                            batch_size = batch["conditioning_pixel_values"].shape[0]
                            
                            if batch["conditioning_pixel_values"].dim() != 4:
                                raise ValueError(f"Previous frames must be 4D [B,C,H,W], got {batch['conditioning_pixel_values'].shape}")

                            lora_latents = []
                            for batch_idx in range(batch_size):
                                single_prev_frame = batch["conditioning_pixel_values"][batch_idx:batch_idx+1]
                                single_latent = vae.encode(single_prev_frame).latent_dist.sample() * vae.config.scaling_factor
                                lora_latents.append(single_latent)
                                
                            if len(lora_latents) != batch_size:
                                raise ValueError(f"Latent count ({len(lora_latents)}) != batch size ({batch_size})")
                
                if (args.train_controlnet or args.use_controlnet) and "controlnet_pixel_values" in batch and controlnet is not None:
                    controlnet_input = batch["controlnet_pixel_values"]
                    
                    if controlnet_input.dim() == 3 and controlnet_input.shape[0] == 10:
                        controlnet_input = controlnet_input.unsqueeze(0)
                        
                    current_time = time.time()
                    if current_time - last_detailed_log > log_interval and step == 0:
                        logger.info(f"ControlNet input: {controlnet_input.shape}")
                        if controlnet_input.shape[1] == 10:
                            logger.info(f"10-channel G-buffer (2 RGB + 4 single + 1 irradiance)")
                        last_detailed_log = current_time
                        
                controlnet_down_block_res_samples = controlnet_mid_block_res_sample = None
                controlnet_influence_metrics = {}
                
                if (args.train_controlnet or args.use_controlnet) and controlnet is not None and controlnet_input is not None:
                    controlnet_output = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_input,
                        conditioning_scale=args.controlnet_conditioning_scale,
                        return_dict=False,
                    )
                    controlnet_down_block_res_samples, controlnet_mid_block_res_sample = controlnet_output
                    
                    controlnet_influence_metrics = {}

                    if controlnet_down_block_res_samples is not None and len(controlnet_down_block_res_samples) > 0:
                        sample_norms = []
                        for i, sample in enumerate(controlnet_down_block_res_samples):
                            if sample is not None:
                                try:
                                    norm = sample.norm().item() / sample.shape[0]
                                    if not math.isnan(norm) and not math.isinf(norm):
                                        sample_norms.append(norm)
                                        controlnet_influence_metrics[f"controlnet/block_{i}_norm"] = norm
                                except Exception as e:
                                    logger.warning(f"Error computing norm for block {i}: {e}")
                        
                        if sample_norms:
                            avg_norm = sum(sample_norms) / len(sample_norms)
                            controlnet_influence_metrics["controlnet/avg_influence"] = avg_norm

                    if not args.train_controlnet and args.train_controllora:
                        if controlnet_down_block_res_samples is None:
                            raise RuntimeError("ControlNet produced None for down_block_res_samples!")

                        for i in range(len(controlnet_down_block_res_samples)):
                            if controlnet_down_block_res_samples[i] is not None:
                                controlnet_down_block_res_samples[i] = controlnet_down_block_res_samples[i].detach().requires_grad_(True)
                        
                        if controlnet_mid_block_res_sample is not None:
                            controlnet_mid_block_res_sample = controlnet_mid_block_res_sample.detach().requires_grad_(True)
                
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    extra_conditions=lora_latents,
                    controlnet_down_block_res_samples=controlnet_down_block_res_samples,
                    controlnet_mid_block_res_sample=controlnet_mid_block_res_sample,
                    return_dict=False,
                )[0]
                
                model_diagnostics = {}
                
                has_gradients = model_pred.requires_grad
                
                if not args.train_controlnet and args.train_controllora:
                    if not has_gradients:
                        raise RuntimeError("UNet output has no gradients! ControlLoRA cannot learn.")
                
                if controlnet_down_block_res_samples is not None and len(controlnet_down_block_res_samples) > 0:
                    model_norm = model_pred.norm().item() / model_pred.shape[0]
                    latent_norm = noisy_latents.norm().item() / noisy_latents.shape[0]
                    
                    model_diagnostics["model/output_norm"] = model_norm
                    model_diagnostics["model/latent_norm"] = latent_norm
                    model_diagnostics["model/output_latent_ratio"] = model_norm/latent_norm
                 
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                detailed_loss_metrics = {}

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                if global_step % 100 == 0 and accelerator.is_main_process:
                    step_to_check = global_step
                    
                accelerator.backward(loss)
                
                if global_step % 10 == 0:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                    if global_step % 50 == 0:
                        if hasattr(gradient_monitor, "step_data"):
                            for key in gradient_monitor.step_data:
                                if len(gradient_monitor.step_data[key]) > 10:
                                    gradient_monitor.step_data[key] = gradient_monitor.step_data[key][-10:]
                            if len(gradient_monitor.step_history) > 10:
                                gradient_monitor.step_history = gradient_monitor.step_history[-10:]
                        gc.collect()
                
                if global_step % 500 == 0 and accelerator.is_main_process:
                    gradient_monitor.log_gradients(step_to_check)

                if accelerator.sync_gradients:
                    params_to_clip = []
                    if args.train_controllora:
                        params_to_clip.extend(lora_A_layers + lora_B_layers)
                    
                    if args.train_controlnet and controlnet is not None:
                        params_to_clip.extend([p for p in controlnet.parameters() if p.requires_grad])
                    
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                # Convert bf16 gradients to fp32 in-place for optimizer compatibility
                if accelerator.mixed_precision == "bf16":
                    for param_group in optimizer.param_groups:
                        for param in param_group["params"]:
                            if param.grad is not None and param.grad.dtype == torch.bfloat16:
                                param.grad.data = param.grad.data.float()

                optimizer.step()
                lr_scheduler.step()
                
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Update dataset with current step for generated conditioning
                if gbuffer_dataset is not None and hasattr(gbuffer_dataset, 'set_current_step'):
                    gbuffer_dataset.set_current_step(global_step)

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                    if args.controlnet_validation_image is not None and (global_step % args.validation_steps == 0 or global_step == 1):
                        image_logs = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            
            # Track generated conditioning usage
            if gen_cond_used > 0:
                logs["gen_cond"] = gen_cond_used
            
            # Add detailed loss component metrics if available
            if detailed_loss_metrics:
                logs.update(detailed_loss_metrics)
                
            # Add all diagnostic metrics to logs for TensorBoard
            if controlnet_influence_metrics:
                logs.update(controlnet_influence_metrics)
            if model_diagnostics:
                logs.update(model_diagnostics)

            # Create a filtered version for terminal display
            display_logs = {
                "loss": logs["loss"], 
                "lr": logs["lr"]
            }

            # Only show avg_influence in terminal, not all the individual block metrics
            if "controlnet/avg_influence" in logs:
                display_logs["avg_infl"] = logs["controlnet/avg_influence"]
            
            # Show when generated conditioning is used
            if "gen_cond" in logs:
                display_logs["gen"] = logs["gen_cond"]

            # Log all metrics to TensorBoard, but only display essential ones in terminal
            progress_bar.set_postfix(**display_logs)  # Use filtered logs for terminal
            accelerator.log(logs, step=global_step)   # Use full logs for TensorBoard
            
            # Stop training if max steps reached
            if global_step >= args.max_train_steps:
                break
            
            # Print diagnostic alerts to console only when thresholds are exceeded
            if (accelerator.is_main_process and 
                controlnet_influence_metrics.get("controlnet/influence_alert", 0.0) > 0.5 and 
                global_step % 100 == 0):
                avg_influence = controlnet_influence_metrics.get("controlnet/avg_influence", 0)
                logger.warning(f"ControlNet influence very strong: {avg_influence:.2f}")

            # Generate new conditioning images if needed
            if (args.train_controllora and 
                args.use_generated_conditioning and 
                (global_step % args.generated_conditioning_freq) == 0 and 
                global_step > 0 and 
                global_step >= args.generated_conditioning_start_step and
                global_step != getattr(args, '_last_generation_step', -1)):
                
                logger.info(f"Generating conditioning images at step {global_step}...")
                
                inference_scheduler = DPMSolverMultistepScheduler.from_config(
                    DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler").config
                )
                
                logger.info("Saving model states before generation...")
                saved_unet_state = None
                saved_controlnet_state = None
                
                if args.train_controllora:
                    saved_unet_state = accelerator.unwrap_model(unet).state_dict()
                
                if args.train_controlnet and controlnet is not None:
                    saved_controlnet_state = accelerator.unwrap_model(controlnet).state_dict()
               
                with torch.no_grad():
                    temp_unet = accelerator.unwrap_model(unet)
                    if hasattr(temp_unet, "_orig_mod"):
                        temp_unet = temp_unet._orig_mod
                    
                    if controlnet is not None:
                        temp_controlnet = accelerator.unwrap_model(controlnet)
                        if hasattr(temp_controlnet, "_orig_mod"):
                            temp_controlnet = temp_controlnet._orig_mod
                    else:
                        temp_controlnet = None
                
                with torch.no_grad():
                    unet_was_training = temp_unet.training
                    temp_unet.eval()
                    
                    controlnet_was_training = None
                    if temp_controlnet is not None:
                        controlnet_was_training = temp_controlnet.training
                        temp_controlnet.eval()
                    
                    vae.eval()
                    text_encoder.eval()
                    
                    assert not temp_unet.training
                    assert not vae.training
                    assert not text_encoder.training
                    if temp_controlnet is not None:
                        assert not temp_controlnet.training
                    
                    logger.info("All models in eval mode for generation")
                    
                    # Create pipeline with the current model state
                    pipe = StableDiffusionDualControlPipeline(
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        unet=temp_unet,
                        controlnet=temp_controlnet,
                        scheduler=inference_scheduler,
                        safety_checker=None,
                        feature_extractor=None,
                        requires_safety_checker=False
                    )

                    # Enable important optimizations
                    if args.enable_xformers_memory_efficient_attention:
                        pipe.enable_xformers_memory_efficient_attention()
                    
                    # Load LoRA weights if training ControlLoRA
                    if args.train_controllora or args.use_controllora:
                        # Instead of trying to check if the adapter exists, explicitly add it
                        # This ensures the adapter is properly set up
                        
                        logger.info(f"Explicitly loading ControlLoRA adapter state into pipeline")
                        
                        # Get the adapter state dict (raise an error if the adapter doesn't exist)
                        try:
                            unet_lora_state_dict = get_peft_model_state_dict(temp_unet, adapter_name=args.lora_adapter_name)
                            
                            # Check if we actually got any adapter weights
                            if not unet_lora_state_dict:
                                raise ValueError(f"Got empty state dict for adapter '{args.lora_adapter_name}'")
                                
                            # Log some information about what we're loading
                            lora_a_count = sum(1 for k in unet_lora_state_dict if ".lora_A." in k)
                            lora_b_count = sum(1 for k in unet_lora_state_dict if ".lora_B." in k)
                            logger.info(f"Loading adapter with {lora_a_count} LoRA A layers and {lora_b_count} LoRA B layers")
                            
                            # If we reach here, we have a valid adapter state dict
                        except Exception as e:
                            logger.error(f"Failed to get ControlLoRA adapter state: {str(e)}")
                            logger.error("Self-conditioning requires a properly initialized ControlLoRA adapter")
                            raise RuntimeError(f"ControlLoRA adapter extraction failed: {str(e)}")
                        
                        # Set the adapter in the pipeline
                        try:
                            # This will raise an error if it fails to set the adapter
                            set_peft_model_state_dict(pipe.unet, unet_lora_state_dict, adapter_name=args.lora_adapter_name)
                            logger.info(f"Successfully loaded ControlLoRA adapter '{args.lora_adapter_name}' into pipeline")
                        except Exception as e:
                            logger.error(f"Failed to set ControlLoRA adapter: {str(e)}")
                            raise RuntimeError(f"Setting ControlLoRA adapter failed: {str(e)}")
                        # After loading LoRA, activate the adapters
                        pipe.unet.set_adapter([args.lora_adapter_name])
                        pipe.unet.activate_extra_condition_adapters()
                        logger.info(f"Activated LoRA adapter '{args.lora_adapter_name}'")

                    # Move to device for inference
                    pipe = pipe.to(accelerator.device, dtype=weight_dtype)
                    # Keep VAE in float32 for stable decoding (must be after pipe.to())
                    pipe.vae = pipe.vae.to(dtype=torch.float32)
                    # Ensure UNet/ControlNet are fully in weight_dtype
                    pipe.unet = pipe.unet.to(dtype=weight_dtype)
                    if pipe.controlnet is not None:
                        pipe.controlnet = pipe.controlnet.to(dtype=weight_dtype)
                    
                    # Disable progress bar except on main process
                    pipe.set_progress_bar_config(disable=not accelerator.is_local_main_process)
                    
                    # Get dataset instance
                    if not isinstance(train_dataset.dataset, GBufferDataset):
                        raise TypeError(f"Expected dataset to be GBufferDataset, got {type(train_dataset.dataset).__name__}")
                    dataset_instance = train_dataset.dataset
                    
                    # Run generation
                    generation_results = generate_conditioning_images(
                        pipeline=pipe,
                        args=args,
                        accelerator=accelerator,
                        tokenizer=tokenizer,
                        weight_dtype=weight_dtype,
                        dataset=dataset_instance,
                        step=global_step,
                        batch_size=args.generated_conditioning_batch_size,
                        previous_generated=previous_generation_results
                    )
                    
                    # Store for next generation's chaining
                    previous_generation_results = generation_results
                    
                    # Update the dataset with new generations
                    if dataset_instance is not None and accelerator.is_main_process and generation_results:
                        dataset_instance.update_generated_images(generation_results, global_step)
                    
                    # Restore training modes
                    if unet_was_training:
                        temp_unet.train()
                    if controlnet_was_training:
                        temp_controlnet.train()
                    
                    # Clean up
                    del pipe
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    
                    # Force PyTorch to reset its autograd engine
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    
                    logger.info("Ready to continue training...")
                    args._last_generation_step = global_step

        # Break epoch loop if max steps reached
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if gradient_monitor:
            gradient_monitor.save_plots()
            
        if args.train_controllora:
            unet = unwrap_model(unet)
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unet, adapter_name=args.lora_adapter_name)
            )
            StableDiffusionDualControlPipeline.save_lora_weights(
                save_directory=args.output_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )
        
        if args.train_controlnet and controlnet is not None:
            controlnet = unwrap_model(controlnet)
            controlnet_dir = os.path.join(args.output_dir, "controlnet")
            os.makedirs(controlnet_dir, exist_ok=True)
            controlnet.save_pretrained(controlnet_dir)

        if args.train_controllora:
            del unet
            unet = None
        if args.train_controlnet and controlnet is not None:
            del controlnet
            controlnet = None
        gc.collect()
        torch.cuda.empty_cache()
        
        image_logs = None
        if args.controlnet_validation_image is not None and not args.skip_final_validation:
            image_logs = log_validation(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=controlnet,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                token=hub_token,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*", "checkpoint-*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)