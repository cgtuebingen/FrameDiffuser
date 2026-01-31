"""
FrameDiffuser G-Buffer Dataset.

Dataset for loading G-buffer sequences with temporal conditioning support.
Handles:
- Hierarchical or flat directory structures
- 10-channel G-buffer (BaseColor, Normals, Depth, Roughness, Metallic, Irradiance)
- Temporal offset sampling for ControlLoRA conditioning
- Self-conditioning with model-generated images
- Noise injection for training robustness

Copyright (c) 2025 Ole Beisswenger, Jan-Niklas Dihlmann, Hendrik Lensch
Licensed under MIT License.
"""

import os
import re
import json
import random
import logging
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from gbuffer_masking_utils import fill_black_pixels_in_basecolor

logger = logging.getLogger(__name__)


class GBufferDataset(Dataset):
    """
    Dataset for G-buffer sequences with temporal and self-conditioning support.
    
    G-buffer format (10 channels):
        0-2: BaseColor (RGB)
        3-5: Normals (RGB) 
        6:   Depth
        7:   Roughness
        8:   Metallic
        9:   Irradiance (computed from previous frame)
    """
    
    # Noise injection parameters
    NOISE_MIN = 0.0
    NOISE_MAX = 0.2
    
    # Default path
    DEFAULT_TRAIN_DIR = "./data/train"

    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        
        # Temporal sampling configuration
        self.max_prev_frames = getattr(args, 'max_prev_frames', 2)
        
        # Self-conditioning configuration
        self.use_generated_conditioning = getattr(args, 'use_generated_conditioning', False)
        self.generated_sample_ratio = getattr(args, 'generated_sample_ratio', 0.25)
        self.generated_conditioning_start_step = getattr(args, 'generated_conditioning_start_step', 0)
        self.current_step = 0
        
        # Multiprocessing sync
        self.sync_file = os.path.join(getattr(args, 'output_dir', '.'), 'generated_sync.json')
        self.sync_version = -1
        
        # Irradiance configuration
        self.use_black_irradiance = getattr(args, 'use_black_irradiance', False)
        
        # Generated images storage
        self.generated_images = []
        
        # Use default path if not specified
        self.root_dir = getattr(args, 'gbuffer_root_dir', None) or self.DEFAULT_TRAIN_DIR
        
        # Build dataset
        self._find_sequences()
        self.exclude_sequence = self._select_validation_sequence()
        self._build_dataset()
        
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.single_channel_transforms = transforms.Compose([
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Tokenize prompt
        train_prompt = args.train_prompt
        self.tokenized_prompts = [self._tokenize_prompt(train_prompt) for _ in range(len(self.target_files))]
    
    def create_irradiance_map(self, final_img, albedo_img):
        """
        Create grayscale irradiance map from final and albedo images.
        
        Irradiance = Final / Albedo (clamped to [0, 2], normalized to [0, 1])
        """
        if isinstance(final_img, Image.Image):
            final_tensor = torch.from_numpy(np.array(final_img)).float() / 255.0
        else:
            final_tensor = torch.from_numpy(final_img).float() / 255.0
            
        if isinstance(albedo_img, Image.Image):
            albedo_tensor = torch.from_numpy(np.array(albedo_img)).float() / 255.0
        else:
            albedo_tensor = torch.from_numpy(albedo_img).float() / 255.0
        
        # Convert to grayscale
        if final_tensor.dim() == 3 and final_tensor.shape[2] == 3:
            final_gray = 0.299 * final_tensor[:, :, 0] + 0.587 * final_tensor[:, :, 1] + 0.114 * final_tensor[:, :, 2]
        else:
            final_gray = final_tensor
            
        if albedo_tensor.dim() == 3 and albedo_tensor.shape[2] == 3:
            albedo_gray = 0.299 * albedo_tensor[:, :, 0] + 0.587 * albedo_tensor[:, :, 1] + 0.114 * albedo_tensor[:, :, 2]
        else:
            albedo_gray = albedo_tensor
        
        # Compute irradiance with valid mask
        valid_mask = albedo_gray > 0.01
        irradiance = torch.ones_like(albedo_gray)
        irradiance[valid_mask] = final_gray[valid_mask] / (albedo_gray[valid_mask] + 1e-6)
        irradiance[~valid_mask] = 2.0
        
        result = torch.clamp(irradiance, 0, 2) / 2.0
        return result.cpu().numpy()
    
    def _find_sequences(self):
        """Find all sequences in directory structure."""
        self.sequences = {}
        self.is_flat_structure = False
        self.flat_sequence_prefix = None  # For files like Train_FinalImage_0000.png
        self.sequence_uses_simple_naming = {}  # Track naming style per sequence
        
        # Check for flat structure (root/FinalImage/...)
        final_image_path = os.path.join(self.root_dir, "FinalImage")
        if os.path.exists(final_image_path):
            frame_numbers = []
            simple_naming = False
            for file in os.listdir(final_image_path):
                if file.endswith((".png", ".jpg", ".jpeg")):
                    # Try simple naming: FinalImage_0000.png
                    match = re.search(r'^FinalImage_(\d+)\.(png|jpg|jpeg)$', file)
                    if match:
                        frame_numbers.append(int(match.group(1)))
                        simple_naming = True
                        continue
                    # Try prefix naming: SequenceName_FinalImage_0000.png
                    match = re.search(r'^(.+)_FinalImage_(\d+)\.(png|jpg|jpeg)$', file)
                    if match:
                        if self.flat_sequence_prefix is None:
                            self.flat_sequence_prefix = match.group(1)
                        frame_numbers.append(int(match.group(2)))
            
            if frame_numbers:
                self.is_flat_structure = True
                self.sequences["_flat"] = [min(frame_numbers), max(frame_numbers)]
                self.sequence_uses_simple_naming["_flat"] = simple_naming
                return
        
        # Look for sequence subdirectories
        for seq_dir in os.listdir(self.root_dir):
            seq_path = os.path.join(self.root_dir, seq_dir)
            
            if not os.path.isdir(seq_path):
                continue
            
            final_image_path = os.path.join(seq_path, "FinalImage")
            if not os.path.exists(final_image_path):
                continue
            
            frame_numbers = []
            simple_naming = False
            for file in os.listdir(final_image_path):
                if file.endswith((".png", ".jpg", ".jpeg")):
                    # Try simple naming: FinalImage_0000.png
                    match = re.search(r'^FinalImage_(\d+)\.(png|jpg|jpeg)$', file)
                    if match:
                        frame_numbers.append(int(match.group(1)))
                        simple_naming = True
                        continue
                    # Try prefix naming: SeqName_FinalImage_0000.png
                    match = re.search(r'_FinalImage_(\d+)\.(png|jpg|jpeg)$', file)
                    if match:
                        frame_numbers.append(int(match.group(1)))
            
            if frame_numbers:
                self.sequences[seq_dir] = [min(frame_numbers), max(frame_numbers)]
                self.sequence_uses_simple_naming[seq_dir] = simple_naming
    
    def _select_validation_sequence(self):
        """Returns sequences to exclude from training based on prefix."""
        exclude_prefix = getattr(self.args, 'exclude_sequence_prefix', None)
        if not exclude_prefix:
            return []
        
        excluded = []
        for seq_name in self.sequences.keys():
            if seq_name.startswith(exclude_prefix):
                excluded.append(seq_name)
        return excluded
    
    def _build_dataset(self):
        """Build dataset from directory structure."""
        self.file_indices = []
        self.target_files = []
        self.file_sequences = []
        self.sequence_frames = defaultdict(list)
        self.sequence_paths = {}
        
        required_buffers = ["BaseColor", "Normals", "Depth", "Roughness"]
        
        for seq_name in sorted(self.sequences.keys()):
            if seq_name in self.exclude_sequence:
                continue
            
            if self.is_flat_structure:
                seq_path = self.root_dir
            else:
                seq_path = os.path.join(self.root_dir, seq_name)
            self.sequence_paths[seq_name] = seq_path
            
            final_image_path = os.path.join(seq_path, "FinalImage")
            
            # Determine pattern based on naming style
            uses_simple = self.sequence_uses_simple_naming.get(seq_name, False)
            if self.is_flat_structure:
                if self.flat_sequence_prefix:
                    pattern = re.compile(rf'^{re.escape(self.flat_sequence_prefix)}_FinalImage_(\d+)\.(png|jpg|jpeg)$')
                else:
                    pattern = re.compile(r'^FinalImage_(\d+)\.(png|jpg|jpeg)$')
            elif uses_simple:
                pattern = re.compile(r'^FinalImage_(\d+)\.(png|jpg|jpeg)$')
            else:
                pattern = re.compile(rf'^{re.escape(seq_name)}_FinalImage_(\d+)\.(png|jpg|jpeg)$')
            
            for file in sorted(os.listdir(final_image_path)):
                match = pattern.match(file)
                if match:
                    frame_idx_str = match.group(1)
                    frame_idx = int(frame_idx_str)
                    
                    # Check all required buffers exist
                    all_exist = True
                    for buffer_type in required_buffers:
                        buffer_path = self._construct_buffer_path(seq_path, seq_name, buffer_type, frame_idx_str)
                        if not os.path.exists(buffer_path):
                            all_exist = False
                            break
                    
                    if all_exist:
                        full_path = os.path.join(final_image_path, file)
                        idx = len(self.target_files)
                        self.file_indices.append(frame_idx_str)
                        self.target_files.append(full_path)
                        self.file_sequences.append(seq_name)
                        self.sequence_frames[seq_name].append((idx, frame_idx))
        
        # Sort frames within each sequence
        for seq_id in self.sequence_frames:
            self.sequence_frames[seq_id].sort(key=lambda x: x[1])
    
    def _construct_buffer_path(self, seq_path, seq_name, buffer_type, frame_idx_str):
        """Construct path for a G-buffer component."""
        uses_simple = self.sequence_uses_simple_naming.get(seq_name, False)
        
        if self.is_flat_structure:
            if self.flat_sequence_prefix:
                filename = f"{self.flat_sequence_prefix}_{buffer_type}_{frame_idx_str}"
            else:
                filename = f"{buffer_type}_{frame_idx_str}"
        elif uses_simple:
            filename = f"{buffer_type}_{frame_idx_str}"
        else:
            filename = f"{seq_name}_{buffer_type}_{frame_idx_str}"
        
        # Try .png first, then .jpg
        path_png = os.path.join(seq_path, buffer_type, filename + ".png")
        path_jpg = os.path.join(seq_path, buffer_type, filename + ".jpg")
        return path_png if os.path.exists(path_png) else path_jpg
    
    def _get_matching_buffer(self, index, buffer_type):
        """Get matching buffer file for index and type."""
        file_idx = self.file_indices[index]
        seq_name = self.file_sequences[index]
        seq_path = self.sequence_paths[seq_name]
        
        buffer_path = self._construct_buffer_path(seq_path, seq_name, buffer_type, file_idx)
        
        if not os.path.exists(buffer_path):
            if buffer_type.lower() == "metallic":
                return None
            raise ValueError(f"Missing {buffer_type} file: {buffer_path}")
        
        return buffer_path
    
    def update_generated_images(self, generation_results, step):
        """Update dataset with newly generated images for self-conditioning."""
        self.current_step = step
        
        if not generation_results:
            return
        
        self.generated_images.clear()
        
        new_version = generation_results.get("version", step)
        
        for seq_id, images in generation_results.get("generated_images", {}).items():
            for img in images:
                if os.path.exists(img["path"]) and os.path.getsize(img["path"]) > 0:
                    gen_info = {
                        'path': img['path'],
                        'seq_id': seq_id,
                        'frame_idx': img['frame_idx'],
                        'was_chained': img.get('was_chained', False),
                        'generation_step': new_version
                    }
                    self.generated_images.append(gen_info)
        
        self.sync_version = new_version
        
        # Write sync file for multiprocessing workers
        sync_data = {
            'version': new_version,
            'step': step,
            'generated_images': self.generated_images
        }
        try:
            with open(self.sync_file, 'w') as f:
                json.dump(sync_data, f)
        except Exception as e:
            pass
    
    def set_current_step(self, step):
        """Update current training step."""
        self.current_step = step
    
    def _reload_from_sync_file(self):
        """Reload generated images from sync file (for multiprocessing workers)."""
        if not os.path.exists(self.sync_file):
            return
        
        try:
            with open(self.sync_file, 'r') as f:
                sync_data = json.load(f)
            
            file_version = sync_data.get('version', -1)
            
            if file_version > self.sync_version:
                self.sync_version = file_version
                self.current_step = sync_data.get('step', self.current_step)
                
                self.generated_images.clear()
                
                for gen_info in sync_data.get('generated_images', []):
                    if os.path.exists(gen_info['path']):
                        self.generated_images.append(gen_info)
        except Exception:
            pass
    
    def get_clean_sample_for_generation(self, dataset_idx, chained_prev_image=None, chained_frame_idx=None):
        """Get clean sample without noise augmentation for generation."""
        temporal_idx, temporal_offset = self._get_temporal_offset(dataset_idx)
        
        if chained_prev_image is not None and chained_frame_idx is not None:
            # Find dataset_idx for chained frame
            seq_id = self.file_sequences[dataset_idx]
            chained_dataset_idx = temporal_idx
            for didx, fidx in self.sequence_frames[seq_id]:
                if fidx == chained_frame_idx:
                    chained_dataset_idx = didx
                    break
            
            gbuffer, target_image = self._load_gbuffer_with_custom_prev(
                dataset_idx, chained_prev_image, chained_dataset_idx
            )
            controllora_image_pil = chained_prev_image
        else:
            gbuffer, target_image = self._load_gbuffer(dataset_idx, ('real', temporal_idx))
            controllora_image_pil = Image.open(self.target_files[temporal_idx]).convert("RGB")
        
        controllora_cond = self.image_transforms(controllora_image_pil)
        
        return {
            "pixel_values": target_image,
            "input_ids": self.tokenized_prompts[dataset_idx],
            "conditioning_pixel_values": controllora_cond,
            "controlnet_pixel_values": gbuffer,
            "file_idx": self.file_indices[dataset_idx],
            "temporal_offset": temporal_offset,
            "uses_generated_conditioning": chained_prev_image is not None
        }
    
    def _load_gbuffer_with_custom_prev(self, idx, custom_prev_image, prev_dataset_idx):
        """Load G-buffer with custom prev image for irradiance (used for chaining)."""
        basecolor_pil = Image.open(self._get_matching_buffer(idx, "BaseColor")).convert("RGB")
        normals_pil = Image.open(self._get_matching_buffer(idx, "Normals")).convert("RGB")
        depth_pil = Image.open(self._get_matching_buffer(idx, "Depth")).convert("L")
        roughness_pil = Image.open(self._get_matching_buffer(idx, "Roughness")).convert("L")
        
        metallic_path = self._get_matching_buffer(idx, "Metallic")
        if metallic_path and os.path.exists(metallic_path):
            metallic_pil = Image.open(metallic_path).convert("L")
        else:
            metallic_pil = Image.new('L', basecolor_pil.size, 0)
        
        target_image_pil = Image.open(self.target_files[idx]).convert("RGB")
        reference_size = basecolor_pil.size
        
        # Resize all to reference size
        depth_pil = depth_pil.resize(reference_size, Image.LANCZOS)
        target_image_pil = target_image_pil.resize(reference_size, Image.LANCZOS)
        normals_pil = normals_pil.resize(reference_size, Image.LANCZOS)
        roughness_pil = roughness_pil.resize(reference_size, Image.LANCZOS)
        metallic_pil = metallic_pil.resize(reference_size, Image.LANCZOS)
        
        masked_basecolor_pil = fill_black_pixels_in_basecolor(basecolor_pil, depth_pil, target_image_pil)
        
        # Compute irradiance from custom prev image
        prev_albedo_path = self._get_matching_buffer(prev_dataset_idx, "BaseColor")
        prev_albedo_pil = Image.open(prev_albedo_path).convert("RGB")
        
        custom_prev_image = custom_prev_image.resize(reference_size, Image.LANCZOS)
        prev_albedo_pil = prev_albedo_pil.resize(reference_size, Image.LANCZOS)
        
        prev_irradiance = self.create_irradiance_map(custom_prev_image, prev_albedo_pil)
        irradiance_uint8 = (prev_irradiance * 255).astype(np.uint8)
        irradiance_pil = Image.fromarray(irradiance_uint8, mode='L')
        
        # Build G-buffer tensor
        gbuffer = torch.cat([
            self.image_transforms(masked_basecolor_pil),
            self.image_transforms(normals_pil),
            self.single_channel_transforms(depth_pil),
            self.single_channel_transforms(roughness_pil),
            self.single_channel_transforms(metallic_pil),
            self.single_channel_transforms(irradiance_pil)
        ], dim=0)
        
        target_image = self.image_transforms(target_image_pil)
        return gbuffer, target_image
    
    def _get_sample_using_generated(self):
        """Get training sample using generated image as conditioning."""
        if not self.generated_images:
            return None
        
        gen_info = random.choice(self.generated_images)
        valid_targets = self._find_target_frames_for_generated(gen_info)
        
        if not valid_targets:
            return None
        
        target_dataset_idx, target_frame_idx, offset = self._select_target_with_weighting(valid_targets)
        
        return {
            'dataset_idx': target_dataset_idx,
            'gen_info': gen_info,
            'temporal_offset': offset
        }
    
    def _find_target_frames_for_generated(self, gen_info):
        """Find valid target frames within temporal window for a generated image."""
        seq_id = gen_info['seq_id']
        gen_frame_idx = gen_info['frame_idx']
        
        if seq_id not in self.sequence_frames:
            return []
        
        valid_targets = []
        for dataset_idx, frame_idx in self.sequence_frames[seq_id]:
            offset = frame_idx - gen_frame_idx
            abs_offset = abs(offset)
            
            if abs_offset == 0:
                if random.random() < 0.2:
                    valid_targets.append((dataset_idx, frame_idx, offset))
            elif abs_offset <= self.max_prev_frames:
                valid_targets.append((dataset_idx, frame_idx, offset))
        
        return valid_targets
    
    def _select_target_with_weighting(self, valid_targets):
        """Select target frame using temporal weighting."""
        if not valid_targets:
            return None
        
        weights = []
        for _, _, offset in valid_targets:
            abs_offset = abs(offset)
            if abs_offset == 0:
                weights.append(8.0)
            elif abs_offset == 1:
                weights.append(4.0)
            elif abs_offset == 2:
                weights.append(1.0)
            else:
                weights.append(1.0 / abs_offset)
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        idx = random.choices(range(len(valid_targets)), weights=weights)[0]
        return valid_targets[idx]
    
    def _get_temporal_offset(self, idx):
        """Get temporal frame offset for sampling previous frame."""
        current_sequence = self.file_sequences[idx]
        current_frame_idx = int(self.file_indices[idx])
        
        sorted_frames = sorted(self.sequence_frames[current_sequence], key=lambda x: x[1])
        
        current_position = -1
        for i, (dataset_idx, frame_idx) in enumerate(sorted_frames):
            if frame_idx == current_frame_idx:
                current_position = i
                break
        
        if current_position == -1:
            return idx, 0
        
        candidates = []
        weights = []
        offsets = []
        
        for offset in range(-self.max_prev_frames, self.max_prev_frames + 1):
            if offset == 0:
                continue
            target_position = current_position + offset
            if 0 <= target_position < len(sorted_frames):
                dataset_idx, _ = sorted_frames[target_position]
                candidates.append(dataset_idx)
                offsets.append(offset)
                
                distance = abs(offset)
                if distance == 1:
                    weights.append(4.0)
                elif distance == 2:
                    weights.append(1.0)
                else:
                    weights.append(1.0 / distance)
        
        # Add same frame with low weight
        if candidates:
            candidates.append(idx)
            offsets.append(0)
            weights.append(0.5)
        
        if not candidates:
            return idx, 0
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        selected_idx = random.choices(range(len(candidates)), weights=weights)[0]
        return candidates[selected_idx], offsets[selected_idx]
    
    def _tokenize_prompt(self, prompt):
        inputs = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return inputs.input_ids[0]
    
    def _add_noise_to_rgb_image(self, image_pil):
        """Add noise to RGB PIL image - Paper Eq. 4: σ ∼ U(0,0.2), ε ∼ N(0,I)"""
        noise_scale = random.uniform(self.NOISE_MIN, self.NOISE_MAX)
        image_array = np.array(image_pil).astype(np.float32)
        noise = np.random.randn(*image_array.shape) * noise_scale * 255
        noisy_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array), noise_scale

    def _load_gbuffer(self, idx, prev_frame_info=None):
        """Load G-buffer components for dataset index."""
        basecolor_pil = Image.open(self._get_matching_buffer(idx, "BaseColor")).convert("RGB")
        normals_pil = Image.open(self._get_matching_buffer(idx, "Normals")).convert("RGB")
        depth_pil = Image.open(self._get_matching_buffer(idx, "Depth")).convert("L")
        roughness_pil = Image.open(self._get_matching_buffer(idx, "Roughness")).convert("L")
        
        metallic_path = self._get_matching_buffer(idx, "Metallic")
        if metallic_path and os.path.exists(metallic_path):
            metallic_pil = Image.open(metallic_path).convert("L")
        else:
            metallic_pil = Image.new('L', basecolor_pil.size, 0)
        
        target_image_pil = Image.open(self.target_files[idx]).convert("RGB")
        reference_size = basecolor_pil.size
        
        # Resize all to reference size
        depth_pil = depth_pil.resize(reference_size, Image.LANCZOS)
        target_image_pil = target_image_pil.resize(reference_size, Image.LANCZOS)
        normals_pil = normals_pil.resize(reference_size, Image.LANCZOS)
        roughness_pil = roughness_pil.resize(reference_size, Image.LANCZOS)
        metallic_pil = metallic_pil.resize(reference_size, Image.LANCZOS)
        
        masked_basecolor_pil = fill_black_pixels_in_basecolor(basecolor_pil, depth_pil, target_image_pil)
        
        # Compute irradiance
        if self.use_black_irradiance:
            irradiance_pil = Image.new('L', reference_size, 0)
        elif prev_frame_info is not None:
            frame_type, frame_data = prev_frame_info
            
            if frame_type == 'generated':
                gen_info = frame_data
                prev_final_pil = Image.open(gen_info['path']).convert("RGB")
                
                # Find albedo for generated frame
                gen_seq_id = gen_info['seq_id']
                gen_frame_idx = gen_info['frame_idx']
                gen_dataset_idx = None
                for dataset_idx, frame_idx in self.sequence_frames[gen_seq_id]:
                    if frame_idx == gen_frame_idx:
                        gen_dataset_idx = dataset_idx
                        break
                
                if gen_dataset_idx is not None:
                    prev_albedo_path = self._get_matching_buffer(gen_dataset_idx, "BaseColor")
                    prev_albedo_pil = Image.open(prev_albedo_path).convert("RGB")
                else:
                    raise RuntimeError(f"Could not find albedo for generated frame")
            else:
                temporal_idx = frame_data
                prev_final_path = self.target_files[temporal_idx]
                prev_albedo_path = self._get_matching_buffer(temporal_idx, "BaseColor")
                prev_albedo_pil = Image.open(prev_albedo_path).convert("RGB")
                
                if hasattr(self, '_cached_noisy_prev') and self._cached_noisy_prev is not None:
                    cached_idx, cached_noisy_prev, _ = self._cached_noisy_prev
                    if cached_idx == temporal_idx:
                        prev_final_pil = cached_noisy_prev
                    else:
                        prev_final_pil = Image.open(prev_final_path).convert("RGB")
                        prev_final_pil, _ = self._add_noise_to_rgb_image(prev_final_pil)
                else:
                    prev_final_pil = Image.open(prev_final_path).convert("RGB")
                    prev_final_pil, _ = self._add_noise_to_rgb_image(prev_final_pil)
            
            prev_final_pil = prev_final_pil.resize(reference_size, Image.LANCZOS)
            prev_albedo_pil = prev_albedo_pil.resize(reference_size, Image.LANCZOS)
            
            prev_irradiance = self.create_irradiance_map(prev_final_pil, prev_albedo_pil)
            irradiance_uint8 = (prev_irradiance * 255).astype(np.uint8)
            irradiance_pil = Image.fromarray(irradiance_uint8, mode='L')
        else:
            # Fallback: compute from temporal offset
            temporal_idx, _ = self._get_temporal_offset(idx)
            prev_final_path = self.target_files[temporal_idx]
            prev_albedo_path = self._get_matching_buffer(temporal_idx, "BaseColor")
            prev_albedo_pil = Image.open(prev_albedo_path).convert("RGB")
            prev_final_pil = Image.open(prev_final_path).convert("RGB")
            prev_final_pil, _ = self._add_noise_to_rgb_image(prev_final_pil)
            
            prev_final_pil = prev_final_pil.resize(reference_size, Image.LANCZOS)
            prev_albedo_pil = prev_albedo_pil.resize(reference_size, Image.LANCZOS)
            
            prev_irradiance = self.create_irradiance_map(prev_final_pil, prev_albedo_pil)
            irradiance_uint8 = (prev_irradiance * 255).astype(np.uint8)
            irradiance_pil = Image.fromarray(irradiance_uint8, mode='L')
        
        # Build G-buffer tensor (10 channels)
        gbuffer = torch.cat([
            self.image_transforms(masked_basecolor_pil),      # 3 channels
            self.image_transforms(normals_pil),               # 3 channels
            self.single_channel_transforms(depth_pil),        # 1 channel
            self.single_channel_transforms(roughness_pil),    # 1 channel
            self.single_channel_transforms(metallic_pil),     # 1 channel
            self.single_channel_transforms(irradiance_pil)    # 1 channel
        ], dim=0)
        
        target_image = self.image_transforms(target_image_pil)
        return gbuffer, target_image
    
    def __len__(self):
        return len(self.target_files)
    
    def __getitem__(self, idx):
        """Get training sample with temporal or generated conditioning."""
        self._reload_from_sync_file()
        
        use_generated = False
        gen_info = None
        prev_frame_info = None
        
        # Check if we should use generated conditioning
        should_use_generated = (
            self.use_generated_conditioning and 
            self.current_step >= self.generated_conditioning_start_step and
            random.random() < self.generated_sample_ratio and
            self.generated_images
        )
        
        if should_use_generated:
            gen_sample = self._get_sample_using_generated()
            if gen_sample:
                use_generated = True
                idx = gen_sample['dataset_idx']
                gen_info = gen_sample['gen_info']
                temporal_offset = gen_sample['temporal_offset']
                prev_frame_info = ('generated', gen_info)
        
        if not use_generated:
            temporal_idx, temporal_offset = self._get_temporal_offset(idx)
            
            # Add noise to prev frame for robustness
            prev_final_path = self.target_files[temporal_idx]
            prev_final_pil = Image.open(prev_final_path).convert("RGB")
            noisy_prev_final, noise_scale = self._add_noise_to_rgb_image(prev_final_pil)
            self._cached_noisy_prev = (temporal_idx, noisy_prev_final, noise_scale)
            
            prev_frame_info = ('real', temporal_idx)
        
        gbuffer, target_image = self._load_gbuffer(idx, prev_frame_info)
        
        if use_generated:
            controllora_image_pil = Image.open(gen_info['path']).convert("RGB")
            controllora_cond = self.image_transforms(controllora_image_pil)
        else:
            controllora_cond = self.image_transforms(noisy_prev_final)
        
        return {
            "pixel_values": target_image,
            "input_ids": self.tokenized_prompts[idx],
            "conditioning_pixel_values": controllora_cond,
            "controlnet_pixel_values": gbuffer,
            "file_idx": self.file_indices[idx],
            "temporal_offset": temporal_offset,
            "uses_generated_conditioning": use_generated
        }
