#!/usr/bin/env python
# coding=utf-8
"""
FrameDiffuser Inference GUI.

Interactive application for autoregressive frame generation with:
- Model loading and configuration
- Batch inference with progress tracking
- Real-time preview and metrics visualization
- Video export and VAE comparison tools

Usage:
    python inference.py

Copyright (c) 2025 Ole Beisswenger, Jan-Niklas Dihlmann, Hendrik Lensch
Licensed under MIT License.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import gc
from pathlib import Path
import json
from datetime import datetime
import tempfile
import subprocess
import threading
import queue
import time
import cv2
from scipy import ndimage
import csv
import re
import pandas as pd

# Metrics imports
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

# PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                            QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit,
                            QGroupBox, QGridLayout, QSlider, QCheckBox,
                            QFileDialog, QMessageBox, QProgressBar, QTabWidget,
                            QScrollArea, QSplitter, QTableWidget, QTableWidgetItem,
                            QHeaderView, QTextBrowser, QRadioButton, QButtonGroup,
                            QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QFont, QPainter, QPen, QBrush

# Matplotlib for graph visualization
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Import custom modules
from diffusers import AutoencoderKL, DDPMScheduler, UniPCMultistepScheduler
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.utils import is_xformers_available

# Add parent directory to path for custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import UNet2DConditionModelEx, ControlNetModelEx
from pipeline import StableDiffusionDualControlPipeline
from gbuffer_masking_utils import fill_black_pixels_in_basecolor

# Define transforms
rgb_transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

grayscale_transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def compute_irradiance_map(final_img, albedo_img, save_path=None):
    """Compute irradiance map from final image and albedo."""
    import numpy as np
    import torch
    
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
    
    result = torch.clamp(irradiance, 0, 2) / 2.0
    result_np = result.cpu().numpy()
    
    if save_path:
        irradiance_uint8 = (result_np * 255).astype(np.uint8)
        irradiance_img = Image.fromarray(irradiance_uint8, mode='L')
        irradiance_img.save(save_path)
    
    return result_np


class InteractiveGraphWidget(FigureCanvas):
    """Interactive matplotlib graph for editing scale patterns."""
    
    values_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), dpi=80)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Scale Value')
        self.ax.set_ylim(-0.1, 2.1)
        self.ax.grid(True, alpha=0.3)
        
        self.period = 24
        self.values = [1.0] * self.period
        self.dragging = False
        self.drag_index = None
        
        self.line = None
        self.points = None
        
        # Connect mouse events
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)
        
        self.update_plot()
        
    def set_period(self, period):
        """Update the period and resize values array."""
        old_period = self.period
        self.period = period
        
        # Resample values to new period
        if old_period != period:
            old_values = self.values.copy()
            self.values = []
            for i in range(period):
                # Linear interpolation from old values
                old_index = i * (old_period - 1) / max(1, period - 1)
                if old_index >= old_period - 1:
                    self.values.append(old_values[-1])
                else:
                    idx = int(old_index)
                    frac = old_index - idx
                    val = old_values[idx] * (1 - frac) + old_values[min(idx + 1, old_period - 1)] * frac
                    self.values.append(val)
        
        self.update_plot()
        
    def set_values(self, values):
        """Set all values at once."""
        self.values = values[:self.period]
        while len(self.values) < self.period:
            self.values.append(1.0)
        self.update_plot()
        
    def set_pattern(self, pattern_type, start_value, end_value):
        """Apply a pattern to the values."""
        self.values = []
        for i in range(self.period):
            if pattern_type == 'constant':
                self.values.append(start_value)
            elif pattern_type == 'linear' or pattern_type == 'sawtooth':
                t = i / max(1, self.period - 1)
                self.values.append(start_value + (end_value - start_value) * t)
            elif pattern_type == 'sine':
                phase = 2 * np.pi * i / self.period
                amplitude = (end_value - start_value) / 2
                center = (start_value + end_value) / 2
                self.values.append(center + amplitude * np.sin(phase))
            elif pattern_type == 'square':
                self.values.append(start_value if i < self.period / 2 else end_value)
            else:
                self.values.append(1.0)
        
        self.update_plot()
        
    def update_plot(self):
        """Redraw the plot."""
        self.ax.clear()
        
        # Set labels and grid
        self.ax.set_xlabel('Frame', fontsize=10)
        self.ax.set_ylabel('Scale Value', fontsize=10)
        self.ax.set_ylim(-0.1, 2.1)
        self.ax.set_xlim(-0.5, self.period + 0.5)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f'Scale Pattern (Period: {self.period})', fontsize=11)
        
        # Plot the line with period + 1 to show reset
        x = list(range(self.period)) + [self.period]
        y = self.values + [self.values[0]]  # Add first value at end to show loop
        
        # Main line
        self.line, = self.ax.plot(x, y, 'b-', linewidth=2, alpha=0.7)
        
        # Interactive points (only for the period, not the reset point)
        self.points = self.ax.scatter(range(self.period), self.values, 
                                     c='red', s=50, zorder=5, picker=True)
        
        # Show reset with dashed line
        self.ax.plot([self.period - 1, self.period], [self.values[-1], self.values[0]], 
                    'b--', linewidth=1, alpha=0.5)
        
        # Add value labels for first few and last points
        label_indices = []
        if self.period <= 10:
            label_indices = range(self.period)
        else:
            # Show first 3, last 2, and some in middle
            label_indices = [0, 1, 2] + [self.period // 2] + [self.period - 2, self.period - 1]
        
        for i in label_indices:
            if i < len(self.values):
                self.ax.annotate(f'{self.values[i]:.2f}', 
                               (i, self.values[i]), 
                               textcoords="offset points", 
                               xytext=(0,10), 
                               ha='center',
                               fontsize=8,
                               alpha=0.7)
        
        self.fig.tight_layout()
        self.draw()
        
    def find_nearest_point(self, event):
        """Find the nearest point to mouse click."""
        if event.xdata is None or event.ydata is None:
            return None
        
        min_dist = float('inf')
        nearest_idx = None
        
        for i in range(self.period):
            dist = np.sqrt((i - event.xdata)**2 + (self.values[i] - event.ydata)**2)
            if dist < min_dist and dist < 0.5:  # 0.5 is threshold for selection
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
        
    def on_press(self, event):
        """Handle mouse press."""
        if event.button == 1:  # Left click
            idx = self.find_nearest_point(event)
            if idx is not None:
                self.dragging = True
                self.drag_index = idx
                
    def on_release(self, event):
        """Handle mouse release."""
        if self.dragging:
            self.dragging = False
            self.drag_index = None
            self.values_changed.emit()
            
    def on_motion(self, event):
        """Handle mouse motion."""
        if self.dragging and self.drag_index is not None and event.ydata is not None:
            # Clamp value between 0 and 2
            new_value = max(0.0, min(2.0, event.ydata))
            self.values[self.drag_index] = new_value
            self.update_plot()
            
    def get_values(self):
        """Get the current values."""
        return self.values.copy()


class ScaleController:
    """Manages dynamic scale values over time with custom per-frame values."""
    
    def __init__(self):
        self.pattern_type = 'custom'
        self.period = 24
        self.custom_values = [1.0] * self.period
        
    def set_custom_values(self, values):
        """Set custom per-frame values."""
        self.custom_values = values
        self.period = len(values)
        self.pattern_type = 'custom'
        
    def get_scale(self, frame_idx, total_frames=None):
        """Get scale value for a given frame."""
        if self.pattern_type == 'custom' and self.custom_values:
            cycle_position = frame_idx % len(self.custom_values)
            return self.custom_values[cycle_position]
        return 1.0
    
    def get_scale_batch(self, frame_indices, total_frames=None):
        """Get scale values for a batch of frames."""
        return [self.get_scale(idx, total_frames) for idx in frame_indices]
    
    def get_description(self):
        """Get a text description of the pattern."""
        if self.pattern_type == 'custom':
            avg_val = np.mean(self.custom_values)
            min_val = min(self.custom_values)
            max_val = max(self.custom_values)
            return f"Custom: avg={avg_val:.2f}, range=[{min_val:.2f}, {max_val:.2f}], period={self.period}"
        return "Custom pattern"


class GBufferProcessor:
    """Handles loading and processing of G-buffer components."""
    
    def __init__(self, gbuffer_dir, original_frames_dir, gbuffer_prefix, resolution=512,
                 injection_source_dir=None, enable_injection=True):
        self.gbuffer_dir = gbuffer_dir
        self.original_frames_dir = original_frames_dir
        self.gbuffer_prefix = gbuffer_prefix
        self.resolution = resolution
        self.depth_cache = {}
        self.albedo_cache = {}
        self.irradiance_dir = None
        self.injection_source_dir = injection_source_dir
        self.enable_injection = enable_injection
        
        # Detect flat structure
        self.is_flat_structure = self._detect_flat_structure()
        
    def _detect_flat_structure(self):
        """Detect if using flat file structure (FinalImage_0000.png) vs old structure (Prefix_FinalImage_0000.png)."""
        # Check for flat structure in FinalImage folder
        final_image_dir = os.path.join(self.gbuffer_dir, "FinalImage")
        if os.path.exists(final_image_dir):
            for f in os.listdir(final_image_dir):
                if re.match(r'^FinalImage_\d+\.png$', f):
                    return True
        # Also check original_frames_dir
        if os.path.exists(self.original_frames_dir):
            for f in os.listdir(self.original_frames_dir):
                if re.match(r'^FinalImage_\d+\.png$', f):
                    return True
        return False
    
    def _get_frame_str(self, frame_idx):
        """Get frame string with correct zero-padding."""
        # Try to detect padding from existing files
        if self.is_flat_structure:
            check_dir = self.original_frames_dir
            pattern = r'^FinalImage_(\d+)\.png$'
        else:
            check_dir = self.original_frames_dir
            pattern = rf'^{re.escape(self.gbuffer_prefix)}_FinalImage_(\d+)\.png$'
        
        if os.path.exists(check_dir):
            for f in os.listdir(check_dir):
                match = re.match(pattern, f)
                if match:
                    num_digits = len(match.group(1))
                    return f"{frame_idx:0{num_digits}d}"
        
        # Default to 4 digits
        return f"{frame_idx:04d}"
        
    def set_irradiance_dir(self, output_dir):
        """Set the directory for saving irradiance maps."""
        self.irradiance_dir = os.path.join(output_dir, "irradiance")
        os.makedirs(self.irradiance_dir, exist_ok=True)
        
    def load_original_frame(self, frame_idx):
        """Load original frame at native resolution."""
        frame_str = self._get_frame_str(frame_idx)
        
        if self.is_flat_structure:
            path = os.path.join(self.original_frames_dir, f"FinalImage_{frame_str}.png")
        else:
            path = os.path.join(self.original_frames_dir, f"{self.gbuffer_prefix}_FinalImage_{frame_str}.png")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Original frame not found: {path}")
        return Image.open(path).convert("RGB")
    
    def load_injection_frame(self, frame_idx):
        """Load injection source frame for basecolor sky filling."""
        if not self.injection_source_dir:
            raise FileNotFoundError(
                f"BaseColor injection is enabled but no injection_source_dir specified. "
                f"Either provide an injection source folder or disable injection."
            )
        frame_str = self._get_frame_str(frame_idx)
        
        if self.is_flat_structure:
            path = os.path.join(self.injection_source_dir, f"FinalImage_{frame_str}.png")
        else:
            path = os.path.join(self.injection_source_dir, f"{self.gbuffer_prefix}_FinalImage_{frame_str}.png")
        
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BaseColor injection source not found: {path}\n"
                f"Either provide all injection source frames or disable injection."
            )
        return Image.open(path).convert("RGB")
    
    def load_albedo(self, frame_idx):
        """Load and cache albedo for a frame."""
        if frame_idx in self.albedo_cache:
            return self.albedo_cache[frame_idx]
        
        frame_str = self._get_frame_str(frame_idx)
        
        if self.is_flat_structure:
            albedo_path = os.path.join(self.gbuffer_dir, "BaseColor", f"BaseColor_{frame_str}.png")
        else:
            prefix = self.gbuffer_prefix
            albedo_path = os.path.join(self.gbuffer_dir, "BaseColor", f"{prefix}_FinalImageBaseColor_{frame_str}.png")
        
        if os.path.exists(albedo_path):
            albedo_img = Image.open(albedo_path).convert("RGB")
            self.albedo_cache[frame_idx] = albedo_img
            return albedo_img
        return None
    
    def load_depth_tensor(self, frame_idx):
        """Load and cache depth tensor for similarity comparison."""
        if frame_idx in self.depth_cache:
            return self.depth_cache[frame_idx]
        
        frame_str = self._get_frame_str(frame_idx)
        
        if self.is_flat_structure:
            depth_path = os.path.join(self.gbuffer_dir, "Depth", f"Depth_{frame_str}.png")
        else:
            prefix = self.gbuffer_prefix
            depth_path = os.path.join(self.gbuffer_dir, "Depth", f"{prefix}_FinalImageDepth_{frame_str}.png")
        
        if os.path.exists(depth_path):
            depth_img = Image.open(depth_path).convert("L")
            depth_tensor = grayscale_transform(depth_img)
            self.depth_cache[frame_idx] = depth_tensor
            return depth_tensor
        return None
    
    def compute_depth_similarity(self, frame_idx1, frame_idx2):
        """Compute similarity between two depth frames."""
        depth1 = self.load_depth_tensor(frame_idx1)
        depth2 = self.load_depth_tensor(frame_idx2)
        
        if depth1 is None or depth2 is None:
            return 0.0
        
        diff = torch.abs(depth1 - depth2)
        mean_diff = diff.mean().item()
        similarity = 1.0 - (mean_diff / 2.0)
        
        return similarity
    
    def create_g_buffer_tensor(self, frame_idx, apply_masking=True, irradiance_source=None, variant='auto'):
        """
        Create G-buffer tensor with masking and irradiance applied.
        
        Args:
            frame_idx: Current frame index
            apply_masking: Whether to apply masking
            irradiance_source: Dict with 'final' and 'albedo' images for irradiance computation
            variant: 'auto' or 'gt' - determines whether to apply input filtering
        """
        frame_str = self._get_frame_str(frame_idx)
        prefix = self.gbuffer_prefix
        
        # Load all components
        if self.is_flat_structure:
            basecolor_path = os.path.join(self.gbuffer_dir, "BaseColor", f"BaseColor_{frame_str}.png")
            normals_path = os.path.join(self.gbuffer_dir, "Normals", f"Normals_{frame_str}.png")
            depth_path = os.path.join(self.gbuffer_dir, "Depth", f"Depth_{frame_str}.png")
            roughness_path = os.path.join(self.gbuffer_dir, "Roughness", f"Roughness_{frame_str}.png")
            metallic_path = os.path.join(self.gbuffer_dir, "Metallic", f"Metallic_{frame_str}.png")
        else:
            basecolor_path = os.path.join(self.gbuffer_dir, "BaseColor", f"{prefix}_FinalImageBaseColor_{frame_str}.png")
            normals_path = os.path.join(self.gbuffer_dir, "Normals", f"{prefix}_FinalImageNormals_{frame_str}.png")
            depth_path = os.path.join(self.gbuffer_dir, "Depth", f"{prefix}_FinalImageDepth_{frame_str}.png")
            roughness_path = os.path.join(self.gbuffer_dir, "Roughness", f"{prefix}_FinalImageRoughness_{frame_str}.png")
            metallic_path = os.path.join(self.gbuffer_dir, "Metallic", f"{prefix}_FinalImageMetallic_{frame_str}.png")
        
        # Load components
        basecolor_img = Image.open(basecolor_path).convert("RGB")
        normals_img = Image.open(normals_path).convert("RGB")
        depth_img = Image.open(depth_path).convert("L")
        roughness_img = Image.open(roughness_path).convert("L")
        
        if os.path.exists(metallic_path):
            metallic_img = Image.open(metallic_path).convert("L")
        else:
            metallic_img = Image.new('L', basecolor_img.size, 0)
        
        if apply_masking:
            reference_size = basecolor_img.size
            
            if depth_img.size != reference_size:
                depth_img = depth_img.resize(reference_size, Image.LANCZOS)
            
            # Apply basecolor injection if enabled
            if self.enable_injection:
                injection_frame = self.load_injection_frame(frame_idx)
                if injection_frame.size != reference_size:
                    injection_frame = injection_frame.resize(reference_size, Image.LANCZOS)
                
                basecolor_img = fill_black_pixels_in_basecolor(
                    basecolor_img, depth_img, injection_frame
                )
            
            # Apply normals masking
            if roughness_img.size != reference_size:
                roughness_img = roughness_img.resize(reference_size, Image.LANCZOS)
            if metallic_img.size != reference_size:
                metallic_img = metallic_img.resize(reference_size, Image.LANCZOS)
            
            component_images = [basecolor_img, depth_img, roughness_img, metallic_img]
            h, w = basecolor_img.size[1], basecolor_img.size[0]
            all_black_mask = np.ones((h, w), dtype=bool)
            
            for img in component_images:
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    is_black = np.all(img_array <= 1, axis=2)
                else:
                    is_black = img_array <= 1
                all_black_mask = all_black_mask & is_black
        
        # Determine save path for irradiance
        irradiance_save_path = None
        if self.irradiance_dir:
            # Create descriptive filename based on source
            if irradiance_source is not None:
                # Irradiance from previous frame or GT
                irradiance_save_path = os.path.join(self.irradiance_dir, f"irradiance_{frame_str}_from_prev.png")
            else:
                # Self-irradiance
                irradiance_save_path = os.path.join(self.irradiance_dir, f"irradiance_{frame_str}_self.png")
        
        # Compute irradiance
        if irradiance_source is not None:
            # Use provided source for irradiance
            final_img = irradiance_source['final']
            albedo_img = irradiance_source['albedo']
            
            # Ensure same size as other components
            reference_size = basecolor_img.size
            if final_img.size != reference_size:
                final_img = final_img.resize(reference_size, Image.LANCZOS)
            if albedo_img.size != reference_size:
                albedo_img = albedo_img.resize(reference_size, Image.LANCZOS)
            
            irradiance = compute_irradiance_map(final_img, albedo_img, irradiance_save_path)
        else:
            # Self-irradiance from current frame's ground truth
            original_frame = self.load_original_frame(frame_idx)
            albedo_frame = self.load_albedo(frame_idx)
            
            reference_size = basecolor_img.size
            if original_frame.size != reference_size:
                original_frame = original_frame.resize(reference_size, Image.LANCZOS)
            if albedo_frame.size != reference_size:
                albedo_frame = albedo_frame.resize(reference_size, Image.LANCZOS)
            
            irradiance = compute_irradiance_map(original_frame, albedo_frame, irradiance_save_path)
        
        # Convert irradiance to PIL Image
        irradiance_uint8 = (irradiance * 255).astype(np.uint8)
        irradiance_pil = Image.fromarray(irradiance_uint8, mode='L')
        
        # Apply transforms
        components = []
        components.append(rgb_transform(basecolor_img))
        components.append(rgb_transform(normals_img))
        
        depth_tensor = grayscale_transform(depth_img)
        self.depth_cache[frame_idx] = depth_tensor
        components.append(depth_tensor)
        
        for img in [roughness_img, metallic_img]:
            components.append(grayscale_transform(img))
        
        # Add irradiance as 10th channel
        components.append(grayscale_transform(irradiance_pil))
        
        # Create 10-channel tensor
        g_buffer_tensor = torch.cat(components, dim=0)
        return g_buffer_tensor


class VAEComparisonWorker(QThread):
    """Worker thread for generating VAE comparisons."""
    
    progress_update = pyqtSignal(int, int)
    log_message = pyqtSignal(str)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True
        
    def stop(self):
        self.is_running = False
        
    def run(self):
        try:
            import torch
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            from diffusers import AutoencoderKL
            import re
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log_message.emit(f"Loading VAE on {device}...")
            
            original_frames_dir = self.config['original_frames_dir']
            gbuffer_prefix = self.config.get('gbuffer_prefix', '')
            self.log_message.emit(f"Original frames directory: {original_frames_dir}")
            self.log_message.emit(f"G-buffer prefix: {gbuffer_prefix}")
            
            if not os.path.exists(original_frames_dir):
                raise RuntimeError(f"Original frames directory not found: {original_frames_dir}")
            
            # Load VAE
            vae = AutoencoderKL.from_pretrained(
                self.config['base_model'],
                subfolder="vae"
            ).to(device)
            vae.eval()
            
            # Get resize dimensions
            resize_width = self.config.get('resize_width', 512)
            resize_height = self.config.get('resize_height', 512)
            target_size = (resize_width, resize_height) if (resize_width != 512 or resize_height != 512) else None
            
            generated_frames = self.config['generated_frames']
            total_frames = len(generated_frames)
            successful_comparisons = 0
            
            self.log_message.emit(f"Processing {total_frames} frames...")
            
            for idx, gen_frame_path in enumerate(generated_frames):
                if not self.is_running:
                    break
                
                # Extract frame number
                match = re.search(r'frame_(\d+)\.png$', gen_frame_path)
                if not match:
                    continue
                frame_num = match.group(1)
                
                # Load generated frame
                gen_img = Image.open(gen_frame_path).convert('RGB')
                
                gt_frame_path = os.path.join(original_frames_dir, f"{gbuffer_prefix}_FinalImage_{frame_num}.png")
                
                # Try flat structure if old format doesn't exist
                if not os.path.exists(gt_frame_path):
                    flat_path = os.path.join(original_frames_dir, f"FinalImage_{frame_num}.png")
                    if os.path.exists(flat_path):
                        gt_frame_path = flat_path
                
                if not os.path.exists(gt_frame_path):
                    error_msg = (
                        f"GT frame not found: {gt_frame_path}\n\n"
                    )
                    
                    try:
                        actual_files = sorted(os.listdir(original_frames_dir))
                        if actual_files:
                            sample_files = actual_files[:10]
                            error_msg += f"Files in directory (first 10):\n"
                            for f in sample_files:
                                error_msg += f"  - {f}\n"
                            
                            subdirs = [f for f in actual_files[:20] if os.path.isdir(os.path.join(original_frames_dir, f))]
                            if subdirs:
                                error_msg += f"\nSubdirectories found: {', '.join(subdirs[:5])}\n"
                        else:
                            error_msg += "Directory is empty.\n"
                    except Exception as e:
                        error_msg += f"Cannot read directory: {e}\n"
                    
                    error_msg += (
                        f"\nExpected filename pattern: {gbuffer_prefix}_FinalImage_XXXX.png or FinalImage_XXXX.png (flat)\n"
                    )
                    
                    self.log_message.emit(error_msg)
                    raise RuntimeError(error_msg)
                
                # Load GT frame
                gt_img = Image.open(gt_frame_path).convert('RGB')
                
                # Resize GT to 512x512 for VAE processing
                gt_img_512 = gt_img.resize((512, 512), Image.LANCZOS)
                
                # Resize GT to 512x512 for VAE processing
                gt_img_512 = gt_img.resize((512, 512), Image.LANCZOS)
                
                # Encode and decode GT through VAE
                gt_tensor = torch.from_numpy(np.array(gt_img_512)).float() / 255.0
                gt_tensor = gt_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                gt_tensor = (gt_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
                
                with torch.no_grad():
                    latent = vae.encode(gt_tensor).latent_dist.mode()
                    latent = latent * vae.config.scaling_factor
                    reconstructed = vae.decode(latent / vae.config.scaling_factor).sample
                
                # Convert back to image
                reconstructed = (reconstructed / 2 + 0.5).clamp(0, 1)
                reconstructed = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
                reconstructed = (reconstructed * 255).astype(np.uint8)
                gt_vae_img = Image.fromarray(reconstructed)
                
                # Resize if requested
                if target_size:
                    gt_vae_img = gt_vae_img.resize(target_size, Image.LANCZOS)
                    gen_img = gen_img.resize(target_size, Image.LANCZOS)
                
                # Create side-by-side comparison
                comparison_width = gt_vae_img.width + gen_img.width + 30  # 30px gap
                comparison_height = max(gt_vae_img.height, gen_img.height) + 60  # 60px for labels
                
                comparison = Image.new('RGB', (comparison_width, comparison_height), color=(40, 40, 40))
                
                # Paste images
                comparison.paste(gt_vae_img, (10, 40))
                comparison.paste(gen_img, (gt_vae_img.width + 20, 40))
                
                # Add labels
                draw = ImageDraw.Draw(comparison)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                draw.text((10, 10), "GT (VAE Reconstructed)", fill=(255, 255, 255), font=font)
                draw.text((gt_vae_img.width + 20, 10), "Model Output", fill=(255, 255, 255), font=font)
                
                # Save comparison
                comparison_path = os.path.join(
                    self.config['output_dir'],
                    f"comparison_{frame_num}.png"
                )
                comparison.save(comparison_path)
                self.log_message.emit(f"Saved: {comparison_path}")
                successful_comparisons += 1
                
                self.progress_update.emit(idx + 1, total_frames)
                self.log_message.emit(f"Processed frame {frame_num}")
            
            del vae
            torch.cuda.empty_cache()
            gc.collect()
            
            if successful_comparisons == 0:
                raise RuntimeError(
                    f"No comparisons created. Check 'Original Frames' path."
                )
            
            self.log_message.emit(f"VAE comparison complete: {successful_comparisons}/{total_frames} frames successfully processed")
            self.log_message.emit(f"Comparison files saved to: {self.config['output_dir']}")
            
            self.finished.emit()
            
        except Exception as e:
            import traceback
            error_msg = f"Error in VAE comparison: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)


class InferenceWorker(QThread):
    """Worker thread for running inference with true batch processing on GPU."""
    
    progress_update = pyqtSignal(int, int)
    frame_generated = pyqtSignal(int, object)  # frame_idx, image
    log_message = pyqtSignal(str)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True
        self.pipeline = None
        self.device = None
        self.generator = None
        self.generated_frames = {}
        self.generated_frames_list = []  # Keep order
        self.previous_batch_last_frame = None
        self.latents_dir = None
        
        # Create latents directory if needed
        if self.config.get('save_latents', False):
            self.latents_dir = os.path.join(self.config['output_dir'], 'latents')
            os.makedirs(self.latents_dir, exist_ok=True)
        
    def stop(self):
        self.is_running = False
        
    def run(self):
        try:
            self.setup_pipeline()
            self.run_inference()
            self.finished.emit()
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"{str(e)}\n{traceback.format_exc()}")
            
    def setup_pipeline(self):
        """Initialize the pipeline and models."""
        self.log_message.emit("Setting up device and models...")
        
        gpu_id = self.config.get('gpu_id', 0)
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.log_message.emit(f"Using device: {self.device}")
        
        seed = self.config.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        
        mixed_precision = self.config.get('mixed_precision', 'fp16')
        if mixed_precision == "fp16":
            weight_dtype = torch.float16
            self.log_message.emit("Using FP16 mixed precision")
        elif mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            self.log_message.emit("Using BF16 mixed precision")
        else:
            weight_dtype = torch.float32
            self.log_message.emit("Using FP32 (no mixed precision)")
        
        self.log_message.emit("Loading base models...")
        base_model = self.config.get('base_model', 'runwayml/stable-diffusion-v1-5')
        tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
        
        self.log_message.emit("Loading UNet...")
        unet = UNet2DConditionModelEx.from_pretrained(base_model, subfolder="unet")
        lora_adapter_name = self.config.get('lora_adapter_name', 'gbuffer')
        unet = unet.add_extra_conditions([lora_adapter_name])
        
        self.log_message.emit("Loading ControlNet...")
        controlnet_path = self.config['controlnet_path']
        
        config_path = os.path.join(controlnet_path, "config.json")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            cross_attention_dim = config.get('cross_attention_dim', 768)
            gbuffer_channels = config.get('gbuffer_channels', 10)
            self.log_message.emit(f"ControlNet config: cross_attention_dim={cross_attention_dim}, gbuffer_channels={gbuffer_channels}")
        except:
            cross_attention_dim = 768
            gbuffer_channels = 10
            self.log_message.emit(f"Using defaults: cross_attention_dim={cross_attention_dim}, gbuffer_channels={gbuffer_channels}")
        
        controlnet = ControlNetModelEx.from_pretrained(
            controlnet_path,
            cross_attention_dim=cross_attention_dim,
            gbuffer_channels=gbuffer_channels
        )
        
        text_encoder.eval()
        vae.eval()
        unet.eval()
        controlnet.eval()
        
        text_encoder.to(device=self.device, dtype=weight_dtype)
        vae.to(device=self.device, dtype=weight_dtype)
        controlnet.to(device=self.device, dtype=weight_dtype)
        
        scheduler = UniPCMultistepScheduler.from_config(
            DDPMScheduler.from_pretrained(base_model, subfolder="scheduler").config
        )
        
        self.pipeline = StableDiffusionDualControlPipeline(
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
        
        # Load ControlLoRA
        self.log_message.emit("Loading ControlLoRA...")
        controllora_path = self.config['controllora_path']
        self.pipeline.load_lora_weights(controllora_path, adapter_name=lora_adapter_name)
        
        # Set adapter
        if hasattr(self.pipeline.unet, 'set_adapter'):
            self.pipeline.unet.set_adapter(lora_adapter_name)
        self.pipeline.unet.to(device=self.device, dtype=weight_dtype)
        
        # Enable xformers if available
        if not self.config.get('disable_xformers', False) and is_xformers_available():
            self.pipeline.enable_xformers_memory_efficient_attention()
            self.log_message.emit("Enabled xformers memory efficient attention")
            
        self.log_message.emit("Pipeline setup complete!")
        
    def run_inference(self):
        """Run the inference loop with true batch processing."""
        # Default injection_source_dir to gbuffer_dir/FinalImage if not specified
        injection_source = self.config.get('injection_source_dir', '')
        if not injection_source and self.config.get('enable_injection', True):
            injection_source = os.path.join(self.config['gbuffer_dir'], 'FinalImage')
        
        gbuffer_processor = GBufferProcessor(
            self.config['gbuffer_dir'],
            self.config['original_frames_dir'],
            self.config['gbuffer_prefix'],
            injection_source_dir=injection_source,
            enable_injection=self.config.get('enable_injection', True)
        )
        
        gbuffer_processor.set_irradiance_dir(self.config['output_dir'])
        self.log_message.emit(f"Irradiance maps will be saved to: {gbuffer_processor.irradiance_dir}")
        
        if self.config.get('save_latents', False):
            self.log_message.emit(f"Output latents will be saved to: {self.latents_dir}")
        
        frame_indices = self.get_frame_indices()
        total_frames = len(frame_indices)
        
        variant = self.config.get('variant', 'auto')
        direction = self.config.get('direction', 'forward')
        batch_size = self.config.get('batch_size', 1)
        
        ground_truth_frames = {}
        ground_truth_albedo = {}
        
        # Always load albedo (BaseColor) for all frames - needed for irradiance
        self.log_message.emit("Loading albedo (BaseColor) for all frames...")
        for frame_idx in frame_indices:
            if not self.is_running:
                return
            albedo = gbuffer_processor.load_albedo(frame_idx)
            if albedo:
                albedo = albedo.resize((512, 512), Image.LANCZOS)
                ground_truth_albedo[frame_idx] = albedo
        
        # Load FinalImage based on mode
        if variant == 'gt':
            # GT mode: need all FinalImages
            self.log_message.emit("GT mode: Loading all FinalImage frames...")
            for frame_idx in frame_indices:
                if not self.is_running:
                    return
                original_frame = gbuffer_processor.load_original_frame(frame_idx)
                original_frame = original_frame.resize((512, 512), Image.LANCZOS)
                ground_truth_frames[frame_idx] = original_frame
        else:
            # Auto mode: only need first frame's FinalImage
            first_frame_idx = frame_indices[0]
            self.log_message.emit(f"Auto mode: Loading only first FinalImage (frame {first_frame_idx})...")
            original_frame = gbuffer_processor.load_original_frame(first_frame_idx)
            original_frame = original_frame.resize((512, 512), Image.LANCZOS)
            ground_truth_frames[first_frame_idx] = original_frame
            
        controlnet_scale_controller = self.config['controlnet_scale_controller']
        controllora_scale_controller = self.config['controllora_scale_controller']
        
        self.log_message.emit(f"Starting inference with batch size {batch_size}...")
        
        i = 0
        while i < len(frame_indices):
            if not self.is_running:
                self.log_message.emit("Inference stopped by user")
                break
            
            batch_end = min(i + batch_size, len(frame_indices))
            batch_indices = frame_indices[i:batch_end]
            batch_positions = list(range(i, batch_end))
            actual_batch_size = len(batch_indices)
            
            if actual_batch_size > 1:
                self.log_message.emit(f"Processing batch of {actual_batch_size} frames: {batch_indices}")
            
            if batch_positions[0] == 0:
                ref_frame_idx = batch_indices[0]
                ref_frame = ground_truth_frames[ref_frame_idx]
                batch_irradiance_source = None
            else:
                prev_frame_idx = frame_indices[batch_positions[0] - 1]
                
                if variant == 'gt':
                    ref_frame = ground_truth_frames[prev_frame_idx]
                else:
                    if prev_frame_idx not in self.generated_frames:
                        raise RuntimeError(
                            f"Auto mode: previous frame {prev_frame_idx} not found. "
                            f"Available: {sorted(self.generated_frames.keys())}"
                        )
                    ref_frame = self.generated_frames[prev_frame_idx]
                
                if variant == 'gt':
                    # GT mode: use GT from previous frame for consistency
                    batch_irradiance_source = {
                        'final': ground_truth_frames[prev_frame_idx],
                        'albedo': ground_truth_albedo[prev_frame_idx]
                    }
                else:
                    if prev_frame_idx not in self.generated_frames:
                        raise RuntimeError(
                            f"Auto mode: previous frame {prev_frame_idx} not found. "
                            f"Available: {sorted(self.generated_frames.keys())}"
                        )
                    batch_irradiance_source = {
                        'final': self.generated_frames[prev_frame_idx],
                        'albedo': ground_truth_albedo[prev_frame_idx]
                    }
            
            ref_frame = ref_frame.resize((512, 512), Image.LANCZOS)
            
            g_buffer_tensors = []
            controlnet_scales = []
            controllora_scales = []
            
            for j, frame_idx in enumerate(batch_indices):
                pos = batch_positions[j]
                
                controlnet_scale = controlnet_scale_controller.get_scale(pos, total_frames)
                controllora_scale = controllora_scale_controller.get_scale(pos, total_frames)
                controlnet_scales.append(controlnet_scale)
                controllora_scales.append(controllora_scale)
                
                self.log_message.emit(f"Frame {frame_idx}: CN={controlnet_scale:.3f}, CL={controllora_scale:.3f}")
                
                frame_irradiance_source = batch_irradiance_source

                
                g_buffer_tensor = gbuffer_processor.create_g_buffer_tensor(
                    frame_idx, 
                    apply_masking=True,
                    irradiance_source=frame_irradiance_source,
                    variant=variant
                )
                g_buffer_tensors.append(g_buffer_tensor)
            
            g_buffer_batch = torch.stack(g_buffer_tensors).to(device=self.device)
            
            avg_controlnet_scale = np.mean(controlnet_scales)
            avg_controllora_scale = np.mean(controllora_scales)
            
            self.pipeline.unet.set_extra_condition_scale(avg_controllora_scale)
            
            with torch.no_grad():
                batch_prompts = [self.config['prompt']] * actual_batch_size
                batch_negative_prompts = [self.config.get('negative_prompt', '')] * actual_batch_size
                batch_ref_frames = [ref_frame] * actual_batch_size
                
                pipeline_output = self.pipeline(
                    batch_prompts,
                    negative_prompt=batch_negative_prompts,
                    lora_images=batch_ref_frames,
                    controlnet_images=g_buffer_batch,
                    num_inference_steps=self.config.get('num_inference_steps', 20),
                    generator=self.generator,
                    controlnet_conditioning_scale=avg_controlnet_scale,
                    extra_condition_scale=avg_controllora_scale,
                    guidance_scale=self.config.get('guidance_scale', 1.0),
                    control_guidance_start=0.0,
                    control_guidance_end=1.0,
                    num_images_per_prompt=1,
                    output_type="latent" if self.config.get('save_latents', False) else "pil",
                )
                
                if self.config.get('save_latents', False):
                    batch_latents = pipeline_output.images
                    with torch.no_grad():
                        batch_images = self.pipeline.vae.decode(batch_latents / self.pipeline.vae.config.scaling_factor).sample
                        batch_images = (batch_images / 2 + 0.5).clamp(0, 1)
                        batch_images = [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for img in batch_images]
                else:
                    batch_images = pipeline_output.images
                    batch_latents = [None] * len(batch_images)
            
            batch_results = list(zip(batch_indices, batch_images))
            
            for frame_idx, result_image in batch_results:
                self.generated_frames[frame_idx] = result_image
                self.generated_frames_list.append((frame_idx, result_image))
                
                output_path = os.path.join(self.config['output_dir'], f"frame_{frame_idx:04d}.png")
                result_image.save(output_path)
                
                if self.config.get('save_latents', False):
                    latent_for_frame = batch_latents[batch_indices.index(frame_idx)]
                    if latent_for_frame is not None:
                        self.save_latent(latent_for_frame, frame_idx)
                
                self.frame_generated.emit(frame_idx, result_image)
                self.progress_update.emit(batch_positions[batch_indices.index(frame_idx)] + 1, total_frames)
            
            i = batch_end
        
        self.log_message.emit("Inference complete!")
        self.log_message.emit(f"Irradiance maps saved in: {gbuffer_processor.irradiance_dir}")
        if self.config.get('save_latents', False):
            self.log_message.emit(f"Output latents saved in: {self.latents_dir}")
        
    def get_frame_indices(self):
        """Get the list of frame indices to process based on direction and offset."""
        import glob
        import re
        
        basecolor_dir = os.path.join(self.config['gbuffer_dir'], "BaseColor")
        
        # Try old format first
        pattern = re.compile(rf"{re.escape(self.config['gbuffer_prefix'])}_FinalImageBaseColor_(\d+)\.png$")
        all_frame_indices = []
        
        for file_path in glob.glob(os.path.join(basecolor_dir, "*.png")):
            match = pattern.search(os.path.basename(file_path))
            if match:
                all_frame_indices.append(int(match.group(1)))
        
        # If no matches, try flat structure
        if not all_frame_indices:
            flat_pattern = re.compile(r'^BaseColor_(\d+)\.png$')
            for file_path in glob.glob(os.path.join(basecolor_dir, "*.png")):
                match = flat_pattern.match(os.path.basename(file_path))
                if match:
                    all_frame_indices.append(int(match.group(1)))
        
        all_frame_indices.sort()
        
        if not all_frame_indices:
            return []
        
        start_frame = self.config.get('start_frame', 0)
        num_frames = self.config.get('num_frames', 100)
        direction = self.config.get('direction', 'forward')
        
        if direction == 'backward':
            end_idx = len(all_frame_indices) - start_frame
            start_idx = max(0, end_idx - num_frames)
            frame_indices = all_frame_indices[start_idx:end_idx]
            frame_indices.reverse()
        else:
            start_idx = start_frame
            end_idx = min(start_idx + num_frames, len(all_frame_indices))
            frame_indices = all_frame_indices[start_idx:end_idx]
        
        if frame_indices:
            self.log_message.emit(f"Sequence has {len(all_frame_indices)} frames: {all_frame_indices[0]} to {all_frame_indices[-1]}")
            self.log_message.emit(f"Direction: {direction}, Start offset: {start_frame}, Num frames: {num_frames}")
            self.log_message.emit(f"Selected {len(frame_indices)} frames: {frame_indices[0]} to {frame_indices[-1]}")
            
        return frame_indices
    
    def save_latent(self, latent_tensor, frame_idx):
        """Save latent tensor to disk."""
        try:
            latent_cpu = latent_tensor.cpu()
            latent_path = os.path.join(self.latents_dir, f"latent_{frame_idx:04d}.pt")
            torch.save(latent_cpu, latent_path)
            
        except Exception as e:
            self.log_message.emit(f"Warning: Failed to save latent for frame {frame_idx}: {e}")


class ImageViewer(QWidget):
    """Widget for viewing generated images."""
    
    def __init__(self):
        super().__init__()
        self.images = {}
        self.current_index = 0
        self.frame_indices = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(512, 512)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #444;")
        
        controls_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous)
        
        self.frame_label = QLabel("Frame: 0")
        self.frame_label.setAlignment(Qt.AlignCenter)
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next)
        
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self.toggle_playback)
        
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(24)
        self.fps_spin.setPrefix("FPS: ")
        self.fps_spin.valueChanged.connect(self.update_playback_speed)
        
        controls_layout.addWidget(self.prev_button)
        controls_layout.addWidget(self.frame_label)
        controls_layout.addWidget(self.next_button)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.fps_spin)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        
        layout.addWidget(self.image_label)
        layout.addLayout(controls_layout)
        layout.addWidget(self.frame_slider)
        
        self.setLayout(layout)
        
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        
    def add_image(self, frame_idx, image):
        """Add a generated image."""
        self.images[frame_idx] = image
        self.frame_indices = sorted(self.images.keys())
        self.frame_slider.setMaximum(max(0, len(self.frame_indices) - 1))
        
        if self.frame_indices and frame_idx == self.frame_indices[self.current_index]:
            self.show_image(frame_idx)
        elif len(self.images) == 1:
            self.show_image(frame_idx)
            
    def show_image(self, frame_idx):
        """Display a specific frame."""
        if frame_idx not in self.images:
            return
        
        image = self.images[frame_idx]
        image_rgb = image.convert("RGB")
        data = image_rgb.tobytes()
        
        qimage = QImage(data, image.width, image.height, 3 * image.width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        
        scaled = pixmap.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.image_label.setPixmap(scaled)
        self.frame_label.setText(f"Frame: {frame_idx}")
        
        if frame_idx in self.frame_indices:
            self.current_index = self.frame_indices.index(frame_idx)
            self.frame_slider.setValue(self.current_index)
                
    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image(self.frame_indices[self.current_index])
            
    def show_next(self):
        if self.current_index < len(self.frame_indices) - 1:
            self.current_index += 1
            self.show_image(self.frame_indices[self.current_index])
            
    def next_frame(self):
        if self.current_index < len(self.frame_indices) - 1:
            self.show_next()
        else:
            self.current_index = 0
            if self.frame_indices:
                self.show_image(self.frame_indices[0])
                
    def slider_changed(self, value):
        if 0 <= value < len(self.frame_indices):
            self.current_index = value
            self.show_image(self.frame_indices[value])
            
    def toggle_playback(self, checked):
        if checked:
            fps = self.fps_spin.value()
            interval = int(1000 / fps)
            self.play_timer.start(interval)
            self.play_button.setText("Stop")
        else:
            self.play_timer.stop()
            self.play_button.setText("Play")
            
    def update_playback_speed(self):
        if self.play_timer.isActive():
            fps = self.fps_spin.value()
            interval = int(1000 / fps)
            self.play_timer.setInterval(interval)
            
    def clear_images(self):
        self.images = {}
        self.frame_indices = []
        self.current_index = 0
        self.image_label.clear()
        self.frame_slider.setMaximum(0)


class ScalePatternWidget(QWidget):
    """Widget for configuring scale patterns with visual graph editor."""
    
    pattern_changed = pyqtSignal()
    
    def __init__(self, name="Scale"):
        super().__init__()
        self.name = name
        self.controller = ScaleController()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Pattern configuration
        config_group = QGroupBox(f"{self.name} Pattern Configuration")
        config_layout = QVBoxLayout()
        
        # Pattern controls
        controls_layout = QGridLayout()
        
        # Pattern type
        controls_layout.addWidget(QLabel("Pattern:"), 0, 0)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["custom", "constant", "linear", "sawtooth", "square", "sine"])
        self.type_combo.currentTextChanged.connect(self.on_pattern_type_changed)
        controls_layout.addWidget(self.type_combo, 0, 1)
        
        # Start value
        controls_layout.addWidget(QLabel("Start Value:"), 0, 2)
        self.start_value_spin = QDoubleSpinBox()
        self.start_value_spin.setRange(0.0, 2.0)
        self.start_value_spin.setSingleStep(0.05)
        self.start_value_spin.setValue(1.0)
        self.start_value_spin.setDecimals(3)
        self.start_value_spin.valueChanged.connect(self.on_values_changed)
        controls_layout.addWidget(self.start_value_spin, 0, 3)
        
        # End value
        controls_layout.addWidget(QLabel("End Value:"), 1, 0)
        self.end_value_spin = QDoubleSpinBox()
        self.end_value_spin.setRange(0.0, 2.0)
        self.end_value_spin.setSingleStep(0.05)
        self.end_value_spin.setValue(1.0)
        self.end_value_spin.setDecimals(3)
        self.end_value_spin.valueChanged.connect(self.on_values_changed)
        controls_layout.addWidget(self.end_value_spin, 1, 1)
        
        # Period
        controls_layout.addWidget(QLabel("Period:"), 1, 2)
        self.period_spin = QSpinBox()
        self.period_spin.setRange(1, 999)
        self.period_spin.setValue(24)
        self.period_spin.valueChanged.connect(self.on_period_changed)
        controls_layout.addWidget(self.period_spin, 1, 3)
        
        config_layout.addLayout(controls_layout)
        
        # Description
        self.description_label = QLabel("Pattern: Custom")
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("QLabel { background-color: #353535; padding: 8px; border-radius: 4px; }")
        config_layout.addWidget(self.description_label)
        
        # Interactive graph
        self.graph_widget = InteractiveGraphWidget()
        self.graph_widget.values_changed.connect(self.on_graph_values_changed)
        config_layout.addWidget(self.graph_widget)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        self.setLayout(layout)
        
        # Update initial state
        self.on_pattern_type_changed()
        
    def on_pattern_type_changed(self):
        """Handle pattern type change."""
        pattern_type = self.type_combo.currentText()
        
        if pattern_type == 'custom':
            # Enable manual editing in graph
            self.start_value_spin.setEnabled(False)
            self.end_value_spin.setEnabled(False)
        else:
            # Generate pattern
            self.start_value_spin.setEnabled(True)
            self.end_value_spin.setEnabled(True)
            
            start_value = self.start_value_spin.value()
            end_value = self.end_value_spin.value()
            
            self.graph_widget.set_pattern(pattern_type, start_value, end_value)
            
        self.update_controller()
        
    def on_values_changed(self):
        """Handle value spin box changes."""
        if self.type_combo.currentText() != 'custom':
            pattern_type = self.type_combo.currentText()
            start_value = self.start_value_spin.value()
            end_value = self.end_value_spin.value()
            
            self.graph_widget.set_pattern(pattern_type, start_value, end_value)
            self.update_controller()
        
    def on_period_changed(self):
        """Handle period change."""
        period = self.period_spin.value()
        self.graph_widget.set_period(period)
        self.update_controller()
        
    def on_graph_values_changed(self):
        """Handle graph value changes."""
        self.type_combo.setCurrentText('custom')
        self.update_controller()
        
    def update_controller(self):
        """Update the controller with current values."""
        values = self.graph_widget.get_values()
        self.controller.set_custom_values(values)
        
        # Update description
        self.description_label.setText(f"Pattern: {self.controller.get_description()}")
        
        # Emit change signal
        self.pattern_changed.emit()
        
    def get_controller(self):
        """Get the scale controller."""
        return self.controller


class MainWindow(QMainWindow):
    """Main application window."""
    
    CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_configs.json")
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.configs = {}  # Store all saved configurations
        self.init_ui()
        self.load_configs()
        self.load_default_paths()
        
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("G-Buffer Inference")
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(700)
        
        # Tabs for organization
        tabs = QTabWidget()
        
        # === Path Configuration Tab ===
        paths_tab = QWidget()
        paths_layout = QVBoxLayout()
        
        # Configuration Management
        config_group = QGroupBox("Configuration Management")
        config_layout = QGridLayout()
        
        config_layout.addWidget(QLabel("Saved Configs:"), 0, 0)
        self.config_dropdown = QComboBox()
        self.config_dropdown.currentIndexChanged.connect(self.on_config_selected)
        config_layout.addWidget(self.config_dropdown, 0, 1)
        
        self.save_config_btn = QPushButton("Save Current Config")
        self.save_config_btn.clicked.connect(self.save_current_config)
        config_layout.addWidget(self.save_config_btn, 0, 2)
        
        config_group.setLayout(config_layout)
        paths_layout.addWidget(config_group)
        
        # Dataset paths
        dataset_group = QGroupBox("Dataset Paths")
        dataset_layout = QGridLayout()
        
        dataset_layout.addWidget(QLabel("G-Buffer Dir:"), 0, 0)
        self.gbuffer_dir_edit = QLineEdit()
        self.gbuffer_dir_edit.textChanged.connect(self.on_gbuffer_dir_changed)
        dataset_layout.addWidget(self.gbuffer_dir_edit, 0, 1)
        self.gbuffer_dir_btn = QPushButton("Browse")
        self.gbuffer_dir_btn.clicked.connect(lambda: self.browse_folder(self.gbuffer_dir_edit))
        dataset_layout.addWidget(self.gbuffer_dir_btn, 0, 2)
        
        dataset_layout.addWidget(QLabel("Original Frames:"), 1, 0)
        self.original_frames_edit = QLineEdit()
        dataset_layout.addWidget(self.original_frames_edit, 1, 1)
        self.original_frames_btn = QPushButton("Browse")
        self.original_frames_btn.clicked.connect(lambda: self.browse_folder(self.original_frames_edit))
        dataset_layout.addWidget(self.original_frames_btn, 1, 2)
        
        dataset_layout.addWidget(QLabel("G-Buffer Prefix:"), 2, 0)
        self.gbuffer_prefix_edit = QLineEdit()
        dataset_layout.addWidget(self.gbuffer_prefix_edit, 2, 1, 1, 2)
        
        dataset_group.setLayout(dataset_layout)
        paths_layout.addWidget(dataset_group)
        
        # BaseColor Injection settings
        injection_group = QGroupBox("BaseColor Injection (Sky Fill)")
        injection_layout = QGridLayout()
        
        self.enable_injection_checkbox = QCheckBox("Enable BaseColor Injection")
        self.enable_injection_checkbox.setChecked(True)
        self.enable_injection_checkbox.setToolTip("Fill black pixels in BaseColor with colors from injection source images")
        injection_layout.addWidget(self.enable_injection_checkbox, 0, 0, 1, 3)
        
        injection_layout.addWidget(QLabel("Injection Source:"), 1, 0)
        self.injection_source_edit = QLineEdit()
        self.injection_source_edit.setToolTip("Path to FinalImage frames for filling sky in BaseColor (defaults to G-Buffer Dir/FinalImage)")
        injection_layout.addWidget(self.injection_source_edit, 1, 1)
        self.injection_source_btn = QPushButton("Browse")
        self.injection_source_btn.clicked.connect(lambda: self.browse_folder(self.injection_source_edit))
        injection_layout.addWidget(self.injection_source_btn, 1, 2)
        
        injection_group.setLayout(injection_layout)
        paths_layout.addWidget(injection_group)
        
        # Model paths
        model_group = QGroupBox("Model Paths")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("ControlNet:"), 0, 0)
        self.controlnet_path_edit = QLineEdit()
        model_layout.addWidget(self.controlnet_path_edit, 0, 1)
        self.controlnet_path_btn = QPushButton("Browse")
        self.controlnet_path_btn.clicked.connect(lambda: self.browse_folder(self.controlnet_path_edit))
        model_layout.addWidget(self.controlnet_path_btn, 0, 2)
        
        model_layout.addWidget(QLabel("ControlLoRA:"), 1, 0)
        self.controllora_path_edit = QLineEdit()
        model_layout.addWidget(self.controllora_path_edit, 1, 1)
        self.controllora_path_btn = QPushButton("Browse")
        self.controllora_path_btn.clicked.connect(lambda: self.browse_file(self.controllora_path_edit))
        model_layout.addWidget(self.controllora_path_btn, 1, 2)
        
        model_layout.addWidget(QLabel("Base Output Dir:"), 2, 0)
        self.output_dir_edit = QLineEdit()
        model_layout.addWidget(self.output_dir_edit, 2, 1)
        self.output_dir_btn = QPushButton("Browse")
        self.output_dir_btn.clicked.connect(lambda: self.browse_folder(self.output_dir_edit))
        model_layout.addWidget(self.output_dir_btn, 2, 2)
        
        model_group.setLayout(model_layout)
        paths_layout.addWidget(model_group)
        
        paths_layout.addStretch()
        paths_tab.setLayout(paths_layout)
        tabs.addTab(paths_tab, "Paths")
        
        # === Inference Settings Tab ===
        inference_tab = QWidget()
        inference_layout = QVBoxLayout()
        
        # Frame settings
        frame_group = QGroupBox("Frame Settings")
        frame_layout = QGridLayout()
        
        frame_layout.addWidget(QLabel("Skip First Frames:"), 0, 0)
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, 9999)
        self.start_frame_spin.setToolTip("Number of frames to skip at the beginning of the sequence (for forward) or end (for backward)")
        frame_layout.addWidget(self.start_frame_spin, 0, 1)
        
        frame_layout.addWidget(QLabel("Num Frames:"), 1, 0)
        self.num_frames_spin = QSpinBox()
        self.num_frames_spin.setRange(1, 9999)
        self.num_frames_spin.setValue(100)
        frame_layout.addWidget(self.num_frames_spin, 1, 1)
        
        frame_layout.addWidget(QLabel("Direction:"), 2, 0)
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["forward", "backward"])
        frame_layout.addWidget(self.direction_combo, 2, 1)
        
        frame_layout.addWidget(QLabel("Variant:"), 3, 0)
        self.variant_combo = QComboBox()
        self.variant_combo.addItems(["auto", "gt"])
        frame_layout.addWidget(self.variant_combo, 3, 1)
        
        frame_group.setLayout(frame_layout)
        inference_layout.addWidget(frame_group)
        
        # Generation settings
        gen_group = QGroupBox("Generation Settings")
        gen_layout = QGridLayout()
        
        # Prompt as multi-line text edit
        gen_layout.addWidget(QLabel("Prompt:"), 0, 0, Qt.AlignTop)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setMaximumHeight(60)
        gen_layout.addWidget(self.prompt_edit, 0, 1, 1, 2)
        
        # Negative prompt as multi-line text edit
        gen_layout.addWidget(QLabel("Negative:"), 1, 0, Qt.AlignTop)
        self.negative_prompt_edit = QTextEdit()
        self.negative_prompt_edit.setPlainText("lowres, blurry, worst quality")
        self.negative_prompt_edit.setMaximumHeight(60)
        gen_layout.addWidget(self.negative_prompt_edit, 1, 1, 1, 2)
        
        gen_layout.addWidget(QLabel("Steps:"), 2, 0)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(10)
        gen_layout.addWidget(self.steps_spin, 2, 1)
        
        gen_layout.addWidget(QLabel("Guidance:"), 3, 0)
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 20.0)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setValue(1.0)
        gen_layout.addWidget(self.guidance_spin, 3, 1)
        
        gen_layout.addWidget(QLabel("Seed:"), 4, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        gen_layout.addWidget(self.seed_spin, 4, 1)
        
        gen_group.setLayout(gen_layout)
        inference_layout.addWidget(gen_group)
        
        # Performance settings
        perf_group = QGroupBox("Performance Settings")
        perf_layout = QGridLayout()
        
        perf_layout.addWidget(QLabel("Batch Size:"), 0, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 16)
        self.batch_size_spin.setValue(1)
        perf_layout.addWidget(self.batch_size_spin, 0, 1)
        
        perf_layout.addWidget(QLabel("Mixed Precision:"), 1, 0)
        self.mixed_precision_combo = QComboBox()
        self.mixed_precision_combo.addItems(["bf16", "fp16", "no"])
        perf_layout.addWidget(self.mixed_precision_combo, 1, 1)
        
        perf_group.setLayout(perf_layout)
        inference_layout.addWidget(perf_group)
        
        inference_layout.addStretch()
        inference_tab.setLayout(inference_layout)
        tabs.addTab(inference_tab, "Inference")
        
        # === Scale Control Tab ===
        scale_tab = QWidget()
        scale_layout = QVBoxLayout()
        
        # ControlNet scale pattern
        self.controlnet_scale_widget = ScalePatternWidget("ControlNet")
        scale_layout.addWidget(self.controlnet_scale_widget)
        
        # ControlLoRA scale pattern
        self.controllora_scale_widget = ScalePatternWidget("ControlLoRA")
        scale_layout.addWidget(self.controllora_scale_widget)
        
        scale_tab.setLayout(scale_layout)
        tabs.addTab(scale_tab, "Scale Control")
        
        left_layout.addWidget(tabs)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        # Comparison and Export group
        comparison_group = QGroupBox("VAE Comparison & Export")
        comparison_layout = QVBoxLayout()
        
        # Save latents checkbox
        self.save_latents_checkbox = QCheckBox("Save Output Latents (.pt)")
        self.save_latents_checkbox.setToolTip("Save the VAE latent representations of model outputs as .pt files")
        comparison_layout.addWidget(self.save_latents_checkbox)
        
        # Resize options
        resize_layout = QHBoxLayout()
        resize_layout.addWidget(QLabel("Output Size:"))
        
        self.resize_width_spin = QSpinBox()
        self.resize_width_spin.setRange(128, 2048)
        self.resize_width_spin.setValue(512)
        self.resize_width_spin.setSingleStep(64)
        resize_layout.addWidget(QLabel("W:"))
        resize_layout.addWidget(self.resize_width_spin)
        
        self.resize_height_spin = QSpinBox()
        self.resize_height_spin.setRange(128, 2048)
        self.resize_height_spin.setValue(512)
        self.resize_height_spin.setSingleStep(64)
        resize_layout.addWidget(QLabel("H:"))
        resize_layout.addWidget(self.resize_height_spin)
        resize_layout.addStretch()
        
        comparison_layout.addLayout(resize_layout)
        
        # Generate comparisons button
        self.generate_comparison_button = QPushButton("Generate VAE Comparisons")
        self.generate_comparison_button.clicked.connect(self.generate_vae_comparisons)
        comparison_layout.addWidget(self.generate_comparison_button)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_output_button = QPushButton("Export Output Video")
        self.export_output_button.clicked.connect(lambda: self.export_video(comparison_mode=False))
        export_layout.addWidget(self.export_output_button)
        
        self.export_comparison_button = QPushButton("Export Comparison Video")
        self.export_comparison_button.clicked.connect(lambda: self.export_video(comparison_mode=True))
        export_layout.addWidget(self.export_comparison_button)
        
        comparison_layout.addLayout(export_layout)
        comparison_group.setLayout(comparison_layout)
        left_layout.addWidget(comparison_group)
        
        self.start_button = QPushButton("Start Inference")
        self.start_button.clicked.connect(self.start_inference)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_inference)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        left_layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)
        
        # Right panel - Viewer and Log
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Splitter for viewer and log
        splitter = QSplitter(Qt.Vertical)
        
        # Image viewer
        self.image_viewer = ImageViewer()
        splitter.addWidget(self.image_viewer)
        
        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        splitter.addWidget(self.log_text)
        
        splitter.setSizes([700, 200])
        right_layout.addWidget(splitter)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # Track current output directory
        self.current_output_dir = None
        
    def load_default_paths(self):
        """Load default paths (empty - use saved configs)."""
        pass
    
    def load_configs(self):
        """Load saved configurations from JSON file."""
        try:
            if os.path.exists(self.CONFIG_FILE):
                config_dir = os.path.dirname(os.path.abspath(self.CONFIG_FILE))
                
                with open(self.CONFIG_FILE, 'r') as f:
                    self.configs = json.load(f)
                
                # Resolve relative paths
                path_keys = ['gbuffer_dir', 'original_frames', 'original_frames_dir', 
                             'controlnet_path', 'controllora_path', 'output_dir',
                             'injection_source_dir']
                
                for config_name, config in self.configs.items():
                    for key in path_keys:
                        if key in config and config[key]:
                            path = config[key]
                            if not os.path.isabs(path):
                                config[key] = os.path.normpath(os.path.join(config_dir, path))
                
                self.log_message(f"Loaded {len(self.configs)} saved configurations")
            else:
                self.configs = {}
                self.log_message("No saved configurations found")
            
            self.populate_config_dropdown()
        except Exception as e:
            self.log_message(f"Error loading configurations: {e}")
            self.configs = {}
    
    def save_configs(self):
        """Save configurations to JSON file."""
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(self.configs, f, indent=2)
            self.log_message(f"Saved {len(self.configs)} configurations to {self.CONFIG_FILE}")
        except Exception as e:
            self.log_message(f"Error saving configurations: {e}")
            QMessageBox.warning(self, "Save Error", f"Failed to save configurations: {e}")
    
    def save_current_config(self):
        """Save current UI state as a configuration."""
        from PyQt5.QtWidgets import QInputDialog
        
        # Ask user for config name
        config_name, ok = QInputDialog.getText(
            self,
            "Save Configuration",
            "Enter a name for this configuration:",
            QLineEdit.Normal,
            ""
        )
        
        if not ok or not config_name.strip():
            return
        
        config_name = config_name.strip()
        
        # Check if config already exists
        if config_name in self.configs:
            reply = QMessageBox.question(
                self,
                "Config Exists",
                f"Configuration '{config_name}' already exists.\n\nDo you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # Save current state
        config = {
            'gbuffer_dir': self.gbuffer_dir_edit.text(),
            'original_frames': self.original_frames_edit.text(),
            'gbuffer_prefix': self.gbuffer_prefix_edit.text(),
            'controlnet_path': self.controlnet_path_edit.text(),
            'controllora_path': self.controllora_path_edit.text(),
            'output_dir': self.output_dir_edit.text(),
            'prompt': self.prompt_edit.toPlainText(),
            'negative_prompt': self.negative_prompt_edit.toPlainText(),
            'enable_injection': self.enable_injection_checkbox.isChecked(),
            'injection_source_dir': self.injection_source_edit.text()
        }
        
        self.configs[config_name] = config
        self.save_configs()
        self.populate_config_dropdown()
        
        # Select the newly saved config
        index = self.config_dropdown.findText(config_name)
        if index >= 0:
            self.config_dropdown.setCurrentIndex(index)
        
        QMessageBox.information(self, "Config Saved", f"Configuration '{config_name}' saved successfully!")
    
    def on_config_selected(self, index):
        """Load selected configuration into UI."""
        if index < 0:
            return
        
        config_name = self.config_dropdown.currentText()
        if config_name == "-- Select a configuration --" or config_name not in self.configs:
            return
        
        config = self.configs[config_name]
        
        # Clear injection source before setting gbuffer_dir so auto-populate works
        self.injection_source_edit.clear()
        self.original_frames_edit.clear()
        
        # Populate UI with config values
        self.gbuffer_dir_edit.setText(config.get('gbuffer_dir', ''))
        self.original_frames_edit.setText(config.get('original_frames', ''))
        self.gbuffer_prefix_edit.setText(config.get('gbuffer_prefix', ''))
        self.controlnet_path_edit.setText(config.get('controlnet_path', ''))
        self.controllora_path_edit.setText(config.get('controllora_path', ''))
        self.output_dir_edit.setText(config.get('output_dir', ''))
        self.prompt_edit.setPlainText(config.get('prompt', ''))
        self.negative_prompt_edit.setPlainText(config.get('negative_prompt', ''))
        
        # Load injection settings (enabled by default)
        self.enable_injection_checkbox.setChecked(config.get('enable_injection', True))
        # Override injection_source if config has explicit value
        config_injection_source = config.get('injection_source_dir', '')
        if config_injection_source:
            self.injection_source_edit.setText(config_injection_source)
        
        self.log_message(f"Loaded configuration: {config_name}")
    
    def populate_config_dropdown(self):
        """Update config dropdown with available configurations."""
        current_text = self.config_dropdown.currentText()
        self.config_dropdown.blockSignals(True)
        self.config_dropdown.clear()
        
        # Add placeholder
        self.config_dropdown.addItem("-- Select a configuration --")
        
        # Add all saved configs
        for config_name in sorted(self.configs.keys()):
            self.config_dropdown.addItem(config_name)
        
        # Restore selection if it still exists
        if current_text in self.configs:
            index = self.config_dropdown.findText(current_text)
            if index >= 0:
                self.config_dropdown.setCurrentIndex(index)
        
        self.config_dropdown.blockSignals(False)
        
    def browse_folder(self, line_edit):
        """Browse for a folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text())
        if folder:
            line_edit.setText(folder)
            
    def browse_file(self, line_edit):
        """Browse for a file."""
        file, _ = QFileDialog.getOpenFileName(self, "Select File", line_edit.text())
        if file:
            line_edit.setText(file)
    
    def on_gbuffer_dir_changed(self, text):
        """Auto-populate injection source and original frames when G-Buffer dir changes."""
        if text:
            # Auto-populate injection source with FinalImage subfolder
            final_image_path = os.path.join(text, 'FinalImage')
            if not self.injection_source_edit.text():
                self.injection_source_edit.setText(final_image_path)
            # Auto-populate original frames if empty
            if not self.original_frames_edit.text():
                self.original_frames_edit.setText(final_image_path)
            
    def start_inference(self):
        """Start the inference process."""
        # Validate paths
        if not all([
            os.path.exists(self.gbuffer_dir_edit.text()),
            os.path.exists(self.original_frames_edit.text()),
            os.path.exists(self.controlnet_path_edit.text()),
            os.path.exists(self.controllora_path_edit.text())
        ]):
            QMessageBox.warning(self, "Error", "Please check all paths are valid!")
            return
            
        # Create timestamped output directory
        base_output_dir = self.output_dir_edit.text()
        if not base_output_dir:
            base_output_dir = "output"
        
        # Create subfolder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_output_dir = os.path.join(base_output_dir, f"inference_{timestamp}")
        os.makedirs(self.current_output_dir, exist_ok=True)
        self.log_message(f"Output directory: {self.current_output_dir}")
        
        # Clear previous results
        self.image_viewer.clear_images()
        self.log_text.clear()
        
        # Prepare configuration
        config = {
            'gbuffer_dir': self.gbuffer_dir_edit.text(),
            'original_frames_dir': self.original_frames_edit.text(),
            'gbuffer_prefix': self.gbuffer_prefix_edit.text(),
            'controlnet_path': self.controlnet_path_edit.text(),
            'controllora_path': self.controllora_path_edit.text(),
            'output_dir': self.current_output_dir,
            'start_frame': self.start_frame_spin.value(),
            'num_frames': self.num_frames_spin.value(),
            'direction': self.direction_combo.currentText(),
            'variant': self.variant_combo.currentText(),
            'prompt': self.prompt_edit.toPlainText(),
            'negative_prompt': self.negative_prompt_edit.toPlainText(),
            'num_inference_steps': self.steps_spin.value(),
            'guidance_scale': self.guidance_spin.value(),
            'seed': self.seed_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'mixed_precision': self.mixed_precision_combo.currentText(),
            'controlnet_scale_controller': self.controlnet_scale_widget.get_controller(),
            'controllora_scale_controller': self.controllora_scale_widget.get_controller(),
            'gpu_id': 0,
            'disable_xformers': False,
            'lora_adapter_name': 'gbuffer',
            'base_model': 'runwayml/stable-diffusion-v1-5',
            'save_latents': self.save_latents_checkbox.isChecked(),
            'enable_injection': self.enable_injection_checkbox.isChecked(),
            'injection_source_dir': self.injection_source_edit.text()
        }
        
        # Create and start worker
        self.worker = InferenceWorker(config)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.frame_generated.connect(self.on_frame_generated)
        self.worker.log_message.connect(self.log_message)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.error_occurred.connect(self.on_error)
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Start worker
        self.worker.start()
        
    def stop_inference(self):
        """Stop the inference process."""
        if self.worker:
            self.worker.stop()
            self.log_message("Stopping inference...")
            
    def update_progress(self, current, total):
        """Update progress bar."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
    def on_frame_generated(self, frame_idx, image):
        """Handle newly generated frame."""
        self.image_viewer.add_image(frame_idx, image)
        
    def log_message(self, message):
        """Add message to log."""
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        
    def on_inference_finished(self):
        """Handle inference completion."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_message("Inference completed!")
        QMessageBox.information(self, "Complete", "Inference completed successfully!")
        
    def on_error(self, error_msg):
        """Handle inference error."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_message(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")
        
    def export_video(self, comparison_mode=False):
        """Export frames as video with Windows-compatible FFmpeg command."""
        if not self.current_output_dir or not os.path.exists(self.current_output_dir):
            QMessageBox.warning(self, "Warning", "No output directory found!")
            return
        
        # Determine which frames to export
        import glob
        import re
        if comparison_mode:
            search_path = os.path.join(self.current_output_dir, "comparison_*.png")
            self.log_message(f"Searching for comparison frames: {search_path}")
            frames = sorted(glob.glob(search_path))
            self.log_message(f"Found {len(frames)} comparison frames")
            if len(frames) > 0:
                self.log_message(f"First frame: {frames[0]}")
                self.log_message(f"Last frame: {frames[-1]}")
            else:
                all_files = os.listdir(self.current_output_dir)
                comparison_files = [f for f in all_files if f.startswith('comparison_')]
                self.log_message(f"All files in directory: {len(all_files)}")
                self.log_message(f"Files starting with 'comparison_': {comparison_files}")
            if not frames:
                QMessageBox.warning(self, "Warning", "No comparison frames found! Generate comparisons first.")
                return
        else:
            frames = sorted(glob.glob(os.path.join(self.current_output_dir, "frame_*.png")))
            if not frames:
                QMessageBox.warning(self, "Warning", "No frames found to export!")
                return
        
        # Get FPS and output path
        fps = self.image_viewer.fps_spin.value()
        default_name = "comparison_video.mp4" if comparison_mode else "output_video.mp4"
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Video", default_name, "MP4 (*.mp4)")
        
        if not output_path:
            return
        
        # Simple regex that matches underscore followed by digits
        match = re.search(r'_(\d+)\.png$', first_frame_name)
        
        if match:
            start_number = int(match.group(1))
            num_digits = len(match.group(1))
            self.log_message(f"DEBUG: Match found! start_number={start_number}, num_digits={num_digits}")
        else:
            start_number = 0
            num_digits = 4
            self.log_message(f"DEBUG: NO MATCH! Using defaults")
        
        # Create frame pattern
        if comparison_mode:
            frame_pattern = f"comparison_%0{num_digits}d.png"
        else:
            frame_pattern = f"frame_%0{num_digits}d.png"
        
        self.log_message(f"DEBUG: Frame pattern: {frame_pattern}")
        input_pattern = os.path.join(self.current_output_dir, frame_pattern)
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-y', 
            '-start_number', str(start_number),
            '-framerate', str(fps),
            '-i', input_pattern,
        ]
        
        # Add resize filter if specified
        if hasattr(self, 'resize_width_spin') and hasattr(self, 'resize_height_spin'):
            width = self.resize_width_spin.value()
            height = self.resize_height_spin.value()
            if width != 512 or height != 512:
                cmd.extend(['-vf', f'scale={width}:{height}'])
        
        cmd.extend([
            '-c:v', 'libx264', 
            '-preset', 'slow', 
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_path
        ])
        
        try:
            self.log_message(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.log_message(f"Video exported to: {output_path}")
            QMessageBox.information(self, "Success", f"Video exported to:\n{output_path}")
        except subprocess.CalledProcessError as e:
            self.log_message(f"Export failed: {e}")
            self.log_message(f"stderr: {e.stderr}")
            QMessageBox.critical(self, "Error", f"Failed to export video:\n{str(e)}\n\nCheck if ffmpeg is installed and in PATH")
        except FileNotFoundError:
            self.log_message("ffmpeg not found")
            QMessageBox.critical(self, "Error", "ffmpeg not found. Please install ffmpeg and add it to PATH")
    
    def generate_vae_comparisons(self):
        """Generate VAE-reconstructed GT frames and create side-by-side comparisons."""
        if not self.current_output_dir or not os.path.exists(self.current_output_dir):
            QMessageBox.warning(self, "Warning", "No output directory found!")
            return
        
        # Check if we have generated frames
        import glob
        generated_frames = sorted(glob.glob(os.path.join(self.current_output_dir, "frame_*.png")))
        if not generated_frames:
            QMessageBox.warning(self, "Warning", "No generated frames found!")
            return
        
        self.log_message("Starting VAE comparison generation...")
        self.log_message(f"Output directory: {self.current_output_dir}")
        self.log_message(f"Found {len(generated_frames)} generated frames")
        if len(generated_frames) > 0:
            self.log_message(f"First frame: {os.path.basename(generated_frames[0])}")
            self.log_message(f"Last frame: {os.path.basename(generated_frames[-1])}")
        
        # Create comparison worker thread
        config = {
            'output_dir': self.current_output_dir,
            'original_frames_dir': self.original_frames_edit.text(),
            'gbuffer_prefix': self.gbuffer_prefix_edit.text(),  # Add prefix for correct filename
            'base_model': 'runwayml/stable-diffusion-v1-5',
            'generated_frames': generated_frames,
            'resize_width': self.resize_width_spin.value() if hasattr(self, 'resize_width_spin') else 512,
            'resize_height': self.resize_height_spin.value() if hasattr(self, 'resize_height_spin') else 512,
        }
        
        self.comparison_worker = VAEComparisonWorker(config)
        self.comparison_worker.progress_update.connect(self.update_progress)
        self.comparison_worker.log_message.connect(self.log_message)
        self.comparison_worker.finished.connect(self.on_comparison_finished)
        self.comparison_worker.error_occurred.connect(self.on_error)
        
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.comparison_worker.start()
    
    def on_comparison_finished(self):
        """Handle comparison generation completion."""
        self.start_button.setEnabled(True)
        self.log_message("VAE comparison generation completed!")
        QMessageBox.information(self, "Complete", "VAE comparisons generated successfully!")


def main():
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
