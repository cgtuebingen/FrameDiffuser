"""
FrameDiffuser G-buffer Masking Utilities.

Handles sky region detection and basecolor filling.

Copyright (c) 2025 Ole Beisswenger, Jan-Niklas Dihlmann, Hendrik Lensch
Licensed under MIT License.
"""

import torch
import numpy as np
from PIL import Image


def fill_black_pixels_in_basecolor(base_color_image, depth_image, final_image):
    """
    Fill black pixels in basecolor where both basecolor AND depth are black.
    This handles sky regions where deferred rendering provides no geometric data.
    
    Args:
        base_color_image: BaseColor G-buffer (PIL Image, numpy array, or torch Tensor)
        depth_image: Depth G-buffer (PIL Image, numpy array, or torch Tensor)
        final_image: Final rendered image to sample sky colors from
        
    Returns:
        PIL Image with sky regions filled from final_image
    """
    # Convert basecolor to numpy
    if isinstance(base_color_image, torch.Tensor):
        base_color_np = ((base_color_image.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
    elif isinstance(base_color_image, Image.Image):
        base_color_np = np.array(base_color_image.convert('RGB'))
    else:
        base_color_np = np.array(base_color_image)
    
    # Convert depth to numpy
    if isinstance(depth_image, torch.Tensor):
        if depth_image.shape[0] == 1:
            depth_np = ((depth_image[0].cpu().numpy() + 1) * 127.5).astype(np.uint8)
        else:
            depth_np = ((depth_image[0].cpu().numpy() + 1) * 127.5).astype(np.uint8)
    elif isinstance(depth_image, Image.Image):
        depth_np = np.array(depth_image.convert('L'))
    else:
        depth_np = np.array(depth_image)
    
    # Convert final image to numpy
    if isinstance(final_image, torch.Tensor):
        final_np = ((final_image.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
    elif isinstance(final_image, Image.Image):
        final_np = np.array(final_image.convert('RGB'))
    else:
        final_np = np.array(final_image)
    
    # Threshold for detecting black pixels (sky regions)
    threshold = 10
    base_color_black = np.all(base_color_np < threshold, axis=2)
    depth_black = depth_np < threshold
    
    # Mask: true where both basecolor and depth are black (sky regions)
    mask = base_color_black & depth_black
    
    # Fill sky regions with final image pixels
    result = base_color_np.copy()
    result[mask] = final_np[mask]
    
    return Image.fromarray(result)
