"""
FrameDiffuser Gradient Monitor.

Utility for monitoring gradient norms during training.

Copyright (c) 2025 Ole Beisswenger, Jan-Niklas Dihlmann, Hendrik Lensch
Licensed under MIT License.
"""

import os
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


class GradientMonitor:
    """Monitors gradients during training for debugging and analysis."""
    
    def __init__(self, output_dir="./gradient_logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.step_data = defaultdict(list)
        self.step_history = []
        
        self.unet = None
        self.controlnet = None
        
        self._last_log_time = 0
        self._log_interval = 60.0
        
    def register_models(self, unet=None, controlnet=None):
        """Register models to monitor."""
        self.unet = unet
        self.controlnet = controlnet
    
    def log_gradients(self, step):
        """Log gradient norms for registered models."""
        self.step_history.append(step)
        
        current_time = time.time()
        should_print = (current_time - self._last_log_time) > self._log_interval
        
        # Monitor UNet LoRA gradients
        if self.unet is not None:
            lora_a_grads = []
            lora_b_grads = []
            
            for name, param in self.unet.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if ".lora_A." in name:
                        lora_a_grads.append(param.grad.norm().item())
                    elif ".lora_B." in name:
                        lora_b_grads.append(param.grad.norm().item())
            
            if lora_a_grads:
                self.step_data["lora_a_mean"].append(np.mean(lora_a_grads))
                self.step_data["lora_a_max"].append(np.max(lora_a_grads))
            
            if lora_b_grads:
                self.step_data["lora_b_mean"].append(np.mean(lora_b_grads))
                self.step_data["lora_b_max"].append(np.max(lora_b_grads))
        
        # Monitor ControlNet gradients
        if self.controlnet is not None:
            controlnet_grads = []
            for name, param in self.controlnet.named_parameters():
                if param.requires_grad and param.grad is not None:
                    controlnet_grads.append(param.grad.norm().item())
            
            if controlnet_grads:
                self.step_data["controlnet_mean"].append(np.mean(controlnet_grads))
                self.step_data["controlnet_max"].append(np.max(controlnet_grads))
        
        if should_print:
            self._print_status(step)
            self._last_log_time = current_time
    
    def _print_status(self, step):
        """Print current gradient status."""
        print(f"\nGradients @ step {step}:")
        
        if "lora_a_mean" in self.step_data and self.step_data["lora_a_mean"]:
            print(f"  LoRA A: {self.step_data['lora_a_mean'][-1]:.6f} mean, {self.step_data['lora_a_max'][-1]:.6f} max")
        
        if "lora_b_mean" in self.step_data and self.step_data["lora_b_mean"]:
            print(f"  LoRA B: {self.step_data['lora_b_mean'][-1]:.6f} mean, {self.step_data['lora_b_max'][-1]:.6f} max")
        
        if "controlnet_mean" in self.step_data and self.step_data["controlnet_mean"]:
            print(f"  ControlNet: {self.step_data['controlnet_mean'][-1]:.6f} mean, {self.step_data['controlnet_max'][-1]:.6f} max")
    
    def save_plots(self):
        """Save gradient plots to output directory."""
        if not self.step_history:
            return
        
        # ControlLoRA gradients
        if "lora_a_mean" in self.step_data:
            plt.figure(figsize=(12, 6))
            plt.plot(self.step_history, self.step_data["lora_a_mean"], label="LoRA A Mean")
            plt.plot(self.step_history, self.step_data["lora_b_mean"], label="LoRA B Mean")
            plt.title("ControlLoRA Gradient Norms")
            plt.xlabel("Step")
            plt.ylabel("Gradient Norm")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "controllora_gradients.png"))
            plt.close()
        
        # ControlNet gradients
        if "controlnet_mean" in self.step_data:
            plt.figure(figsize=(12, 6))
            plt.plot(self.step_history, self.step_data["controlnet_mean"], label="ControlNet Mean")
            plt.title("ControlNet Gradient Norms")
            plt.xlabel("Step")
            plt.ylabel("Gradient Norm")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "controlnet_gradients.png"))
            plt.close()
