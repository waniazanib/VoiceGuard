"""
Post-hoc model interpretability using GradCAM (Gradient-weighted Class Activation Mapping).
Highlights regions in the CQT spectrogram that contribute most to the Spoof vs Bonafide decision.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    """
    Computes GradCAM for the final convolutional layers of the SpoofDetector model.
    Enables explainable AI output showing why a neural net detected spoof features.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initializes the GradCAM generator.
        
        Args:
            model: PyTorch model instance (SpoofDetector).
            target_layer: The NN module representing the last conv block (e.g. model.layer4).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
        self._register_hooks()
        
    def _register_hooks(self):
        """Registers forward and backward hooks to intercept activations and gradients."""
        def forward_hook(module, tf_in, tf_out):
            self.activations = tf_out
            
        def backward_hook(module, grad_in, grad_out):
            # grad_out represents the gradient with respect to output tensor
            self.gradients = grad_out[0]
            
        # Hook activations and gradients
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))
        else:
            self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))
            
    def remove_hooks(self):
        """Cleans hooks up after usage to prevent memory leakage and forward interference."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generates a normalized GradCAM heatmap of dimensions matching the input_tensor.
        
        Args:
            input_tensor: PyTorch tensor of shape (1, 1, CQT_BINS, T).
            class_idx: Target index (0=Bonafide, 1=Spoof). If None, defaults to predicted class.
            
        Returns:
            heatmap: 2D float array of shape (CQT_BINS, T) normalized to [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()
        
        # 1. Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
            
        # 2. Backward pass with respect to predicted class logit
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 3. Compute weights via global average pooling of feature gradients
        grads = self.gradients.detach().cpu().numpy()[0]  # Shape: (512, H, W)
        acts = self.activations.detach().cpu().numpy()[0]  # Shape: (512, H, W)
        
        # Average gradients spatially
        weights = np.mean(grads, axis=(1, 2))  # Shape: (512,)
        
        # 4. Integrate channels with weights
        cam = np.zeros(acts.shape[1:], dtype=np.float32)  # Shape: (H, W)
        for i, w in enumerate(weights):
            cam += w * acts[i]
            
        # 5. Rectified Linear Unit (ReLU) activation to capture positive attributes
        cam = np.maximum(cam, 0)
        
        # Prevent division by zero
        cam_max = np.max(cam)
        if cam_max > 1e-8:
            cam = cam / cam_max
            
        # 6. Bilinearly upscale heatmap to fit original input dimensions
        h_orig, w_orig = input_tensor.shape[2], input_tensor.shape[3]
        cam_pil = Image.fromarray((cam * 255.0).astype(np.uint8))
        cam_resized = np.array(cam_pil.resize((w_orig, h_orig), resample=Image.Resampling.BILINEAR))
        cam_normalized = cam_resized / 255.0
        
        return cam_normalized


def overlay_on_spectrogram(spec: np.ndarray, heatmap: np.ndarray) -> plt.Figure:
    """
    Renders a Matplotlib figure featuring 3 subplots side-by-side:
    1. CQT spectrogram (viridis)
    2. Heatmap projection (jet)
    3. Blended Overlay of spectrogram and heatmap
    
    Args:
        spec: Original normalized 2D Constant-Q spectrogram (bins, frames).
        heatmap: Up-sampled 2D GradCAM heatmap of shape (bins, frames).
        
    Returns:
        fig: Complete Matplotlib Figure object ready to render in the UI.
    """
    # Create side-by-side figure layout
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    
    # Custom display settings (Olive leaf, Black forest themed colors)
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#283618")  # Dark forest background
    
    for ax in axes:
        ax.set_facecolor("#283618")
        ax.tick_params(colors="#fefae0", labelsize=9)
        # Style axes edges
        for spine in ax.spines.values():
            spine.set_color("#606c38")
            
    # Subplot 1: Original Spectrogram
    im0 = axes[0].imshow(spec, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Original CQT Spectrogram", color="#fefae0", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Frequency (CQT Bins)", color="#fefae0", fontsize=10)
    axes[0].set_xlabel("Time (Frames)", color="#fefae0", fontsize=10)
    
    # Subplot 2: Heatmap Only
    im1 = axes[1].imshow(heatmap, aspect="auto", origin="lower", cmap="jet")
    axes[1].set_title("GradCAM Heatmap Only", color="#fefae0", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Time (Frames)", color="#fefae0", fontsize=10)
    
    # Subplot 3: Visual Overlay
    axes[2].imshow(spec, aspect="auto", origin="lower", cmap="gray")  # Base in grayscale
    im2 = axes[2].imshow(heatmap, aspect="auto", origin="lower", cmap="jet", alpha=0.55)  # Overlay with alpha
    axes[2].set_title("Heatmap Overlay (Interpretation)", color="#fefae0", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Time (Frames)", color="#fefae0", fontsize=10)
    
    # Add a unified colorbar on the right
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(color="#fefae0", labelcolor="#fefae0", labelsize=8)
    cbar.outline.set_color("#606c38")
    
    fig.subplots_adjust(left=0.06, right=0.92, wspace=0.15)
    
    return fig
