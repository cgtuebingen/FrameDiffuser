# FrameDiffuser

<p align="center">
  <strong>G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering</strong>
</p>

https://github.com/user-attachments/assets/b43532e0-5f5d-40f3-9d3c-e49814faa77f

<p align="center">
  <a href="https://framediffuser.jdihlmann.com/">🌐 Project Page</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2512.16670">📄 Paper</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/Obeisswenger/FrameDiffuser-Models">🤗 Models</a>
</p>

---

## Installation

```bash
# Create conda environment
conda create -n framediffuser python=3.10 -y
conda activate framediffuser

# Install PyTorch with CUDA (required)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## Overview

FrameDiffuser is an autoregressive neural rendering framework that generates temporally consistent, photorealistic frames by conditioning on G-buffer data and the model's own previous output. The dual-conditioning architecture combines ControlNet for structural guidance with ControlLoRA for temporal coherence.

## Architecture

- **ControlNet**: Processes 10-channel G-buffer input (BaseColor, Normals, Depth, Roughness, Metallic, Irradiance)
- **ControlLoRA**: Conditions on previous frame encoded in VAE latent space for temporal coherence
- **Base Model**: Stable Diffusion 1.5

## Pretrained Models

Pretrained weights are available on [HuggingFace](https://huggingface.co/Obeisswenger/FrameDiffuser-Models).

| Model | Scene Type | Notes |
|-------|------------|-------|
| `DowntownWest` | Outdoor | **Recommended** for outdoor scenes |
| `Hillside` | Indoor | **Recommended** for indoor scenes |
| `CityPark` | Outdoor | |
| `CitySample` | Outdoor | |
| `ElectricDreams` | Outdoor | Rainforest environment |
| `DerelictCorridor` | Indoor | Small environment with dark lighting |

Each model directory contains:
- `controlnet/` - ControlNet weights (G-buffer encoder)
- `controllora.safetensors` - ControlLoRA weights (temporal conditioning)

For best results:
- **Outdoor scenes**: Use `DowntownWest`
- **Indoor scenes**: Use `Hillside`

## Usage

### Training

1. Place your data in `data/train/` and `data/validation/`
2. Edit `train_3_stages.bat` to set your prompt and paths
3. Run:

```bash
train_3_stages.bat
```

The provided batch file serves as an example configuration. For best performance, experiment with adjusted settings for your specific environment and dataset.

### Inference

```bash
python inference.py
```

To add new models or datasets, use the GUI to select paths and save configurations.

## Rendering G-Buffers in Unreal Engine

G-buffer data can be exported from Unreal Engine using the **Movie Render Queue** with custom Post Process Materials.

### Setup

1. Enable the **Movie Render Queue** plugin: `Edit > Plugins > Movie Render Queue` (restart required)

2. Create Post Process Materials for each G-buffer channel (BaseColor, Normals, Depth, Roughness, Metallic) that output the corresponding Scene Texture to Emissive Color

3. In the **Movie Render Queue**, add your Level Sequence and open **Settings**

4. Under **Rendering > Deferred Rendering**, expand **Deferred Renderer Data**

5. In **Additional Post Process Materials**, add array elements for each G-buffer material:
   - Enable the element
   - Set **Name** to the buffer type (e.g., "BaseColor", "Depth")
   - Assign the corresponding Post Process Material

6. Add a **.png Sequence** output format under **Exports**

For more details, see the [Cinematic Render Passes](https://dev.epicgames.com/documentation/en-us/unreal-engine/cinematic-render-passes-in-unreal-engine) documentation.

## Dataset Structure

Place your G-buffer renders in the following structure:

```
data/
├── train/
│   ├── FinalImage/
│   │   ├── FinalImage_0000.png
│   │   ├── FinalImage_0001.png
│   │   └── ...
│   ├── BaseColor/
│   │   ├── BaseColor_0000.png
│   │   └── ...
│   ├── Normals/
│   ├── Depth/
│   ├── Roughness/
│   └── Metallic/          (optional)
└── validation/
    ├── FinalImage/
    ├── BaseColor/
    ├── Normals/
    ├── Depth/
    ├── Roughness/
    └── Metallic/          (optional)
```

**Requirements:**
- All buffers must have matching frame numbers
- Validation needs at least 2 frames (for previous frame conditioning)
- Supported formats: PNG, JPG
- Required: FinalImage, BaseColor, Normals, Depth, Roughness
- Optional: Metallic (creates black channel if missing)

## Citation

If you find this work useful, please cite:

```bibtex
@article{beisswenger2025framediffuser,
  title={FrameDiffuser: G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering},
  author={Beisswenger, Ole and Dihlmann, Jan-Niklas and Lensch, Hendrik},
  journal={arXiv preprint arXiv:2512.16670},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation).

This project builds upon:
- [control-lora-v3](https://github.com/HighCWu/control-lora-v3) by Wu Hecong (MIT License)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) by CompVis
- [diffusers](https://github.com/huggingface/diffusers) by Hugging Face
