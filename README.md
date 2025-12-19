# FrameDiffuser

<p align="left">
  <strong>
    G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering
  </strong>
</p>

https://github.com/user-attachments/assets/b43532e0-5f5d-40f3-9d3c-e49814faa77f

<p align="center">
    <span> 🌐  <a href="https://framediffuser.jdihlmann.com/"> Project Page </a> </span>&nbsp;&nbsp;&nbsp;
    <span> 📄  <a href="https://arxiv.org/abs/2512.16670"> Paper (Arxiv) </a> </span>&nbsp;&nbsp;&nbsp;
  <span>  📦  <a href="https://drive.google.com/file/d/1yZuiW7DOZzHFDuZnKV_SU_Ofoq2wuoBG/view?usp=sharing"> Materials </a> </span>&nbsp;&nbsp;&nbsp;
  <span>  ✍🏻
     <a href="https://github.com/cgtuebingen/FrameDiffuser?tab=readme-ov-file#citation"> Citation </a> </span>&nbsp;&nbsp;&nbsp;
</p>

---

**FrameDiffuser** is an autoregressive neural rendering framework that generates temporally consistent, photorealistic frames from G-buffer data. Our approach enables frame-by-frame generation for interactive applications where future frames depend on user input.


## Features

- **Autoregressive Generation** — Frame-by-frame rendering for interactive applications
- **Temporal Consistency** — Stable generation over thousands of frames
- **Automatic Lighting** — Synthesizes global illumination, shadows, and reflections from G-buffer
- **Environment-Specific** — Specialized models for superior quality


## Code
Paper is currently under review, we will realse the code shortly (~ end of january 2026) in a cleaned version. Up until then if you have any questions regarding the project or need material to compare against fast, feel free to contact us.

## Citation

```bibtex
@article{beisswenger2024framediffuser,
  title={FrameDiffuser: G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering},
  author={Beisswenger, Ole and Dihlmann, Jan-Niklas and Lensch, Hendrik},
  journal={...},
  year={2024}
}
```

## Acknowledgments

This repository includes code from:
- **RGB↔X** (Zeng et al., SIGGRAPH 2024): `load_image.py` and `pipeline_x2rgb.py`  
  Source: https://github.com/zheng95z/rgbx  
  Copyright Adobe Inc., licensed under the Adobe Research License.
