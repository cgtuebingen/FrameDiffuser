# FrameDiffuser

<p align="left">
  <strong>
    G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering
  </strong>
</p>

https://github.com/user-attachments/assets/b43532e0-5f5d-40f3-9d3c-e49814faa77f

### [Project Page](https://framediffuser.jdihlmann.com/) | [Paper](https://arxiv.org/abs/XXXX.XXXXX)

[Ole Beißwenger](https://github.com/obeisswenger), [Jan-Niklas Dihlmann](https://jdihlmann.com/), [Hendrik P.A. Lensch](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/computergrafik/lehrstuhl/mitarbeiter/prof-dr-ing-hendrik-lensch/)

University of Tübingen

---

**FrameDiffuser** is an autoregressive neural rendering framework that generates temporally consistent, photorealistic frames from G-buffer data. Our approach enables frame-by-frame generation for interactive applications where future frames depend on user input.


## Features

- **Autoregressive Generation** — Frame-by-frame rendering for interactive applications
- **Temporal Consistency** — Stable generation over thousands of frames
- **Automatic Lighting** — Synthesizes global illumination, shadows, and reflections from G-buffer
- **Environment-Specific** — Specialized models for superior quality

## Code

Coming soon.

Additional materials and supplemental videos can be found [here](https://drive.google.com/file/d/1yZuiW7DOZzHFDuZnKV_SU_Ofoq2wuoBG/view?usp=drive_link).

## Citation

```bibtex
@article{beisswenger2024framediffuser,
  title={FrameDiffuser: G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering},
  author={Beisswenger, Ole and Dihlmann, Jan-Niklas and Lensch, Hendrik},
  journal={...},
  year={2024}
}
```

## License

This code is released under the **Adobe Research License** for **noncommercial research purposes only**.

## Acknowledgments

This repository includes code from:
- **RGB↔X** (Zeng et al., SIGGRAPH 2024): `load_image.py` and `pipeline_x2rgb.py`  
  Source: https://github.com/zheng95z/rgbx  
  Copyright Adobe Inc., licensed under the Adobe Research License.
