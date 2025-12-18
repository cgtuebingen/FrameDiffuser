# FrameDiffuser

<p align="left">
  <strong>
    G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering
  </strong>
</p>

https://github.com/cgtuebingen/FrameDiffuser/assets/XXXXX/XXXXX

### [Project Page](https://framediffuser.jdihlmann.com/) | [Paper](https://arxiv.org/abs/XXXX.XXXXX)

[Ole Beißwenger](https://github.com/obeisswenger), [Jan-Niklas Dihlmann](https://jdihlmann.com/), [Hendrik P.A. Lensch](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/computergrafik/lehrstuhl/mitarbeiter/prof-dr-ing-hendrik-lensch/)

University of Tübingen

---

**FrameDiffuser** is an autoregressive neural rendering framework that generates temporally consistent, photorealistic frames from G-buffer data. Our approach enables frame-by-frame generation for interactive applications where future frames depend on user input.

![Teaser](images/pipeline.png)

## Features

- **Autoregressive Generation** — Frame-by-frame rendering for interactive applications
- **Temporal Consistency** — Stable generation over thousands of frames
- **Automatic Lighting** — Synthesizes global illumination, shadows, and reflections from G-buffer
- **Environment-Specific** — Specialized models for superior quality

## Code

Coming soon.

## Citation

```bibtex
@inproceeding{framediffuser,
  author = {Beißwenger, Ole and Dihlmann, Jan-Niklas and Lensch, Hendrik P.A.},
  title = {FrameDiffuser: G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering},
  booktitle = {arXiv preprint},
  year = {2025}
}
```

## License

This code is released under the **Adobe Research License** for **noncommercial research purposes only**.

## Acknowledgments

This repository includes code from:
- **RGB↔X** (Zeng et al., SIGGRAPH 2024): `load_image.py` and `pipeline_x2rgb.py`  
  Source: https://github.com/zheng95z/rgbx  
  Copyright Adobe Inc., licensed under the Adobe Research License.
