# MintFlow: Minimal Trajectory Intervention for Constrained Flow Matching

**MintFlow** is a training-free framework for constrained generation with pretrained flow matching models. Instead of modifying the pretrained flow or projecting the final sample, MintFlow applies a **minimal intervention to an intermediate state** and then follows the original flow dynamics to the terminal time.

This allows MintFlow to satisfy diverse constraints while better preserving the distribution learned by the pretrained model.


## Applications

We provide implementations for three application settings:

- `image_inverse_problems/` — Image inverse problems on AFHQ-Cat and FFHQ
- `image_editing/` — Text-guided image editing on PIE-Bench
- `physics/` — Physics-informed generation with PDE constraints

## Getting Started

Please refer to the README of each application:

- [`image_inverse_problems/`](./image_inverse_problems/)
- [`image_editing/`](./image_editing/)
- [`physics/`](./physics/)

