---
title: Edge-Preserving Multi-Scale Tone Manipulation
emoji: 🖼️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
license: mit
---

# Edge-Preserving Multi-Scale Tone Manipulation

Implementation of [Farbman et al. 2008](https://doi.org/10.1145/1360612.1360666) — "Edge-Preserving Decompositions for Multi-Scale Tone and Detail Manipulation."

Upload an image and manipulate tone at different spatial scales using edge-preserving decompositions.

## Features

- **3 smoothing methods:** WLS (paper's method), Bilateral Filter, Guided Filter
- **2 pyramid modes:** Non-iterative (each level from original) and Iterative (each level from previous)
- **Per-scale detail boosting:** Independent control over fine, medium, and coarse detail layers
- **Exposure and base contrast** controls
- **Auto-adapts** filter parameters to image size
