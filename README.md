# Simple Gaussian Splatting

## What is this? 

For learning purposes, we offer a set of tools and documentation to study 3D Gaussian Splatting. We also plan to provide an unofficial implementation of the paper [3D Gaussian Splatting
for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).


## Overview 

* Detailed documentation to demonstrate the mathematical principles of 3D Gaussian Splatting
    - [Documentation for Forward (rendering)](docs/forward.pdf)
    - [Documentation for Backward (training)](docs/backward.pdf)

- Based on our documentation, re-implement 3D Gaussian Splatting.
    - Forward on CPU
    - Forward on GPU
    - Backward on CPU
    - Backward on GPU
    - Full training implemention (80%)

- Provide tools for learning 3D Gaussian Splatting.
    -  A efficient viewer based on pyqtgraph for showing 3D Gaussian data (trained model).
    -  A demo showing how spherical harmonics work.
    -  Verify all computation processes of backward using numerical differentiation.

## Requirements 

```bash
pip3 install -r requirements.txt
pip3 install gsplatcu/.
```

## Rendering

Given camera information, render 3D Gaussian data onto the 2D image.

The TRAINED_FILE is the .ply file generated by the official Gaussian Splatting, or the .npy file generated by our train.py.

**CPU version**
```bash
python3 forward_cpu.py --gs='THE_PATH_OF_YOUR_TRAINED_FILE'
```

<img src="https://camo.qiitausercontent.com/b379b038898126c199436e94f7b76635f59037ff/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e61702d6e6f727468656173742d312e616d617a6f6e6177732e636f6d2f302f3134393136382f31396664316562342d333761612d316436652d346338302d3835323935346233613135382e676966" width="500px">


**GPU version.**
```bash
python3 forward_gpu.py --gs='THE_PATH_OF_YOUR_TRAINED_FILE'
```
![forward demo](imgs/forward.png)

## Training

Download the T&T+DB COLMAP datasets.
[T&T+DB COLMAP (650MB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) 

```bash
python3 train.py --path='THE_PATH_OF_UNZIPPED_DATASET'
```

## 3D Gaussian Viewer 

A efficient 3D Gaussian splatting viewer for showing 3D Gaussian data. 

```bash
python3 gaussian_viewer.py --gs='THE_PATH_OF_YOUR_TRAINED_PLY_OR_NPY_FILE'
```

<img src="imgs/viewer.gif" width="640px">



## Spherical harmonics demo

A demo showing how spherical harmonics work.

```bash
python3 sh_demo.py
```

![sh demo](imgs/sh_demo.gif)
<sup><sub>"The ground truth Earth image is modified from [URL](https://commons.wikimedia.org/wiki/File:Solarsystemscope_texture_8k_earth_daymap.jpg). By Solar System Scope. Licensed under CC-BY-4.0"</sub></sup>

## Verify the backward process

Verify all computation processes of backward using numerical differentiation.

```bash
python3 backward_cpu.py
python3 backward_gpu.py
```