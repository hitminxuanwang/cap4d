# 🧢 CAP4D
Official repository for the paper

**CAP4D: Creating Animatable 4D Portrait Avatars with Morphable Multi-View Diffusion Models**, ***CVPR 2025 (Oral)***.

<a href="https://felixtaubner.github.io/" target="_blank">Felix Taubner</a><sup>1,2</sup>, <a href="https://scholar.google.com/citations?user=KFx-0xIAAAAJ&hl=en" target="_blank">Ruihang Zhang</a><sup>1</sup>, <a href="https://mathieutuli.com/" target="_blank">Mathieu Tuli</a><sup>3</sup>, <a href="https://davidlindell.com/" target="_blank">David B. Lindell</a><sup>1,2</sup>

<sup>1</sup>University of Toronto, <sup>2</sup>Vector Institute, <sup>3</sup>LG Electronics

<a href='https://arxiv.org/abs/2412.12093'><img src='https://img.shields.io/badge/arXiv-2301.02379-red'></a> <a href='https://felixtaubner.github.io/cap4d/'><img src='https://img.shields.io/badge/project page-CAP4D-Green'></a> <a href='#citation'><img src='https://img.shields.io/badge/cite-blue'></a>

![Preview](assets/banner.gif)

TL;DR: CAP4D turns any number of reference images into an animatable avatar. 

## ⚡️ Quick start guide

### 🛠️ 1. Create conda environment and install requirements

```bash
# 1. Clone repo
git clone https://github.com/felixtaubner/cap4d/
cd cap4d

# 2. Create conda environment for CAP4D:
conda create --name cap4d_env python=3.10
conda activate cap4d_env

# 3. Install requirements
pip install -r requirements.txt
```
Follow the [instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) and install Pytorch3D. Make sure to install with CUDA support. We recommend to install from source: ```pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"```

### 📦 2. Download FLAME and MMDM weights
Follow instructions on the [FLAME](https://flame.is.tue.mpg.de/index.html) website to download the FLAME blendshapes files. Locate ```flame2023_no_jaw.pkl``` and place it in ```data/assets/flame/```. 

Download the MMDM weights with this [link](https://www.dropbox.com/scl/fi/xmgozlkg67v0n2ib6oat5/cap4d_mmdm_100k.ckpt?rlkey=xuhrgyvyre7cezws11afqy1v2&st=j8gtx33j&dl=0), and place ```cap4d_mmdm_100k.ckpt``` in ```data/weights/mmdm/checkpoints/```. 

### ✅ 3. Check installation with a test run
Run the pipeline in debug settings to test the installation.

```bash
bash scripts/test_pipeline.sh
```

Check if a video is exported to `examples/debug_output/tesla/sequence_00/renders.mp4`.
If it appears to show a blurry cartoon Nicola Tesla, you're all set! 

### 🎬 4. Inference 
Run the provided scripts to generate avatars and animate them with a single script:

```bash
bash scripts/generate_felix.sh
bash scripts/generate_lincoln.sh
bash scripts/generate_tesla.sh
```

The output directories contain exported animations which you can view in real-time.
Open the [real-time viewer](https://felixtaubner.github.io/cap4d/viewer/) in your browser (powered by [Brush](https://github.com/ArthurBrussee/brush/)). Click `Load file` and
upload the exported animation found in `examples/output/{SUBJECT}/animation_{ID}/exported_animation.ply`.

## 🔧 Custom inference

### ⚙️ 1. Run FlowFace tracking
Coming soon! For now, only generations using the provided identities with precomputed [FlowFace](https://felixtaubner.github.io/flowface/) annotations are supported. 

### 🖼️ 2. Generate images using MMDM

```bash
# Generate images with single reference image
python cap4d/inference/generate_images.py --config_path configs/generation/single_ref.yaml --reference_data_path examples/input/lincoln/ --output_path examples/output/lincoln/

# Generate images with multiple reference images
python cap4d/inference/generate_images.py --config_path configs/generation/multi_ref.yaml --reference_data_path examples/input/felix/ --output_path examples/output/felix/
```
Note: the generation script will use all visible CUDA devices. The more available devices, the faster it runs! This will take hours, and requires lots of RAM (ideally > 64 GB) to run smoothly.

### 👤 3. Fit Gaussian avatar 

```bash
python gaussianavatars/train.py --config_path configs/avatar/default.yaml --source_paths examples/output/{SUBJECT}/reference_images/ examples/output/{SUBJECT}/generated_images/ --model_path examples/output/{SUBJECT}/avatar/ --interval 5000
```

### 🕺 4. Animate your avatar

For now, only animations with precomputed FLAME annotations are supported. 
These animations are located in `examples/input/animation/`.

```bash
python gaussianavatars/animate.py --model_path examples/output/lincoln/avatar/ --target_animation_path examples/input/animation/sequence_00/fit.npz  --target_cam_trajectory_path examples/input/animation/sequence_00/orbit.npz  --output_path examples/output/lincoln/animation_00/ --export_ply 1 --compress_ply 0
```

The `--target_animation_path` contains FLAME expressions and pose, while the (optional) `--target_cam_trajectory_path` contains the relative camera trajectory. 

## 📚 Related Resources

The MMDM code is based on [ControlNet](https://github.com/lllyasviel/ControlNet). The 4D Gaussian avatar code is based on [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars). Special thanks to the authors for making their code public!

Related work: 
- [CAT3D](https://cat3d.github.io/): Create Anything in 3D with Multi-View Diffusion Models
- [GaussianAvatars](https://shenhanqian.github.io/gaussian-avatars): Photorealistic Head Avatars with Rigged 3D Gaussians
- [FlowFace](https://felixtaubner.github.io/flowface/): 3D Face Tracking from 2D Video through Iterative Dense UV to Image Flow
- [StableDiffusion](https://github.com/Stability-AI/stablediffusion): High-Resolution Image Synthesis with Latent Diffusion Models

Awesome concurrent work:
- [Pippo](https://yashkant.github.io/pippo/): High-Resolution Multi-View Humans from a Single Image
- [Avat3r](https://tobias-kirschstein.github.io/avat3r/): Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars

## 📖 Citation

```tex
@inproceedings{taubner2025cap4d,
    author    = {Taubner, Felix and Zhang, Ruihang and Tuli, Mathieu and Lindell, David B.},
    title     = {{CAP4D}: Creating Animatable {4D} Portrait Avatars with Morphable Multi-View Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {5318-5330}
}
```

## Acknowledgement
This work was developed in collaboration with and with sponsorship from **LG Electronics**. We gratefully acknowledge their support and contributions throughout the course of this project.
