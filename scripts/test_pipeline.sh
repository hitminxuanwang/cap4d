#!/bin/bash

# Run quick inference to test pipeline installation
mkdir examples/debug_output

# Test MMDM installation by generating a few images
python cap4d/inference/generate_images.py \
    --config_path configs/generation/debug.yaml \
    --reference_data_path examples/input/tesla/ \
    --output_path examples/debug_output/tesla/

# Test GaussianAvatars installation by fitting for a few iterations
python gaussianavatars/train.py \
    --config_path configs/avatar/debug.yaml \
    --source_paths examples/debug_output/tesla/reference_images/ examples/debug_output/tesla/generated_images/ \
    --model_path examples/debug_output/tesla/avatar/

# Test rendering and export 
python gaussianavatars/animate.py \
    --model_path examples/debug_output/tesla/avatar/ \
    --target_animation_path examples/input/animation/sequence_00/fit.npz \
    --target_cam_trajectory_path examples/input/animation/sequence_00/orbit.npz  \
    --output_path examples/debug_output/tesla/animation_00/ \
    --export_ply 1 \
    --compress_ply 0
