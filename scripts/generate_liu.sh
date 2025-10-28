#!/bin/bash

# Run quick inference to test pipeline installation
#mkdir examples/output

# Test MMDM installation by generating a few images
# python cap4d/inference/generate_images.py \
#     --config_path configs/generation/single_ref.yaml \
#     --reference_data_path examples/input/6/ \
#     --output_path examples/output/6/

# Test GaussianAvatars installation by fitting for a few iterations
# python gaussianavatars/train.py \
#     --config_path configs/avatar/default.yaml \
#     --source_paths examples/output/liu/reference_images/ examples/output/liu/generated_images/ \
#     --model_path examples/output/6/avatar_debug/


# python gaussianavatars/train.py \
#     --config_path configs/avatar/default.yaml \
#     --source_paths examples/output/liu/sparse/ \
#     --model_path examples/output/liu/sparse/avatar/

# # Test rendering and export 
# python gaussianavatars/animate.py \
#     --model_path examples/output/6/avatar/ \
#     --target_animation_path examples/input/animation/sequence_01/fit.npz \
#     --target_cam_trajectory_path examples/input/animation/sequence_01/orbit.npz  \
#     --output_path examples/output/6/animation_01/ \
#     --export_ply 1 \
#     --compress_ply 0


# python gaussianavatars/animate.py \
# --model_path examples/output/6/avatar_debug/ \
# --output_path examples/output/6/static_render/ \
# --source_paths examples/output/liu/reference_images/ examples/output/liu/generated_images/ \
# --render_static \
# --timestep 0 \
# --camera_id 0 \
# --export_ply 1
