#!/bin/bash

# Run quick inference to test pipeline installation
#mkdir examples/output

# Test MMDM installation by generating a few images
# python cap4d/inference/generate_images.py \
#     --config_path configs/generation/single_ref.yaml \
#     --reference_data_path examples/input/6/ \
#     --output_path examples/output/6/

# Test GaussianAvatars installation by fitting for a few iterations
python gaussianavatars/train_fullbody.py \
   --config_path configs/avatar/default.yaml \
   --source_paths examples/output/yidong/ \
   --model_path examples/output/yidong/avatar_refine/ \
   #--test_iterations  1 2 5 10

# # Test rendering and export 
#  python gaussianavatars/animate_smpl.py \
#  --model_path examples/output/yidong/avatar_refine/ \
#  --output_path examples/output/yidong/static_render/ \
#  --source_paths examples/output/yidong/ \
#  --render_static \
#  --timestep 0 \
#  --camera_id 0 \
#  --export_ply 1

#python gaussianavatars/show_smpl.py

# python gaussianavatars/animate_smpl.py \
# --model_path examples/output/yidong/avatar_refine/ \
# --output_path examples/output/yidong/animated_render/ \
# --source_paths examples/output/yidong/ \
# --target_animation_path combined_animation.npz \
# #--target_cam_trajectory_path examples/input/animation/sequence_01/orbit.npz \
# --export_ply 1
