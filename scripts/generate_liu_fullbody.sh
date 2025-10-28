#!/bin/bash

# Run quick inference to test pipeline installation
#mkdir examples/output

# Test MMDM installation by generating a few images
# python cap4d/inference/generate_images.py \
#     --config_path configs/generation/single_ref.yaml \
#     --reference_data_path examples/input/6/ \
#     --output_path examples/output/6/

# Test GaussianAvatars installation by fitting for a few iterations
# python gaussianavatars/train_fullbody.py \
#    --config_path configs/avatar/default.yaml \
#    --source_paths examples/output/full_body/ \
#    --model_path examples/output/full_body/avatar_refine/

# # Test rendering and export 
 python gaussianavatars/animate_smpl.py \
 --model_path examples/output/full_body/avatar_refine/ \
 --output_path examples/output/full_body/static_render/ \
 --source_paths examples/output/full_body/ \
 --render_static \
 --timestep 0 \
 --camera_id 0 \
 --export_ply 1

#python gaussianavatars/show_smpl.py
