


# python gaussianavatars/train_smplx.py \
#   --config_path configs/avatar/default.yaml \
#   --source_paths examples/output/smplx_demo1/ \
#   --model_path examples/output/smplx_demo1/avatar/ \


#  python gaussianavatars/animate_smplx.py \
#  --model_path examples/output/smplx_demo1/avatar/ \
#  --output_path examples/output/smplx_demo1/static_render/ \
#  --source_paths examples/output/smplx_demo1/ \
#  --render_static \
#  --timestep 0 \
#  --camera_id 0 \
#  --export_ply 1




# python gaussianavatars/train_smplx.py \
#   --config_path configs/avatar/default.yaml \
#   --source_paths examples/output/smplx_demo2/ \
#   --model_path examples/output/smplx_demo2/avatar/ \



# python gaussianavatars/animate_smplx.py \
#  --model_path examples/output/smplx_demo2/avatar/ \
#  --output_path examples/output/smplx_demo2/static_render/ \
#  --source_paths examples/output/smplx_demo2/ \
#  --render_static \
#  --timestep 0 \
#  --camera_id 0 \
#  --export_ply 1


 python gaussianavatars/animate_smplx.py \
--model_path examples/output/smplx_demo1/avatar/ \
--output_path examples/output/smplx_demo1/animated_render/ \
--source_paths examples/output/smplx_demo1/ \
--target_animation_path combined_animation.npz \
#--target_cam_trajectory_path examples/input/animation/sequence_01/orbit.npz \
--export_ply 1