


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
#   --source_paths examples/output/smplx_demo1/ \
#   --model_path examples/output/smplx_demo1/avatar_refine/ \



# python gaussianavatars/animate_smplx.py \
#  --model_path examples/output/smplx_demo1/avatar_refine/ \
#  --output_path examples/output/smplx_demo1/static_render/ \
#  --source_paths examples/output/smplx_demo1/ \
#  --render_static \
#  --timestep 0 \
#  --camera_id 0 \
#  --export_ply 1


python gaussianavatars/animate_smplx.py \
--model_path examples/output/smplx_demo1/avatar_refine/ \
--output_path examples/output/smplx_demo1/animated_render/ \
--source_paths examples/output/smplx_demo1/ \
--target_animation_path combined_animation_smplx.npz \
--export_ply 1
