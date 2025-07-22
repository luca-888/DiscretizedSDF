#!/bin/bash

root_dir="data/GlossySynthetic_blender/"
list="tbell cat potion teapot angel bell horse luyu"

for i in $list; do

python train.py \
--render_mode defer+split_sum \
-s ${root_dir}${i} \
-m outputs/glossy/${i}/ \
-w --sh_degree -1 \
--lambda_predicted_normal 0.2 \
--lambda_zero_one 0.4 \
--env_res 512 \
--env_mode envmap \
--port 12991 \
--lambda_base_smoothness 0.03 \
--lambda_light_reg 0.005 \
--lambda_distortion 2000 \
--gaussian_type 2d \
--use_sdf \
--lambda_proj 10. \
--lambda_dev 1.0 \
--sphere_init
done
