#!/bin/zsh

light_dir="data/glossy_relight/hdr2k"
output_dir=$1
scene_list="bell angel cat horse luyu potion tbell teapot"
light_abbr="corridor golf neon"

for i in $scene_list; do
    for j in $light_abbr; do
        python relight.py --eval \
        -m ${output_dir}/${i} \
        -s data/glossy_relight/relight_gt_blender/${i}_${j} \
        --render_mode defer+split_sum \
        --skip_train \
        --gaussian_type 2d \
        --use_sdf \
        --envmap ${light_dir}/${j}.exr
    done
done