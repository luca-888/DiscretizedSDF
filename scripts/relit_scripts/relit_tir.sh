#!/bin/zsh

light_dir="data/tensoir_synthetic/env/high_res_envmaps_2k/"
output_dir=$1
scene_list="armadillo lego ficus hotdog"
light_list="bridge city fireplace forest night"

for i in $scene_list; do
    for j in $light_list; do
        python relight.py \
        -m $output_dir/${i} \
        --render_mode defer+split_sum \
        --skip_train \
        --gaussian_type 2d \
        --use_sdf \
        --transform tir \
        --envmap ${light_dir}/${j}.hdr
    done
done