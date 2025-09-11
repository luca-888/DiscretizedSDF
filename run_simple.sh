#!/bin/bash

echo "=== DiscretizedSDF 简化训练脚本 ==="
echo "PBR和重光照功能已禁用，使用基础高斯散射训练"

# 设置数据路径
DATA_PATH=${1:-"/home/farsee2/data/huijin/chunks/Tile_0001"}
OUTPUT_PATH=${2:-"/home/farsee2/data/outputs/huijin/Tile_0001_v1"}
ITERATIONS=${3:-30000}

echo "数据路径: $DATA_PATH"
echo "输出路径: $OUTPUT_PATH"
echo "迭代次数: $ITERATIONS"

mkdir -p $OUTPUT_PATH
cp run_simple.sh $OUTPUT_PATH/

python train.py \
    --source_path "$DATA_PATH" \
    --model_path "$OUTPUT_PATH" \
    --iterations $ITERATIONS \
    --render_mode defer+split_sum \
    --gaussian_type 2d \
    --use_sdf \
    --test_iterations 1000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 1000 5000 10000 15000 20000 25000 30000 \
    --resolution 800 \
    --lambda_distortion 10.0      \
    --lambda_dev 10.0             \
    --lambda_predicted_normal 5e-2  \
    --lambda_base_smoothness 1e-3   \
    --lambda_brdf_smoothness 1e-4   \
    --position_lr_init 8e-5        \
    --envmap_lr_init 8e-3          \
    --roughness_lr 1e-4    

    # --sphere_init \


# tensorboard --logdir /home/farsee2/data/outputs/huijin/Tile_0001_v1