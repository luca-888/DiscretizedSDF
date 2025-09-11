#!/bin/bash

echo "=== DiscretizedSDF 复杂场景稳定训练脚本 ==="
echo "针对室内复杂场景优化的保守参数配置"

# 设置数据路径
DATA_PATH=${1:-"/home/farsee2/data/huijin/chunks/Tile_0001"}
OUTPUT_PATH=${2:-"/home/farsee2/data/outputs/huijin/Tile_0001_stable"}
ITERATIONS=${3:-30000}

echo "数据路径: $DATA_PATH"
echo "输出路径: $OUTPUT_PATH"
echo "迭代次数: $ITERATIONS"

mkdir -p $OUTPUT_PATH
cp train_complex_scene.sh $OUTPUT_PATH/

echo "开始稳定训练..."

# 运行训练，同时开启tensorboard
tensorboard --logdir $OUTPUT_PATH --port 6006
python train.py \
    --source_path "$DATA_PATH" \
    --model_path "$OUTPUT_PATH" \
    --iterations $ITERATIONS \
    --render_mode defer+split_sum \
    --gaussian_type 2d \
    --use_sdf \
    --test_iterations 1000 3000 5000 8000 10000 15000 20000 25000 30000 \
    --save_iterations 1000 3000 5000 8000 10000 15000 20000 25000 30000 \
    --resolution 800 \
    \
    `# === 关键稳定性参数 ===` \
    --lambda_dev 1.0 \
    --lambda_distortion 5.0 \
    --lambda_predicted_normal 1e-1 \
    --lambda_base_smoothness 5e-4 \
    --lambda_brdf_smoothness 1e-4 \
    --lambda_opacity_reg 1e-5 \
    \
    `# === 保守学习率 ===` \
    --position_lr_init 8e-5 \
    --position_lr_final 8e-7 \
    --envmap_lr_init 8e-3 \
    --envmap_lr_final 8e-4 \
    --roughness_lr 5e-5 \
    --metallic_lr 5e-5 \
    --albedo_lr 1e-4 \
    --sdf_lr 0.01 \
    --deviation_lr 0.001 \
    --opacity_lr 0.01 \
    --scaling_lr 0.005 \
    --rotation_lr 0.001 \
    \
    `# === 保守密度化参数 ===` \
    --densify_grad_threshold 0.001 \
    --opacity_cull 0.01 \
    --densification_interval 200 \
    --opacity_reset_interval 5000 \
    --densify_until_iter 10000 \
    --densify_from_iter 1000 \
    \
    `# === 正则化控制 ===` \
    --normal_reg_from_iter 2000 \
    --dist_from_iteration 2000 \
    --proj_from_iteration 3000

echo "训练完成! 结果保存在: $OUTPUT_PATH"

# 自动运行tensorboard (可选)
# echo "启动TensorBoard监控..."
# tensorboard --logdir $OUTPUT_PATH --port 6006 &

echo ""
echo "=== 使用说明 ==="
echo "1. 如果仍然不稳定，可以进一步降低 lambda_dev 到 0.5"
echo "2. 监控前3000步的高斯球数量变化"
echo "3. 如果密度化过快，增加 densification_interval 到 300"
echo "4. 使用 tensorboard --logdir $OUTPUT_PATH 监控训练过程"