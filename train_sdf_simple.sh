#!/bin/bash

echo "=== DiscretizedSDF 纯几何优化训练脚本 ==="
echo "保留SDF几何优化，去掉PBR复杂度，输出SuperSplat兼容格式"

# 设置数据路径
DATA_PATH=${1:-"/home/farsee2/data/huijin/chunks/Tile_0001"}
OUTPUT_PATH=${2:-"/home/farsee2/data/outputs/huijin/Tile_0001_sdf_simple"}
ITERATIONS=${3:-30000}

echo "数据路径: $DATA_PATH"
echo "输出路径: $OUTPUT_PATH"
echo "迭代次数: $ITERATIONS"

mkdir -p $OUTPUT_PATH
cp train_sdf_simple.sh $OUTPUT_PATH/

echo "开始SDF几何优化训练..."

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
    \
    `# === 保留SDF几何优化 ===` \
    --lambda_dev 0.5 \
    --lambda_distortion 2.0 \
    --lambda_predicted_normal 1e-2 \
    --lambda_base_smoothness 5e-4 \
    \
    `# === 完全禁用PBR损失 ===` \
    --lambda_brdf_smoothness 0.0 \
    --lambda_light_reg 0.0 \
    --lambda_opacity_reg 0.0 \
    \
    `# === 稳定学习率 ===` \
    --position_lr_init 1e-4 \
    --position_lr_final 1e-6 \
    --sdf_lr 0.01 \
    --deviation_lr 0.001 \
    --opacity_lr 0.02 \
    --scaling_lr 0.005 \
    --rotation_lr 0.001 \
    \
    `# === PBR参数学习率设为0 ===` \
    --albedo_lr 0.0 \
    --metallic_lr 0.0 \
    --roughness_lr 0.0 \
    --envmap_lr_init 0.0 \
    --envmap_lr_final 0.0 \
    \
    `# === 保守densification ===` \
    --densify_grad_threshold 0.0008 \
    --opacity_cull 0.015 \
    --densification_interval 250 \
    --opacity_reset_interval 6000 \
    --densify_until_iter 12000 \
    --densify_from_iter 1000 \
    \
    `# === SDF正则化时机 ===` \
    --normal_reg_from_iter 1500 \
    --dist_from_iteration 1500 \
    --proj_from_iteration 2500

echo "训练完成! 结果保存在: $OUTPUT_PATH"
echo ""
echo "=== 下一步：转换为SuperSplat格式 ==="
echo "运行转换脚本: python convert_to_supersplat.py -m $OUTPUT_PATH/point_cloud/iteration_30000/"