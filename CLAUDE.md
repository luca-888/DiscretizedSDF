# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览
DiscretizedSDF 是一个基于高斯散射(Gaussian Splatting)和离散化SDF的可重光照3D资产生成系统。该项目结合了几何、材质和光照的分解，用于多视图观察的物体重光照。

## 环境设置

### 依赖安装
```bash
# 创建conda环境
conda create -n dsdf python=3.10
conda activate dsdf

# 安装PyTorch (推荐CUDA 11.8)
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118

# 安装基础依赖
pip install -r requirements.txt

# 安装nvdiffrast
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast

# 安装自定义子模块
pip install ./submodules/fused-ssim
pip install ./submodules/diff-surfel-sdf-rasterization
pip install ./submodules/simple-knn
```

## 常用命令

### 训练模型
```bash
# Glossy Synthetic数据集训练
sh scripts/train_scripts/train_glossy.sh

# Shiny Blender数据集训练
sh scripts/train_scripts/train_shiny.sh

# TensoIR Synthetic数据集训练
sh scripts/train_scripts/train_tir.sh

# 手动训练单个场景
python train.py \
  --render_mode defer+split_sum \
  -s data/GlossySynthetic_blender/[SCENE_NAME] \
  -m outputs/[OUTPUT_DIR]/ \
  --gaussian_type 2d \
  --use_sdf \
  --sphere_init
```

### 渲染和重光照
```bash
# 渲染训练好的模型
python render.py -m [MODEL_PATH] --iteration [ITERATION]

# 重光照渲染
python relight.py -m [MODEL_PATH] --envmap [ENV_MAP_PATH]

# 快速演示 (3分钟生成重光照视频)
sh scripts/demo.sh
```

### 评估
```bash
# Glossy Synthetic数据集重光照质量评估
sh scripts/relit_scripts/eval_relit_glossy.sh [OUTPUT_DIR]
sh scripts/relit_scripts/avg_relit_glossy.sh [OUTPUT_DIR]

# 网格质量评估 (Shiny Blender)
python eval/eval_mesh.py

# 法线质量评估 (MAE)
python eval/metrics_mae.py -i [IMG_PATH] -g [GT_PATH]

# TensoIR数据集重光照评估
sh scripts/relit_scripts/eval_relit_tir.sh [OUTPUT_DIR]
```

### 数据预处理
```bash
# Glossy Synthetic数据集格式转换
python nero2blender.py --path [PATH_TO_DATASET]
python nero2blender_relight.py --path [PATH_TO_DATASET]
```

## 代码架构

### 核心模块
- **train.py / train_minimal.py**: 主训练脚本，包含完整训练循环
- **render.py**: 渲染脚本，支持多种渲染模式
- **relight.py**: 重光照脚本，支持环境光照变换

### 场景管理 (scene/)
- **Scene**: 主场景类，管理相机和数据加载
- **GaussianModel**: 高斯模型核心，处理几何和材质表示
- **dataset_readers.py**: 数据集读取器，支持Blender和COLMAP格式

### 渲染器 (gaussian_renderer/)
- **RENDER_DICT**: 渲染函数字典，支持不同高斯类型
- **render_lighting**: 光照渲染功能

### 工具模块 (utils/)
- **image_utils.py**: 图像处理工具
- **loss_utils.py**: 损失函数实现
- **camera_utils.py**: 相机相关工具
- **mesh_utils.py**: 网格处理工具

### 参数配置 (arguments/)
- **ModelParams**: 模型参数配置
- **PipelineParams**: 渲染管道参数
- **OptimizationParams**: 优化参数，包含学习率和损失权重

## 关键参数说明

### 训练参数
- `--render_mode`: 渲染模式 (defer, defer+split_sum等)
- `--gaussian_type`: 高斯类型 (2d, 3d)
- `--use_sdf`: 启用SDF离散化
- `--sphere_init`: 使用球面初始化
- `--env_mode`: 环境光照模式 (envmap)
- `--env_res`: 环境贴图分辨率

### 损失权重
- `--lambda_predicted_normal`: 法线预测损失权重
- `--lambda_zero_one`: 零一损失权重
- `--lambda_base_smoothness`: 基础平滑损失权重
- `--lambda_light_reg`: 光照正则化权重
- `--lambda_distortion`: 扭曲损失权重
- `--lambda_proj`: 投影损失权重

## 数据集结构
预期的数据集目录结构：
```
./data/
├── GlossySynthetic_blender/
├── shiny_blender/
├── tensoir_synthetic/
└── glossy_relight/
```

## 输出目录
- `outputs/[dataset]/[scene]/`: 训练输出
- `point_cloud/iteration_[N]/`: 保存的点云文件
- `envmap/iteration_[N]/`: 环境贴图文件
- `relight/`: 重光照结果
- 该工程在改动时，必须保留sdf相关的属性、转换、损失，sdf是我期望保留的主要特性，也就是希望重建更好的几何