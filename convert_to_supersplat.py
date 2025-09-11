#!/usr/bin/env python3
"""
转换DiscretizedSDF输出为SuperSplat兼容格式
将2D高斯转换为3D高斯，去掉SDF和PBR参数
"""

import os
import numpy as np
import argparse
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB

def convert_to_supersplat(input_path, output_path=None, thin_scale=0.001):
    """
    转换2D高斯为SuperSplat兼容的3D高斯格式
    
    Args:
        input_path: 输入的point_cloud.ply文件路径
        output_path: 输出路径，默认为输入路径_supersplat.ply
        thin_scale: 2D高斯第三维的scale值
    """
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_supersplat.ply"
    
    print(f"读取: {input_path}")
    print(f"输出: {output_path}")
    
    # 读取原始ply文件
    plydata = PlyData.read(input_path)
    vertices = plydata['vertex']
    
    # 提取基本参数
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    # 提取SH特征并转换为RGB (只取DC分量)
    f_dc_0 = np.array(vertices['f_dc_0'])[:, np.newaxis]
    f_dc_1 = np.array(vertices['f_dc_1'])[:, np.newaxis] 
    f_dc_2 = np.array(vertices['f_dc_2'])[:, np.newaxis]
    shs = np.concatenate([f_dc_0, f_dc_1, f_dc_2], axis=1)
    colors = SH2RGB(shs)
    colors = np.clip(colors, 0, 1)
    
    # 提取透明度
    opacities = np.array(vertices['opacity'])
    
    # 提取旋转 (四元数)
    rot_0 = np.array(vertices['rot_0'])
    rot_1 = np.array(vertices['rot_1'])
    rot_2 = np.array(vertices['rot_2'])
    rot_3 = np.array(vertices['rot_3'])
    rotations = np.vstack([rot_0, rot_1, rot_2, rot_3]).T
    
    # 关键：2D scale转3D scale
    scale_0 = np.array(vertices['scale_0'])  # sx
    scale_1 = np.array(vertices['scale_1'])  # sy
    scale_2 = np.full_like(scale_0, thin_scale)  # sz (固定小值)
    scales_3d = np.vstack([scale_0, scale_1, scale_2]).T
    
    print(f"高斯球数量: {len(positions)}")
    print(f"2D scale范围: [{scale_0.min():.4f}, {scale_0.max():.4f}] x [{scale_1.min():.4f}, {scale_1.max():.4f}]")
    print(f"添加的第三维scale: {thin_scale}")
    
    # 构建SuperSplat兼容的属性列表
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),           # position
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),        # normal (可选，设为0)
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),  # SH DC
        ('opacity', 'f4'),                                # opacity
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),  # scale (3D)
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),  # rotation
    ]
    
    # 构建数据数组
    num_points = len(positions)
    elements = np.empty(num_points, dtype=dtype_full)
    
    # 填充数据
    elements['x'] = positions[:, 0]
    elements['y'] = positions[:, 1] 
    elements['z'] = positions[:, 2]
    elements['nx'] = 0.0  # 法线设为0
    elements['ny'] = 0.0
    elements['nz'] = 0.0
    elements['f_dc_0'] = shs[:, 0]
    elements['f_dc_1'] = shs[:, 1]
    elements['f_dc_2'] = shs[:, 2]
    elements['opacity'] = opacities
    elements['scale_0'] = scales_3d[:, 0]
    elements['scale_1'] = scales_3d[:, 1]
    elements['scale_2'] = scales_3d[:, 2]  # 新添加的第三维
    elements['rot_0'] = rotations[:, 0]
    elements['rot_1'] = rotations[:, 1]
    elements['rot_2'] = rotations[:, 2]
    elements['rot_3'] = rotations[:, 3]
    
    # 保存SuperSplat兼容的ply文件
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)
    
    print(f"转换完成! SuperSplat兼容文件保存在: {output_path}")
    print(f"现在可以用SuperSplat打开此文件进行实时浏览")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换DiscretizedSDF输出为SuperSplat格式")
    parser.add_argument("-m", "--model_path", required=True, help="模型输出目录路径")
    parser.add_argument("-i", "--iteration", default="30000", help="迭代次数 (默认: 30000)")
    parser.add_argument("-o", "--output", default=None, help="输出文件路径")
    parser.add_argument("--thin_scale", type=float, default=0.001, help="2D高斯第三维scale值 (默认: 0.001)")
    
    args = parser.parse_args()
    
    # 构建输入文件路径
    input_file = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        print(f"请确认模型路径和迭代次数是否正确")
        exit(1)
    
    convert_to_supersplat(input_file, args.output, args.thin_scale)