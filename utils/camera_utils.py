#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch 
from scene.cameras import Camera
WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    if cam_info.normal_image is not None:
        resized_normal_image_rgb = PILtoTorch(cam_info.normal_image, resolution)
        gt_normal_image = resized_normal_image_rgb[:3, ...]
    else:
        gt_normal_image = None
    if cam_info.albedo_image is not None:
        resized_albedo_image_rgb = PILtoTorch(cam_info.albedo_image, resolution)
        gt_albedo_image = resized_albedo_image_rgb[:3, ...]
    else:
        gt_albedo_image = None
    loaded_mask = PILtoTorch(cam_info.alpha_mask, resolution)


    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, 
                  data_device=args.data_device, 
                  normal_image=gt_normal_image,
                  albedo_image=gt_albedo_image)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
def rot2quaternion(rotation_matrix):
    r3 = Rotation.from_matrix(rotation_matrix)
    qua = r3.as_quat()
    return qua 


def quaternion2rot(quaternion):
    r = Rotation.from_quat(quaternion)
    rot = r.as_matrix()
    return rot

def interpolate_camera(cam_list, num_inter=4, use_dummy_image=True):
    R_list = Rotation.from_matrix(np.stack([cam.R for cam in cam_list], 0))
    T_list = np.stack([cam.T for cam in cam_list])
    slerp = Slerp(list(range(len(R_list))), R_list)
    times = np.linspace(0, len(R_list)-1, (num_inter+2)*len(R_list))
    Q_inters = slerp(times)
    inter_1d = interp1d(np.array(list(range(len(R_list)))), T_list, axis=0)
    T_inters = inter_1d(times)
    
    # 获取原始图像尺寸以保持渲染分辨率
    orig_image = cam_list[0].original_image
    orig_height, orig_width = orig_image.shape[1], orig_image.shape[2]
    
    # 创建小尺寸共享图像以节省显存，但保持正确的渲染分辨率
    if use_dummy_image:
        # 创建小尺寸dummy图像，通过override参数设置正确的渲染尺寸
        shared_dummy_image = torch.zeros((3, 64, 64), dtype=torch.float32, device='cpu')
        shared_dummy_mask = torch.ones((1, 64, 64), dtype=torch.float32, device='cpu')
        image_name = "interpolated"
        print(f"使用小尺寸CPU图像(64x64)，渲染分辨率: {orig_height}x{orig_width}")
    else:
        shared_dummy_image = orig_image
        shared_dummy_mask = torch.ones_like(orig_image)
        image_name = cam_list[0].image_name
        print(f"使用完整图像，尺寸: {orig_height}x{orig_width}")
    
    # 创建插值相机列表
    interpolated_cameras = []
    for i in range(len(times)):
        if use_dummy_image:
            # 使用小图像但覆盖尺寸为原始分辨率
            cam = Camera(-1, Q_inters.as_matrix()[i], T_inters[i], cam_list[0].FoVx, cam_list[0].FoVy, 
                         shared_dummy_image, shared_dummy_mask, image_name, -1, 
                         data_device='cpu', override_width=orig_width, override_height=orig_height)
        else:
            # 使用完整图像
            cam = Camera(-1, Q_inters.as_matrix()[i], T_inters[i], cam_list[0].FoVx, cam_list[0].FoVy, 
                         shared_dummy_image, shared_dummy_mask, image_name, -1, data_device='cuda')
        interpolated_cameras.append(cam)
    
    print(f"插值生成{len(interpolated_cameras)}个相机")
    return interpolated_cameras
    
    