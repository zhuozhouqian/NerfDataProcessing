import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser(description="preprocess d455 dataset to sdfstudio dataset")

parser.add_argument("--input_path", dest="input_path", help="path to d455 scene")
parser.set_defaults(im_name="NONE")

parser.add_argument("--output_path", dest="output_path", help="path to output")
parser.set_defaults(store_name="NONE")

args = parser.parse_args()

output_path = Path(args.output_path)  # "data/d455/chair01"
input_path = Path(args.input_path)  # "/home/pc/project/zeyutt/sdfstudio/data/SJTU/chair01"

output_path.mkdir(parents=True, exist_ok=True)

# load color
color_path = input_path / "color" 
color_paths = sorted(glob.glob(os.path.join(color_path, "*.png")), key=lambda x: int(os.path.basename(x)[:-4]))


# load depth
depth_path = input_path / "depth" 
depth_paths = sorted(glob.glob(os.path.join(depth_path, "*.png")), key=lambda x: int(os.path.basename(x)[:-4]))

# load mask
mask_path = input_path / "mask" 
mask_paths = sorted(glob.glob(os.path.join(mask_path, "*.png")), key=lambda x: int(os.path.basename(x)[:-4]))

# load intrinsic
intrinsic_path = input_path / "intrinsic" / "intrinsic_color.txt"
camera_intrinsic = np.loadtxt(intrinsic_path)

# load pose
pose_path = input_path / "pose"
poses = []
pose_paths = sorted(glob.glob(os.path.join(pose_path, "*.txt")), key=lambda x: int(os.path.basename(x)[:-4]))
for pose_path in pose_paths:
    c2w = np.loadtxt(pose_path)
    poses.append(c2w)
poses = np.array(poses)

# deal with invalid poses
valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)

center = (min_vertices + max_vertices) / 2.0
scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)
print(center, scale)

# we should normalize pose to unit cube
poses[:, :3, 3] -= center
poses[:, :3, 3] *= scale

# inverse normalization
scale_mat = np.eye(4).astype(np.float32)  # world to gt
scale_mat[:3, 3] -= center
scale_mat[:3] *= scale
scale_mat = np.linalg.inv(scale_mat)

# copy image
sample_img = cv2.imread(str(color_paths[0]))
H, W, _ = sample_img.shape  # 480 x 640

# get smallest side to generate square crop
target_crop = min(H, W)

target_size = 384
trans_totensor = transforms.Compose(
    [
        transforms.CenterCrop(target_crop),
        transforms.Resize(target_size, interpolation=PIL.Image.BILINEAR),
    ]
)
trans_depth = transforms.Compose(
    [
        transforms.CenterCrop(target_crop),
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
    ]
)
trans_mask = transforms.Compose(
    [
        transforms.CenterCrop(target_crop),
        transforms.Resize(target_size, interpolation=PIL.Image.NEAREST),
    ]
)

# center crop by min_dim
offset_x = (W - target_crop) * 0.5
offset_y = (H - target_crop) * 0.5
camera_intrinsic[0, 2] -= offset_x
camera_intrinsic[1, 2] -= offset_y
# resize from min_dim x min_dim -> to 384 x 384
resize_factor = target_size / target_crop
camera_intrinsic[:2, :] *= resize_factor

K = camera_intrinsic

# print(valid_poses.sum())
# print(poses.shape)
# print(len(color_paths))
# print(len(depth_paths))
# print(len(mask_paths))

# sys.exit()

frames = []
out_index = 0
for idx, (valid, pose, image_path, depth_path, mask_path) in enumerate(zip(valid_poses, poses, color_paths, depth_paths, mask_paths)):

    # if idx % 10 != 0: % select per 10 frame
    #     print(idx)
    #     continue
    if not valid:
        continue

    target_image = output_path / f"{out_index:06d}_rgb.png"
    print(target_image)
    img = Image.open(image_path)
    img_tensor = trans_totensor(img)
    img_tensor.save(target_image)

    target_depth = output_path / f"{out_index:06d}_depth.npy"
    depth = cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)
    depth = torch.from_numpy(depth.astype(np.float32))[None,None]
    depth_tensor = trans_depth(depth)
    # print(depth_tensor.shape)
    # print(depth_tensor.dtype)
    # sys.exit()
    depth_tensor =(depth_tensor/1000).clamp(max=5.0)
    np.save(target_depth,depth_tensor[0,0].numpy())

    target_mask = output_path / f"{out_index:06d}_mask.npy"
    # mask = np.load(mask_path,dtype=np.int64)
    mask = Image.open(mask_path)
    mask_tensor = trans_mask(mask)
    mask_tensor.save(output_path / f"{out_index:06d}_mask.png")
    np.save(target_mask,np.array(mask_tensor).astype(np.float32)/255)
    

    rgb_path = str(target_image.relative_to(output_path))
    out_depth_path = str(target_depth.relative_to(output_path))
    out_mask_path = str(target_mask.relative_to(output_path))
    frame = {
        "rgb_path": rgb_path,
        "depth_path": out_depth_path,
        "mask_path" : out_mask_path,
        "camtoworld": pose.tolist(),
        "intrinsics": K.tolist(),
        "mono_depth_path": rgb_path.replace("_rgb.png", "_depth.npy"),
        "mono_normal_path": rgb_path.replace("_rgb.png", "_normal.npy"),
    }

    frames.append(frame)
    out_index += 1

# scene bbox for the scannet scene
scene_box = {
    "aabb": [[-1, -1, -1], [1, 1, 1]],
    "near": 0.05,
    "far": 2.5,
    "radius": 1.0,
    "collider_type": "near_far",
}

# meta data
output_data = {
    "camera_model": "OPENCV",
    "height": target_size,
    "width": target_size,
    "has_mono_prior": True,
    "pairs": None,
    "worldtogt": scale_mat.tolist(),
    "scene_box": scene_box,
}

output_data["frames"] = frames

# save as json
with open(output_path / "meta_data.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)
