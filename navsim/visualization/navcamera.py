from typing import List, Optional, Tuple,Dict, Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import ImageColor
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import matplotlib.patches as patches
import torch
from navsim.common.dataclasses import Camera, Lidar, Annotations, Trajectory
from navsim.common.enums import LidarIndex, BoundingBoxIndex
from navsim.visualization.config import AGENT_CONFIG
from navsim.visualization.lidar import filter_lidar_pc, get_lidar_pc_color
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
import os
import matplotlib

def add_camera_ax(ax: plt.Axes, camera: Camera) -> plt.Axes:
    """
    Adds camera image to matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :return: ax object with image
    """
    ax.imshow(camera.image)
    return ax


def add_lidar_to_camera_ax(ax: plt.Axes, camera: Camera, lidar: Lidar) -> plt.Axes:
    """
    Adds camera image with lidar point cloud on matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param lidar: navsim lidar dataclass
    :return: ax object with image
    """

    image, lidar_pc = camera.image.copy(), lidar.lidar_pc.copy()
    image_height, image_width = image.shape[:2]
    print(lidar_pc.shape)
    lidar_pc = filter_lidar_pc(lidar_pc)
    lidar_pc_colors = np.array(get_lidar_pc_color(lidar_pc))

    pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(
        lidar_pc,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
        camera.intrinsics,
        img_shape=(image_height, image_width),
    )
    print(image_height, image_width)
    pc_in_cam = pc_in_cam[pc_in_fov_mask]
    print(pc_in_cam.shape)
    for (x, y), color in zip(pc_in_cam, lidar_pc_colors[pc_in_fov_mask]):
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.circle(image, (int(x), int(y)), 5, color, -1)

    ax.imshow(image)
    return ax

def dense_map(Pts, n, m, grid):
    """
    Generate a dense depth map from the sparse points.
    :param Pts: The sparse depth points (x, y, depth)
    :param n: Image width
    :param m: Image height
    :param grid: Neighborhood grid size for interpolation
    :return: Dense depth map
    """
    ng = 2 * grid + 1
    mX = np.zeros((m, n)) + float("inf")
    mY = np.zeros((m, n)) + float("inf")
    mD = np.zeros((m, n))
    
    # Fill the sparse depth points into the mX, mY, and mD matrices
    mX[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    mY[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    mD[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[2]
    
    KmX = np.zeros((ng, ng, m - ng, n - ng))
    KmY = np.zeros((ng, ng, m - ng, n - ng))
    KmD = np.zeros((ng, ng, m - ng, n - ng))
    
    for i in range(ng):
        for j in range(ng):
            KmX[i, j] = mX[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmY[i, j] = mY[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmD[i, j] = mD[i: (m - ng + i), j: (n - ng + j)]
    
    S = np.zeros_like(KmD[0, 0])
    Y = np.zeros_like(KmD[0, 0])
    
    for i in range(ng):
        for j in range(ng):
            s = 1 / np.sqrt(KmX[i, j] * KmX[i, j] + KmY[i, j] * KmY[i, j])
            Y = Y + s * KmD[i, j]
            S = S + s
    
    S[S == 0] = 1
    out = np.zeros((m, n))
    out[grid + 1: -grid, grid + 1: -grid] = Y / S
    return out

def add_lidar_to_camera_ax_with_depth(
    ax: plt.Axes, camera: Camera, lidar: Lidar
) -> Tuple[plt.Axes, npt.NDArray[np.float32]]:
    """
    Adds camera image with lidar point cloud on matplotlib ax object and generates depth map.
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param lidar: navsim lidar dataclass
    :return: ax object with image and the dense depth map
    """
    
    # 获取图像和LIDAR点云
    image, lidar_pc = camera.image.copy(), lidar.lidar_pc.copy()
    image_height, image_width = image.shape[:2]
    print("LIDAR point cloud shape:", lidar_pc.shape)

    lidar_pc = filter_lidar_pc(lidar_pc)
    lidar_pc_colors = np.array(get_lidar_pc_color(lidar_pc))

    pc_in_cam, pc_in_fov_mask, depth_values = _transform_pcs_to_images_with_depth(
        lidar_pc,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
        camera.intrinsics,
        img_shape=(image_height, image_width),
    )

    print("Image dimensions:", image_height, image_width)
    print("Number of points in field of view:", np.sum(pc_in_fov_mask))

    # 只保留视野内的点
    pc_in_cam = pc_in_cam[pc_in_fov_mask]
    depth_values = depth_values[pc_in_fov_mask]  # 只保留视野内的深度值
    lidar_pc_colors = lidar_pc_colors[pc_in_fov_mask]  # 只保留视野内的颜色

    # 创建空的深度图，初始化为无穷远值（大于可能的最大深度）
    depth_map_intermediate = np.full((image_height, image_width), np.inf)

    # 将 LIDAR 点投影到图像平面并绘制
    for (x, y), depth, color in zip(pc_in_cam[:, 0:2], depth_values, lidar_pc_colors):  # 只提取 (x, y)
        color = (int(color[0]), int(color[1]), int(color[2]))  # 转换为整数颜色
        x, y = int(x), int(y)
        if 0 <= x < image_width and 0 <= y < image_height:
            color = (int(color[0]), int(color[1]), int(color[2]))
            cv2.circle(image, (int(x), int(y)), 5, color, -1)
            depth_map_intermediate[y, x] = min(depth_map_intermediate[y, x], depth)  # 更新深度图

    ax.imshow(image)

    # 打印深度图的最小值和最大值，检查其范围
    valid_depth_values = depth_map_intermediate[depth_map_intermediate != np.inf]

    print("Depth map min value:", np.min(valid_depth_values))
    print("Depth map max value:", np.max(valid_depth_values))

    # 处理 np.inf 值：将 np.inf 替换为深度图中的最大有效深度
    if len(valid_depth_values) > 0:
        max_depth = np.max(valid_depth_values)
        depth_map_intermediate[np.isinf(depth_map_intermediate)] = max_depth
    else:
        max_depth = 0  # 如果没有有效的深度值，则使用 0 作为最大深度值
        depth_map_intermediate[np.isinf(depth_map_intermediate)] = max_depth

    # 使用 dense_map 函数生成密集深度图
    dense_depth_map = dense_map(
        np.array([pc_in_cam[:, 0], pc_in_cam[:, 1], depth_values]),  # x, y, depth
        image_width,
        image_height,
        grid=8  # 你可以调整这个值来控制平滑度
    )

    # 归一化深度图到 0-255 范围
    dense_depth_map_normalized = cv2.normalize(dense_depth_map, None, 0, 255, cv2.NORM_MINMAX)

    # 使用 'Spectral_r' 颜色映射应用于密集深度图
    colormap = plt.get_cmap('Spectral_r')
    colored_dense_depth_map = colormap(dense_depth_map_normalized / 255.0)  # 归一化后映射到 [0, 1]

    # 将 rgba 转换为 rgb
    colored_dense_depth_map = (colored_dense_depth_map[:, :, :3] * 255).astype(np.uint8)

    # 获取文件名和输出目录
    outdir = './output_dense_depth_maps'
    os.makedirs(outdir, exist_ok=True)
    filename = "lidar_dense_depth_image"  # 这里可以用你自己的文件名

    # 保存密集深度图
    cv2.imwrite(os.path.join(outdir, filename + '_dense_depth.png'), colored_dense_depth_map)

    # 合并图像和密集深度图
    split_region = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
    combined_result = cv2.hconcat([image, split_region, colored_dense_depth_map])

    # 保存合并结果
    cv2.imwrite(os.path.join(outdir, filename + '_combined.png'), combined_result)

    return ax, dense_depth_map

def add_annotations_to_camera_ax(ax: plt.Axes, camera: Camera, annotations: Annotations) -> plt.Axes:
    """
    Adds camera image with bounding boxes on matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param annotations: navsim annotations dataclass
    :return: ax object with image
    """

    box_labels = annotations.names
    boxes = _transform_annotations_to_camera(
        annotations.boxes,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
    )
    box_positions, box_dimensions, box_heading = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.DIMENSION],
        boxes[:, BoundingBoxIndex.HEADING],
    )
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box_dimensions.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
    
    corners = _rotation_3d_in_axis(corners, box_heading, axis=1)
    corners += box_positions.reshape(-1, 1, 3)

    # Then draw project corners to image.
    box_corners, corners_pc_in_fov = _transform_points_to_image(corners.reshape(-1, 3), camera.intrinsics)
    box_corners = box_corners.reshape(-1, 8, 2)
    corners_pc_in_fov = corners_pc_in_fov.reshape(-1, 8)
    valid_corners = corners_pc_in_fov.any(-1)

    box_corners, box_labels = box_corners[valid_corners], box_labels[valid_corners]
    image = _plot_rect_3d_on_img(camera.image.copy(), box_corners, box_labels)

    ax.imshow(image)
    return ax

def add_2d_annotations_to_camera_ax(ax: plt.Axes, camera: Camera, annotations: Annotations) -> plt.Axes:
    """
    Adds camera image with 2D bounding boxes on matplotlib ax object.
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param annotations: navsim annotations dataclass
    :return: ax object with image
    """

    box_labels = annotations.names
    boxes = _transform_annotations_to_camera(
        annotations.boxes,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
    )
    box_positions, box_dimensions, box_heading = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.DIMENSION],
        boxes[:, BoundingBoxIndex.HEADING],
    )
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box_dimensions.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

    corners = _rotation_3d_in_axis(corners, box_heading, axis=1)
    corners += box_positions.reshape(-1, 1, 3)

    # Project corners to image.
    box_corners, corners_pc_in_fov = _transform_points_to_image(corners.reshape(-1, 3), camera.intrinsics)
    box_corners = box_corners.reshape(-1, 8, 2)
    corners_pc_in_fov = corners_pc_in_fov.reshape(-1, 8)
    valid_corners = corners_pc_in_fov.any(-1)

    box_corners, box_labels = box_corners[valid_corners], box_labels[valid_corners]

    # Calculate 2D bounding boxes from projected 3D corners.
    box_2d_list = []
    for corner_set in box_corners:
        min_x = np.min(corner_set[:, 0])
        max_x = np.max(corner_set[:, 0])
        min_y = np.min(corner_set[:, 1])
        max_y = np.max(corner_set[:, 1])
        box_2d_list.append([min_x, min_y, max_x, max_y])

    box_2d_list = np.array(box_2d_list)
    print(box_2d_list)
    image = _plot_rect_2d_on_img(camera.image.copy(), box_2d_list, box_labels)

    ax.imshow(image)
    return ax

def _plot_rect_2d_on_img(image: np.ndarray, boxes_2d: np.ndarray, labels: list, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    Draws 2D bounding boxes on an image.
    :param image: input image (numpy array)
    :param boxes_2d: 2D bounding boxes (numpy array), each box is [x_min, y_min, x_max, y_max]
    :param labels: list of labels for each bounding box
    :param color: color of the bounding box (BGR format)
    :param thickness: thickness of the bounding box lines
    :return: image with bounding boxes drawn
    """

    image_copy = image.copy()

    if boxes_2d is not None and len(boxes_2d) > 0:
        # if labels is None:
        #   labels = []
        for i, box in enumerate(boxes_2d):
            x_min, y_min, x_max, y_max = map(int, box)  # Convert coordinates to integers
            cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, thickness)

            # # Add label
            # if labels and i < len(labels):
            #     label = labels[i]
            #     cv2.putText(image_copy, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return image_copy


def _transform_annotations_to_camera(
    boxes: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Helper function to transform bounding boxes into camera frame
    TODO: Refactor
    :param boxes: array representation of bounding boxes
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :return: bounding boxes in camera coordinates
    """

    locs, rots = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.HEADING :],
    )
    dims_cam = boxes[
        :, [BoundingBoxIndex.LENGTH, BoundingBoxIndex.HEIGHT, BoundingBoxIndex.WIDTH]
    ]  # l, w, h -> l, h, w

    rots_cam = np.zeros_like(rots)
    for idx, rot in enumerate(rots):
        rot = Quaternion(axis=[0, 0, 1], radians=rot)
        rot = Quaternion(matrix=sensor2lidar_rotation).inverse * rot
        rots_cam[idx] = -rot.yaw_pitch_roll[0]

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    locs_cam = np.concatenate([locs, np.ones_like(locs)[:, :1]], -1)  # -1, 4
    locs_cam = lidar2cam_rt.T @ locs_cam.T
    locs_cam = locs_cam.T
    locs_cam = locs_cam[:, :-1]
    return np.concatenate([locs_cam, dims_cam, rots_cam], -1)


def _rotation_3d_in_axis(points: npt.NDArray[np.float32], angles: npt.NDArray[np.float32], axis: int = 0):
    """
    Rotate 3D points by angles according to axis.
    TODO: Refactor
    :param points: array of points
    :param angles: array of angles
    :param axis: axis to perform rotation, defaults to 0
    :raises value: _description_
    :raises ValueError: if axis invalid
    :return: rotated points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                np.stack([rot_cos, zeros, -rot_sin]),
                np.stack([zeros, ones, zeros]),
                np.stack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                np.stack([rot_cos, -rot_sin, zeros]),
                np.stack([rot_sin, rot_cos, zeros]),
                np.stack([zeros, zeros, ones]),
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                np.stack([zeros, rot_cos, -rot_sin]),
                np.stack([zeros, rot_sin, rot_cos]),
                np.stack([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError(f"axis should in range [0, 1, 2], got {axis}")
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def _plot_rect_3d_on_img(
    image: npt.NDArray[np.float32],
    box_corners: npt.NDArray[np.float32],
    box_labels: List[str],
    thickness: int = 3,
) -> npt.NDArray[np.uint8]:
    """
    Plot the boundary lines of 3D rectangular on 2D images.
    TODO: refactor
    :param image:  The numpy array of image.
    :param box_corners: Coordinates of the corners of 3D, shape of [N, 8, 2].
    :param box_labels: labels of boxes for coloring
    :param thickness: pixel width of liens, defaults to 3
    :return: image with 3D bounding boxes
    """
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    for i in range(len(box_corners)):
        layer = tracked_object_types[box_labels[i]]
        color = ImageColor.getcolor(AGENT_CONFIG[layer]["fill_color"], "RGB")
        corners = box_corners[i].astype(int)
        for start, end in line_indices:
            cv2.line(
                image,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )
    return image.astype(np.uint8)


def _transform_points_to_image(
    points: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    image_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    Transforms points in camera frame to image pixel coordinates
    TODO: refactor
    :param points: points in camera frame
    :param intrinsic: camera intrinsics
    :param image_shape: shape of image in pixel
    :param eps: lower threshold of points, defaults to 1e-3
    :return: points in pixel coordinates, mask of values in frame
    """
    points = points[:, :3]

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

    pc_img = np.concatenate([points, np.ones_like(points)[:, :1]], -1)
    pc_img = viewpad @ pc_img.T
    pc_img = pc_img.T

    cur_pc_in_fov = pc_img[:, 2] > eps
    pc_img = pc_img[..., 0:2] / np.maximum(pc_img[..., 2:3], np.ones_like(pc_img[..., 2:3]) * eps)
    if image_shape is not None:
        img_h, img_w = image_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (pc_img[:, 0] < (img_w - 1))
            & (pc_img[:, 0] > 0)
            & (pc_img[:, 1] < (img_h - 1))
            & (pc_img[:, 1] > 0)
        )
    return pc_img, cur_pc_in_fov


def _transform_pcs_to_images(
    lidar_pc: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    img_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    Transforms points in camera frame to image pixel coordinates
    TODO: refactor
    :param lidar_pc: lidar point cloud
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :param intrinsic: camera intrinsics
    :param img_shape: image shape in pixels, defaults to None
    :param eps: threshold for lidar pc height, defaults to 1e-3
    :return: lidar pc in pixel coordinates, mask of values in frame
    """
    pc_xyz = lidar_pc[LidarIndex.POSITION, :].T

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    lidar2img_rt = viewpad @ lidar2cam_rt.T

    cur_pc_xyz = np.concatenate([pc_xyz, np.ones_like(pc_xyz)[:, :1]], -1)
    cur_pc_cam = lidar2img_rt @ cur_pc_xyz.T
    cur_pc_cam = cur_pc_cam.T
    cur_pc_in_fov = cur_pc_cam[:, 2] > eps
    cur_pc_cam = cur_pc_cam[..., 0:2] / np.maximum(cur_pc_cam[..., 2:3], np.ones_like(cur_pc_cam[..., 2:3]) * eps)

    if img_shape is not None:
        img_h, img_w = img_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (cur_pc_cam[:, 0] < (img_w - 1))
            & (cur_pc_cam[:, 0] > 0)
            & (cur_pc_cam[:, 1] < (img_h - 1))
            & (cur_pc_cam[:, 1] > 0)
        )
    return cur_pc_cam, cur_pc_in_fov

def _transform_pcs_to_images_with_depth(
    lidar_pc: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    img_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
    """
    Transforms points in camera frame to image pixel coordinates and returns depth values.
    :param lidar_pc: lidar point cloud
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :param intrinsic: camera intrinsics
    :param img_shape: image shape in pixels, defaults to None
    :param eps: threshold for lidar pc height, defaults to 1e-3
    :return: lidar pc in pixel coordinates, mask of values in frame, depth values in frame
    """
    # 获取 LIDAR 点云的 (x, y, z) 坐标
    pc_xyz = lidar_pc[LidarIndex.POSITION, :].T  # shape should be (n, 3), each row [x, y, z]
    print(f"pc_xyz shape: {pc_xyz.shape}")  # Debug: Check shape of lidar point cloud

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    lidar2img_rt = viewpad @ lidar2cam_rt.T

    # 添加齐次坐标，确保是 4D 向量 (x, y, z, 1)
    cur_pc_xyz = np.concatenate([pc_xyz, np.ones_like(pc_xyz)[:, :1]], axis=-1)  # shape: (n, 4)
    print(f"cur_pc_xyz shape after concat: {cur_pc_xyz.shape}")  # Debug: Check shape after concat

    # 投影到相机坐标系
    cur_pc_cam = lidar2img_rt @ cur_pc_xyz.T  # 投影矩阵 * 点云
    cur_pc_cam = cur_pc_cam.T  # 转置回来 (n, 4)
    print(f"cur_pc_cam shape after projection: {cur_pc_cam.shape}")  # Debug: Check shape after projection

    # 过滤掉视野外的点
    cur_pc_in_fov = cur_pc_cam[:, 2] > eps  # z > eps 才是有效点

    # 进行齐次坐标的标准化：通过 Z 值标准化 (x, y)，得到二维坐标
    cur_pc_cam[:, 0:2] = cur_pc_cam[:, 0:2] / np.maximum(cur_pc_cam[:, 2:3], np.ones_like(cur_pc_cam[:, 2:3]) * eps)
    print(f"cur_pc_cam shape after normalization: {cur_pc_cam.shape}")  # Debug: Check shape after normalization

    # 获取深度值（即 Z 坐标）
    depth_values = cur_pc_cam[:, 2]  # 深度值为 Z 坐标
    print(f"depth_values shape: {depth_values.shape}")  # Debug: Check depth_values shape

    # 只保留在视野内的点
    if img_shape is not None:
        img_h, img_w = img_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (cur_pc_cam[:, 0] < (img_w - 1))
            & (cur_pc_cam[:, 0] > 0)
            & (cur_pc_cam[:, 1] < (img_h - 1))
            & (cur_pc_cam[:, 1] > 0)
        )

    return cur_pc_cam, cur_pc_in_fov, depth_values





def _is_point_in_image(point: npt.NDArray[np.float32], image_width: int, image_height: int) -> bool:
    """Returns True if a 2D pixel coordinate lies inside the image rectangle."""
    x, y = float(point[0]), float(point[1])
    return 0.0 <= x <= (image_width - 1) and 0.0 <= y <= (image_height - 1)


def _clip_line_segment_to_image_rect(
    p0: npt.NDArray[np.float32],
    p1: npt.NDArray[np.float32],
    image_width: int,
    image_height: int,
) -> Optional[Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:
    """
    Clips a 2D line segment to the image rectangle using the Liang-Barsky algorithm.
    Returns the clipped endpoints, or None if the segment does not intersect the image.
    """
    x_min, x_max = 0.0, float(image_width - 1)
    y_min, y_max = 0.0, float(image_height - 1)

    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx = x1 - x0
    dy = y1 - y0

    p = [-dx, dx, -dy, dy]
    q = [x0 - x_min, x_max - x0, y0 - y_min, y_max - y0]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
            continue

        t = qi / pi
        if pi < 0:
            u1 = max(u1, t)
        else:
            u2 = min(u2, t)

        if u1 > u2:
            return None

    clipped_p0 = np.array([x0 + u1 * dx, y0 + u1 * dy], dtype=np.float32)
    clipped_p1 = np.array([x0 + u2 * dx, y0 + u2 * dy], dtype=np.float32)
    return clipped_p0, clipped_p1


def _build_visible_polylines(
    projected_points: npt.NDArray[np.float32],
    in_front_mask: npt.NDArray[np.bool_],
    image_width: int,
    image_height: int,
) -> List[npt.NDArray[np.float32]]:
    """
    Builds continuous visible trajectory polylines by clipping each valid segment to the image rectangle.
    Segments that leave and re-enter the image are split rather than being incorrectly bridged.
    """
    polylines: List[npt.NDArray[np.float32]] = []
    current_polyline: List[npt.NDArray[np.float32]] = []
    endpoint_tol = 1e-2

    for idx in range(len(projected_points) - 1):
        if not (in_front_mask[idx] and in_front_mask[idx + 1]):
            if len(current_polyline) >= 2:
                polylines.append(np.asarray(current_polyline, dtype=np.float32))
            current_polyline = []
            continue

        clipped_segment = _clip_line_segment_to_image_rect(
            projected_points[idx],
            projected_points[idx + 1],
            image_width=image_width,
            image_height=image_height,
        )
        if clipped_segment is None:
            if len(current_polyline) >= 2:
                polylines.append(np.asarray(current_polyline, dtype=np.float32))
            current_polyline = []
            continue

        seg_start, seg_end = clipped_segment
        if not current_polyline:
            current_polyline = [seg_start, seg_end]
            continue

        if np.linalg.norm(current_polyline[-1] - seg_start) <= endpoint_tol:
            current_polyline.append(seg_end)
        else:
            if len(current_polyline) >= 2:
                polylines.append(np.asarray(current_polyline, dtype=np.float32))
            current_polyline = [seg_start, seg_end]

    if len(current_polyline) >= 2:
        polylines.append(np.asarray(current_polyline, dtype=np.float32))

    cleaned_polylines: List[npt.NDArray[np.float32]] = []
    for polyline in polylines:
        cleaned_points = [polyline[0]]
        for point in polyline[1:]:
            if np.linalg.norm(point - cleaned_points[-1]) > 1e-4:
                cleaned_points.append(point)
        if len(cleaned_points) >= 2:
            cleaned_polylines.append(np.asarray(cleaned_points, dtype=np.float32))

    return cleaned_polylines


def _trajectory_poses_to_numpy(trajectory: Trajectory) -> npt.NDArray[np.float32]:
    """Converts trajectory poses to a NumPy array with shape [N, >=2]."""
    poses = trajectory.poses
    if isinstance(poses, torch.Tensor):
        poses = poses.detach().cpu().numpy()
    else:
        poses = np.asarray(poses)
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] < 2:
        raise ValueError("trajectory.poses must have shape [N, >=2]")
    return poses


def project_trajectory_to_camera(
    camera: Camera,
    trajectory: Trajectory,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Projects a trajectory to camera pixel coordinates and returns visibility in front of camera."""
    poses_2d = _trajectory_poses_to_numpy(trajectory)[:, :2]
    zeros = np.zeros((poses_2d.shape[0], 1), dtype=np.float32)
    poses_3d = np.concatenate([poses_2d, zeros], axis=1)
    origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    poses = np.concatenate([origin, poses_3d], axis=0)

    projected_points, in_front_mask = _transform_pcs_to_images(
        poses.T,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
        camera.intrinsics,
        img_shape=None,
    )
    return projected_points[1:], in_front_mask[1:]


def get_centered_camera_crop(
    camera: Camera,
    trajectories: Any,
    crop_width: Optional[float] = None,
    side_margin_px: float = 40.0,
) -> Tuple[float, float]:
    """
    Computes a horizontal crop window centered around the visible projected trajectory/trajectories.
    The default crop width is a square crop (image height), which matches a square BEV panel width.
    """
    image_height, image_width = camera.image.shape[:2]
    base_crop_width = float(image_height if crop_width is None else crop_width)
    base_crop_width = float(np.clip(base_crop_width, 32.0, float(image_width)))

    if isinstance(trajectories, (list, tuple)):
        traj_list = [traj for traj in trajectories if traj is not None]
    else:
        traj_list = [trajectories] if trajectories is not None else []

    x_samples: List[np.ndarray] = []
    for trajectory in traj_list:
        try:
            projected_points, in_front_mask = project_trajectory_to_camera(camera, trajectory)
        except Exception:
            continue
        in_image_mask = (
            in_front_mask
            & (projected_points[:, 0] >= 0.0)
            & (projected_points[:, 0] <= float(image_width - 1))
            & (projected_points[:, 1] >= 0.0)
            & (projected_points[:, 1] <= float(image_height - 1))
        )
        if np.any(in_image_mask):
            x_samples.append(projected_points[in_image_mask, 0].astype(np.float32, copy=False))

    if x_samples:
        all_x = np.concatenate(x_samples, axis=0)
        x_min = float(np.min(all_x))
        x_max = float(np.max(all_x))
        x_center = 0.5 * (x_min + x_max)
        needed_width = (x_max - x_min) + 2.0 * float(side_margin_px)
        crop_width_px = float(np.clip(max(base_crop_width, needed_width), 32.0, float(image_width)))
    else:
        x_center = 0.5 * float(image_width - 1)
        crop_width_px = base_crop_width

    x_left = x_center - crop_width_px / 2.0
    x_right = x_center + crop_width_px / 2.0

    if x_left < 0.0:
        x_right -= x_left
        x_left = 0.0
    if x_right > float(image_width):
        x_left -= x_right - float(image_width)
        x_right = float(image_width)
    x_left = max(0.0, x_left)
    x_right = min(float(image_width), x_right)

    return float(x_left), float(x_right)


def apply_centered_camera_crop(
    ax: plt.Axes,
    camera: Camera,
    trajectories: Any,
    crop_width: Optional[float] = None,
    side_margin_px: float = 40.0,
) -> Tuple[float, float]:
    """Applies a trajectory-centered horizontal crop to a camera axis and keeps the full image height."""
    image_height, _ = camera.image.shape[:2]
    x_left, x_right = get_centered_camera_crop(
        camera=camera,
        trajectories=trajectories,
        crop_width=crop_width,
        side_margin_px=side_margin_px,
    )
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(float(image_height), 0.0)
    try:
        ax.set_box_aspect(float(image_height) / max(float(x_right - x_left), 1e-6))
    except Exception:
        pass
    ax.set_anchor("C")
    return x_left, x_right


def add_trajectory_to_camera_ax(ax: plt.Axes, camera: Camera, trajectory: Trajectory, config: Dict[str, Any]) -> plt.Axes:
    """
    Adds a trajectory polyline to the camera image and draws an endpoint-aligned arrow.
    The final visible segment is trimmed so the arrow can fully cover the line endpoint.
    """
    projected_points, in_front_mask = project_trajectory_to_camera(camera, trajectory)

    image_height, image_width = camera.image.shape[:2]
    visible_polylines = _build_visible_polylines(
        projected_points,
        in_front_mask,
        image_width=image_width,
        image_height=image_height,
    )

    if not visible_polylines:
        return ax

    line_capstyle = config.get("line_capstyle", "round")
    line_joinstyle = config.get("line_joinstyle", "round")

    arrow_polyline = visible_polylines[-1]
    if len(arrow_polyline) < 2:
        for polyline in visible_polylines:
            ax.plot(
                polyline[:, 0],
                polyline[:, 1],
                color=config["line_color"],
                alpha=config["line_color_alpha"],
                linewidth=config["line_width"],
                linestyle=config["line_style"],
                marker=config.get("marker", None),
                markersize=config.get("marker_size", 0),
                markeredgecolor=config.get("marker_edge_color", None),
                solid_capstyle=line_capstyle,
                solid_joinstyle=line_joinstyle,
                zorder=config["zorder"],
            )
        return ax

    last_point = arrow_polyline[-1]
    second_last_point = arrow_polyline[-2]
    direction = last_point - second_last_point
    segment_length = float(np.linalg.norm(direction))
    if segment_length < 1e-6:
        for polyline in visible_polylines:
            ax.plot(
                polyline[:, 0],
                polyline[:, 1],
                color=config["line_color"],
                alpha=config["line_color_alpha"],
                linewidth=config["line_width"],
                linestyle=config["line_style"],
                marker=config.get("marker", None),
                markersize=config.get("marker_size", 0),
                markeredgecolor=config.get("marker_edge_color", None),
                solid_capstyle=line_capstyle,
                solid_joinstyle=line_joinstyle,
                zorder=config["zorder"],
            )
        return ax

    unit_direction = direction / segment_length
    default_arrow_length_px = max(16.0, 4.0 * float(config["line_width"]))
    arrow_length_px = float(config.get("arrow_length_px", default_arrow_length_px))
    arrow_length_px = min(arrow_length_px, segment_length)

    arrow_tip_backoff_px = float(config.get("arrow_tip_backoff_px", 0.0))
    arrow_tip_backoff_px = min(max(arrow_tip_backoff_px, 0.0), arrow_length_px * 0.5)

    arrow_tip = last_point - unit_direction * arrow_tip_backoff_px
    arrow_tail = arrow_tip - unit_direction * arrow_length_px

    cover_reserve_px = max(float(config["line_width"]) * 2.2, arrow_length_px * 0.72)
    cover_reserve_px = min(cover_reserve_px, max(segment_length - 1e-3, 1e-3))

    trimmed_last_polyline = arrow_polyline.copy()
    trimmed_last_polyline[-1] = last_point - unit_direction * cover_reserve_px

    for polyline in visible_polylines[:-1]:
        ax.plot(
            polyline[:, 0],
            polyline[:, 1],
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            marker=config.get("marker", None),
            markersize=config.get("marker_size", 0),
            markeredgecolor=config.get("marker_edge_color", None),
            solid_capstyle=line_capstyle,
            solid_joinstyle=line_joinstyle,
            zorder=config["zorder"],
        )

    ax.plot(
        trimmed_last_polyline[:, 0],
        trimmed_last_polyline[:, 1],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        marker=config.get("marker", None),
        markersize=config.get("marker_size", 0),
        markeredgecolor=config.get("marker_edge_color", None),
        solid_capstyle="butt",
        solid_joinstyle=line_joinstyle,
        zorder=config["zorder"],
    )

    mutation_scale = float(config.get("arrow_mutation_scale", max(18.0, 3.8 * float(config["line_width"]))))

    arrow = patches.FancyArrowPatch(
        posA=(float(arrow_tail[0]), float(arrow_tail[1])),
        posB=(float(arrow_tip[0]), float(arrow_tip[1])),
        arrowstyle=config.get("arrow_style", "-|>"),
        mutation_scale=mutation_scale,
        shrinkA=0.0,
        shrinkB=0.0,
        fc=config["arrow_color"],
        ec=config.get("arrow_edge_color", config["arrow_color"]),
        alpha=config["arrow_alpha"],
        linewidth=config.get("arrow_line_width", max(config["line_width"], 4.6)),
        connectionstyle="arc3,rad=0.0",
        capstyle="round",
        joinstyle="round",
        zorder=config["zorder"] + 2,
    )
    ax.add_patch(arrow)

    return ax


# from scipy.interpolate import splprep, splev

# def add_trajectory_to_camera_ax(ax: plt.Axes, camera: Camera, trajectory: Trajectory, config: Dict[str, Any]) -> plt.Axes:
#     poses_2d = trajectory.poses[:, :2]
#     poses_3d = np.concatenate([poses_2d, np.zeros((poses_2d.shape[0], 1))], axis=1)
#     poses = np.concatenate([np.array([[0, 0, 0]]), poses_3d])

#     pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(
#         poses.T,
#         camera.sensor2lidar_rotation,
#         camera.sensor2lidar_translation,
#         camera.intrinsics,
#         img_shape=camera.image.shape[:2]
#     )

#     image_height, image_width = camera.image.shape[:2]

#     # === 起点处理：保证轨迹从视野内或边界进入 ===
#     points_to_plot = []
#     first_point = pc_in_cam[1] if len(pc_in_cam) > 1 else None
#     second_point = pc_in_cam[2] if len(pc_in_cam) > 2 else None

#     if first_point is not None and pc_in_fov_mask[1]:
#         points_to_plot.append(first_point)
#     elif first_point is not None and second_point is not None:
#         intersection_point = _get_intersection_with_image_bottom_boundary(
#             first_point, second_point, image_width, image_height
#         )
#         if intersection_point is not None:
#             points_to_plot.append(intersection_point)

#     for i in range(2, len(pc_in_cam)):
#         if pc_in_fov_mask[i]:
#             points_to_plot.append(pc_in_cam[i])

#     valid_points = np.array(points_to_plot)

#     if len(valid_points) < 2:
#         return ax

#     # === spline 平滑轨迹 ===
#     x, y = valid_points[:, 0], valid_points[:, 1]
#     try:
#         tck, u = splprep([x, y], s=2)  # s 控制平滑程度
#         u_fine = np.linspace(0, 1, 200)
#         x_smooth, y_smooth = splev(u_fine, tck)
#     except Exception:
#         # 如果点数不足以 spline，就直接连线
#         x_smooth, y_smooth = x, y

#     # === 干净的线条（无 glow，无箭头） ===
#     ax.plot(x_smooth, y_smooth,
#             color=config["line_color"],
#             alpha=config["line_color_alpha"],
#             linewidth=config["line_width"],
#             linestyle=config["line_style"],
#             zorder=config["zorder"])

#     return ax



def _get_intersection_with_image_bottom_boundary(origin_point, end_point, image_width, image_height):
    """
    Backward-compatible wrapper retained for older callers.
    For trajectory rendering, the code now clips against the full image rectangle via
    `_clip_line_segment_to_image_rect`, which is more general than intersecting only the bottom edge.
    """
    clipped_segment = _clip_line_segment_to_image_rect(
        np.asarray(origin_point, dtype=np.float32),
        np.asarray(end_point, dtype=np.float32),
        image_width=image_width,
        image_height=image_height,
    )
    if clipped_segment is None:
        return None

    clipped_start, clipped_end = clipped_segment
    y_bottom = float(image_height - 1)
    for point in (clipped_start, clipped_end):
        if abs(float(point[1]) - y_bottom) <= 1e-4:
            return point
    return None
