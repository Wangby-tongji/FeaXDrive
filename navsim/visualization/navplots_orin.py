from typing import Any, Callable, List, Tuple
import io

import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from navsim.evaluate.pdm_score import pdm_score

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Scene
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG, CAMERAS_PLOT_CONFIG
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.camera import (
    add_annotations_to_camera_ax,
    add_lidar_to_camera_ax,
    add_camera_ax,
    add_trajectory_to_camera_ax,
    apply_centered_camera_crop,
)


AGENT_TRAJECTORY_BLUE = "#E1343A"
GT_TRAJECTORY_CYAN = "#3B8516"


def _make_camera_trajectory_config(color: str) -> dict:
    """Shared camera-trajectory style with endpoint-aligned arrows."""
    # 修改点：根据颜色判断，如果是 GT，则图层置于最上层 (zorder=4)，否则置于底层 (zorder=3)
    layer_zorder = 3 if color == GT_TRAJECTORY_CYAN else 4
    
    return {
        "line_color": color,
        "line_color_alpha": 0.80,
        "line_width": 4.0,
        "line_style": "-",
        "marker": None,
        "zorder": layer_zorder,
        "line_capstyle": "round",
        "line_joinstyle": "round",
        "arrow_color": color,
        "arrow_edge_color": color,
        "arrow_alpha": 1.0,
        "arrow_line_width": 3.0,
        "arrow_length_px": 20.0,
        "arrow_tip_backoff_px": 0.0,
        "arrow_mutation_scale": 12.0,
    }


def _make_bev_trajectory_config(base_key: str, color: str) -> dict:
    cfg = dict(TRAJECTORY_CONFIG[base_key])
    cfg["line_color"] = color
    cfg["fill_color"] = color
    
    # 修改点：添加黑色轮廓并适当加粗边缘
    cfg["marker_edge_color"] = "black"
    cfg["marker_edge_width"] = 1.5 
    
    cfg["line_color_alpha"] = 0.96
    cfg["fill_color_alpha"] = 0.96
    cfg["line_width"] = max(float(cfg.get("line_width", 2.0)), 2.0)
    
    # 修改点：根据角色判断，如果是 Agent，则图层置于最上层 (zorder=4)，否则置于底层 (zorder=3)
    cfg["zorder"] = 4 if base_key == "agent" else 3
    
    return cfg


def _get_square_camera_crop_width(camera) -> float:
    cam_w, cam_h = _camera_image_wh(camera)
    return float(min(cam_w, cam_h))


def _apply_centered_camera_view(ax: plt.Axes, camera, trajectories) -> plt.Axes:
    crop_width = _get_square_camera_crop_width(camera)
    apply_centered_camera_crop(
        ax,
        camera,
        trajectories=trajectories,
        crop_width=crop_width,
        side_margin_px=42.0,
    )
    return ax


def _apply_full_figure_layout(fig: plt.Figure) -> plt.Figure:
    """
    Make subplots fill the whole figure area with minimal padding.
    Uses constrained-layout when available and keeps a tiny fallback margin.
    """
    try:
        fig.set_constrained_layout(True)
        try:
            fig.set_constrained_layout_pads(
                w_pad=0.0,
                h_pad=0.0,
                wspace=0.0,
                hspace=0.0,
            )
        except Exception:
            pass
    except Exception:
        pass

    try:
        fig.subplots_adjust(
            left=0.001,
            right=0.999,
            bottom=0.001,
            top=0.999,
            wspace=0.001,
            hspace=0.001,
        )
    except Exception:
        pass

    return fig

def _camera_image_wh(camera) -> Tuple[int, int]:
    """Return image width, height for a camera dataclass."""
    image = camera.image
    height, width = image.shape[:2]
    return int(width), int(height)



def _match_axes_box_to_image(ax: plt.Axes, width: float, height: float) -> plt.Axes:
    """Make the physical axes box follow the displayed image aspect ratio to avoid inner white padding."""
    if width <= 0 or height <= 0:
        return ax
    try:
        ax.set_box_aspect(float(height) / float(width))
    except Exception:
        ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.set_anchor("C")
    return ax


def _make_full_bleed_dual_view_figure(camera_width: int, camera_height: int, base_height: float = 8.0) -> Tuple[plt.Figure, Any]:
    """Create an equal-width BEV + camera layout. The camera is displayed as a square crop with the trajectory centered."""
    displayed_camera_width = float(min(camera_width, camera_height))
    panel_aspect = displayed_camera_width / max(float(camera_height), 1.0)
    fig_width = 2.0 * base_height * panel_aspect
    fig = plt.figure(figsize=(fig_width, base_height), constrained_layout=True)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.0],
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
        wspace=0.004,
    )
    return fig, gs


def configure_bev_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the plt ax object for birds-eye-view plots
    :param ax: matplotlib ax object
    :return: configured ax object
    """

    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_aspect("equal")

    # NOTE: x forward, y sideways
    ax.set_xlim(-margin_y / 2, margin_y / 2)
    ax.set_ylim(-margin_x / 2, margin_x / 2)

    # NOTE: left is y positive, right is y negative
    ax.invert_xaxis()

    return ax


def configure_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the ax object for general plotting
    :param ax: matplotlib ax object
    :return: ax object without a,y ticks
    """
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def configure_all_ax(ax: List[List[plt.Axes]]) -> List[List[plt.Axes]]:
    """
    Iterates through 2D ax list/array to apply configurations
    :param ax: 2D list/array of matplotlib ax object
    :return: configure axes
    """
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            configure_ax(ax[i][j])

    return ax


def plot_bev_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, plt.Axes]:
    """
    General plot for birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def plot_bev_with_agent(scene: Scene, scene_traj: Scene,agent: AbstractAgent) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots GT and agent trajectories in birds-eye-view visualization.
    """

    human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory_vis(scene_traj.get_agent_input())

    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, human_trajectory, _make_bev_trajectory_config("human", GT_TRAJECTORY_CYAN))
    add_trajectory_to_bev_ax(ax, agent_trajectory, _make_bev_trajectory_config("agent", AGENT_TRAJECTORY_BLUE))
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def plot_cameras_frame_with_agent(
    scene: Scene, 
    scene_traj: Scene, 
    frame_idx: int, 
    agent: AbstractAgent
) -> Tuple[plt.Figure, List[List[plt.Axes]], plt.Axes]:
    """
    Plots 8x cameras in 3x3 grid (left) and birds-eye-view with GT/agent trajectories (right).
    """
    frame = scene.frames[frame_idx]
    
    fig = plt.figure(figsize=(24, 12), constrained_layout=True)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[3, 1],
        left=0.001,
        right=0.999,
        bottom=0.001,
        top=0.999,
        wspace=0.001,
    )

    left_gs = gs[0].subgridspec(3, 3, wspace=0.001, hspace=0.001)
    left_ax = [[fig.add_subplot(left_gs[i, j]) for j in range(3)] for i in range(3)]
    
    add_camera_ax(left_ax[0][0], frame.cameras.cam_l0)
    add_camera_ax(left_ax[0][1], frame.cameras.cam_f0)
    add_camera_ax(left_ax[1][0], frame.cameras.cam_l1)
    add_camera_ax(left_ax[2][1], frame.cameras.cam_b0)
    add_camera_ax(left_ax[1][2], frame.cameras.cam_r1)
    add_camera_ax(left_ax[2][0], frame.cameras.cam_l2)
    left_ax[1][1].axis("off")
    add_camera_ax(left_ax[2][2], frame.cameras.cam_r2)

    for row in left_ax:
        for ax in row:
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

    right_ax = fig.add_subplot(gs[1])
    add_configured_bev_on_ax(right_ax, scene.map_api, frame)
    human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory_vis(scene_traj.get_agent_input())
    add_trajectory_to_bev_ax(right_ax, human_trajectory, _make_bev_trajectory_config("human", GT_TRAJECTORY_CYAN))
    add_trajectory_to_bev_ax(right_ax, agent_trajectory, _make_bev_trajectory_config("agent", AGENT_TRAJECTORY_BLUE))

    configure_bev_ax(right_ax)

    _apply_full_figure_layout(fig)

    return fig, left_ax, right_ax


def plot_bev_and_camera_with_agent(
    scene: Scene,
    scene_traj: Scene,
    frame_idx: int,
    agent: AbstractAgent
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    左侧 BEV、右侧前视相机。相机采用以轨迹为中心的方形裁剪，因此显示宽度与 BEV 面板一致。
    Agent 轨迹为蓝色，GT 轨迹为青色。
    """
    frame = scene.frames[frame_idx]

    cam_w, cam_h = _camera_image_wh(frame.cameras.cam_f0)
    displayed_cam_w = _get_square_camera_crop_width(frame.cameras.cam_f0)
    fig, gs = _make_full_bleed_dual_view_figure(cam_w, cam_h, base_height=8.0)

    bev_ax = fig.add_subplot(gs[0])
    add_configured_bev_on_ax(bev_ax, scene.map_api, frame)
    human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory_vis(scene_traj.get_agent_input())
    add_trajectory_to_bev_ax(bev_ax, human_trajectory, _make_bev_trajectory_config("human", GT_TRAJECTORY_CYAN))
    add_trajectory_to_bev_ax(bev_ax, agent_trajectory, _make_bev_trajectory_config("agent", AGENT_TRAJECTORY_BLUE))
    configure_bev_ax(bev_ax)
    try:
        bev_ax.set_box_aspect(1)
    except Exception:
        bev_ax.set_aspect("equal", adjustable="box", anchor="C")
    bev_ax.set_anchor("C")
    bev_ax.set_xticks([])
    bev_ax.set_yticks([])

    camera_ax = fig.add_subplot(gs[1])
    add_camera_ax(camera_ax, frame.cameras.cam_f0)
    add_trajectory_to_camera_ax(camera_ax, frame.cameras.cam_f0, human_trajectory, _make_camera_trajectory_config(GT_TRAJECTORY_CYAN))
    add_trajectory_to_camera_ax(camera_ax, frame.cameras.cam_f0, agent_trajectory, _make_camera_trajectory_config(AGENT_TRAJECTORY_BLUE))
    camera_ax.axis("off")
    camera_ax.set_xticks([])
    camera_ax.set_yticks([])
    _apply_centered_camera_view(camera_ax, frame.cameras.cam_f0, [human_trajectory, agent_trajectory])
    _match_axes_box_to_image(camera_ax, displayed_cam_w, cam_h)

    _apply_full_figure_layout(fig)

    return fig, bev_ax, camera_ax


def plot_traj_with_agent(
    scene: Scene,
    scene_traj: Scene,
    frame_idx: int,
    agent: AbstractAgent,
    metric_cache,
    simulator,
    scorer
) -> Tuple[plt.Figure, List[List[plt.Axes]], plt.Axes]:
    frame = scene.frames[frame_idx]

    cam_w, cam_h = _camera_image_wh(frame.cameras.cam_f0)
    displayed_cam_w = _get_square_camera_crop_width(frame.cameras.cam_f0)
    base_h = 6.0
    fig = plt.figure(figsize=(base_h * (displayed_cam_w / max(cam_h, 1.0)), base_h), constrained_layout=True)
    ax_f0 = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    add_camera_ax(ax_f0, frame.cameras.cam_f0)
    ax_f0.axis("off")
    ax_f0.set_xticks([])
    ax_f0.set_yticks([])

    human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory_vis(scene_traj.get_agent_input())
    add_trajectory_to_camera_ax(ax_f0, frame.cameras.cam_f0, human_trajectory, _make_camera_trajectory_config(GT_TRAJECTORY_CYAN))
    add_trajectory_to_camera_ax(ax_f0, frame.cameras.cam_f0, agent_trajectory, _make_camera_trajectory_config(AGENT_TRAJECTORY_BLUE))
    _apply_centered_camera_view(ax_f0, frame.cameras.cam_f0, [human_trajectory, agent_trajectory])
    _match_axes_box_to_image(ax_f0, displayed_cam_w, cam_h)

    pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=agent_trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer
    )

    _apply_full_figure_layout(fig)

    return fig, [[ax_f0]], None,pdm_result,agent_trajectory


def plot_cameras_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"], constrained_layout=True)

    add_camera_ax(ax[0, 0], frame.cameras.cam_l0)
    add_camera_ax(ax[0, 1], frame.cameras.cam_f0)
    add_camera_ax(ax[0, 2], frame.cameras.cam_r0)

    add_camera_ax(ax[1, 0], frame.cameras.cam_l1)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_camera_ax(ax[1, 2], frame.cameras.cam_r1)

    add_camera_ax(ax[2, 0], frame.cameras.cam_l2)
    add_camera_ax(ax[2, 1], frame.cameras.cam_b0)
    add_camera_ax(ax[2, 2], frame.cameras.cam_r2)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    _apply_full_figure_layout(fig)

    return fig, ax


def plot_cameras_frame_with_lidar(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the lidar pc) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"], constrained_layout=True)

    add_lidar_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.lidar)

    add_lidar_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.lidar)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_lidar_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.lidar)

    add_lidar_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.lidar)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    _apply_full_figure_layout(fig)

    return fig, ax


# def plot_cameras_frame_with_annotations(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
#     """
#     Plots 8x cameras (including the bounding boxes) and birds-eye-view visualization in 3x3 grid
#     :param scene: navsim scene dataclass
#     :param frame_idx: index of selected frame
#     :return: figure and ax object of matplotlib
#     """

#     frame = scene.frames[frame_idx]
#     fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

#     add_annotations_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.annotations)
#     add_annotations_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.annotations)
#     add_annotations_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.annotations)

#     add_annotations_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.annotations)
#     add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
#     add_annotations_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.annotations)

#     add_annotations_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.annotations)
#     add_annotations_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.annotations)
#     add_annotations_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.annotations)

#     configure_all_ax(ax)
#     configure_bev_ax(ax[1, 1])
#     fig.tight_layout()
#     fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

#     return fig, ax

def plot_cameras_frame_with_annotations(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots only the cam_f0 camera image with annotations.
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """
    frame = scene.frames[frame_idx]

    # 创建单一子图
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)  # 可以根据实际需求调整figsize

    # 添加cam_f0的图像和标注
    add_annotations_to_camera_ax(ax, frame.cameras.cam_f0, frame.annotations)

    # 可选：配置坐标轴（去掉刻度、边框等）
    ax.axis("off")

    _apply_full_figure_layout(fig)

    return fig, ax

def frame_plot_to_pil(
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
) -> List[Image.Image]:
    """
    Plots a frame according to plotting function and return a list of PIL images
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices to save
    :return: list of PIL images
    """

    images: List[Image.Image] = []

    for frame_idx in tqdm(frame_indices, desc="Rendering frames"):
        fig, ax = callable_frame_plot(scene, frame_idx)

        # Creating PIL image from fig
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        images.append(Image.open(buf).copy())

        # close buffer and figure
        buf.close()
        plt.close(fig)

    return images


def frame_plot_to_gif(
    file_name: str,
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
    duration: float = 500,
) -> None:
    """
    Saves a frame-wise plotting function as GIF (hard G)
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices
    :param file_name: file path for saving to save
    :param duration: frame interval in ms, defaults to 500
    """
    images = frame_plot_to_pil(callable_frame_plot, scene, frame_indices)
    images[0].save(file_name, save_all=True, append_images=images[1:], duration=duration, loop=0)