# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Geometry utilities for robust, geometry-agnostic PINN room modeling.

Provides domain extraction, internal obstacle & column detection,
flow direction verification, and dynamic multi-window partitioning.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
import torch

from physicsnemo.mesh.io import from_pyvista


def extract_room_domain(volume_pv: pv.PolyData) -> pv.PolyData:
    """Extract primary room volume body, supporting both single-body and multi-body STL exports.

    For clean single-body CAD STLs, extracts the surface directly.
    For multi-body STLs, dynamically selects the largest enclosed body by bounding box volume.
    """
    bodies = volume_pv.split_bodies()
    if len(bodies) == 1:
        return bodies[0].extract_surface(algorithm="dataset_surface")

    # In multi-body exports, select the body with the largest bounding box volume
    def body_volume(b: pv.PolyData) -> float:
        bounds = b.bounds
        dx = max(0.0, bounds[1] - bounds[0])
        dy = max(0.0, bounds[3] - bounds[2])
        dz = max(0.0, bounds[5] - bounds[4])
        return dx * dy * dz

    primary_body = max(bodies, key=body_volume)
    return primary_body.extract_surface(algorithm="dataset_surface")


def split_walls_and_obstacles(
    walls_pv: pv.PolyData,
    geom_dir: Optional[str] = None,
) -> Tuple[pv.PolyData, List[pv.PolyData]]:
    """Identify outer wall boundary and all internal obstacle sub-bodies.

    In RoomVolume_Walls.stl:
      - Body 0 (or body with largest surface area/extent) is the outer wall shell.
      - Bodies 1..4 are the permanent cylindrical columns.
      - Bodies >= 5 are any future obstacles (desks, partitions, equipment) added in CAD.

    Also checks for optional separate external obstacle meshes (e.g., geometries/Obstacles.stl)
    and combines them into the internal obstacle list.

    Returns:
        Tuple of (combined_walls_surface, internal_obstacle_bodies).
    """
    walls_bodies = walls_pv.split_bodies()
    obstacle_bodies: List[pv.PolyData] = []

    if len(walls_bodies) > 1:
        # All sub-bodies beyond the outer shell are internal obstacles (columns, furniture, etc.)
        obstacle_bodies.extend(walls_bodies[1:])

    # Check for optional external obstacle files
    if geom_dir is not None:
        for ext_name in ["Obstacles.stl", "InternalObstacles.stl", "Furniture.stl"]:
            ext_path = os.path.join(geom_dir, ext_name)
            if os.path.exists(ext_path):
                ext_pv = pv.read(ext_path)
                ext_bodies = ext_pv.split_bodies()
                obstacle_bodies.extend(ext_bodies)

    return walls_pv, obstacle_bodies


def exclude_obstacle_volumes(
    interior_pts: np.ndarray,
    obstacle_bodies: List[pv.PolyData],
) -> Tuple[np.ndarray, int]:
    """Exclude solid interior collocation points inside all obstacle bodies (columns + furniture).

    For slender vertical cylinders (like the 4 columns), uses an exact cylindrical radial check.
    For arbitrary 3D closed shapes, uses polydata point enclosure.

    Returns:
        Tuple of (filtered_interior_pts, total_points_excluded).
    """
    if len(obstacle_bodies) == 0:
        return interior_pts, 0

    valid_pts = interior_pts
    total_excluded = 0

    for obs in obstacle_bodies:
        bx = obs.bounds[1] - obs.bounds[0]
        by = obs.bounds[3] - obs.bounds[2]
        bz = obs.bounds[5] - obs.bounds[4]

        # Detect vertical slender pillar/column geometry
        if bx < 1.0 and by < 1.0 and bz > 2.0:
            cx, cy, _ = obs.center
            cb = obs.bounds
            # Column radius with safety margin covering full cylinder
            radius = 0.285 if (0.55 <= max(bx, by) <= 0.58) else (max(bx, by) / 2.0 + 0.002)
            dist_sq = (valid_pts[:, 0] - cx) ** 2 + (valid_pts[:, 1] - cy) ** 2
            in_column = (
                (dist_sq <= radius**2)
                & (valid_pts[:, 2] >= cb[4] - 0.01)
                & (valid_pts[:, 2] <= cb[5] + 0.01)
            )
            col_count = int(np.sum(in_column))
            total_excluded += col_count
            valid_pts = valid_pts[~in_column]
        else:
            # Arbitrary 3D shape (e.g. desks, podiums, partitions)
            try:
                obs_surface = obs.extract_surface(algorithm="dataset_surface")
                cloud = pv.PolyData(valid_pts)
                enclosed = cloud.select_interior_points(obs_surface, check_surface=False)
                mask = enclosed["selected_points"].astype(bool)
                obs_count = int(np.sum(mask))
                total_excluded += obs_count
                valid_pts = valid_pts[~mask]
            except Exception:
                # Fallback: conservative bounding box exclusion
                in_box = (
                    (valid_pts[:, 0] >= obs.bounds[0])
                    & (valid_pts[:, 0] <= obs.bounds[1])
                    & (valid_pts[:, 1] >= obs.bounds[2])
                    & (valid_pts[:, 1] <= obs.bounds[3])
                    & (valid_pts[:, 2] >= obs.bounds[4])
                    & (valid_pts[:, 2] <= obs.bounds[5])
                )
                box_count = int(np.sum(in_box))
                total_excluded += box_count
                valid_pts = valid_pts[~in_box]

    return valid_pts, total_excluded


def verify_flow_direction(windows_pv: pv.PolyData, doors_pv: pv.PolyData) -> bool:
    """Verify that flow goes from Windows to Doors along the -Y axis.

    Confirms that windows are on the upstream side (higher Y) relative to doors (lower Y).
    """
    y_win_mean = (windows_pv.bounds[2] + windows_pv.bounds[3]) / 2.0
    y_door_mean = (doors_pv.bounds[2] + doors_pv.bounds[3]) / 2.0
    if y_win_mean <= y_door_mean:
        raise ValueError(
            f"Physical flow assumption violated: Windows mean Y ({y_win_mean:.2f} m) "
            f"is not upstream of Doors mean Y ({y_door_mean:.2f} m). Inflow towards -Y requires Y_win > Y_door."
        )
    return True


def decompose_windows(windows_pv: pv.PolyData) -> Tuple[List[pv.PolyData], int]:
    """Decompose Windows.stl into individual window bodies sorted along X (West -> East).

    Returns:
        Tuple of (sorted_window_bodies, num_windows).
    """
    window_bodies = sorted(windows_pv.split_bodies(), key=lambda b: b.center[0])
    return window_bodies, len(window_bodies)


def find_seated_breathing_center(
    volume_pv: pv.PolyData,
    breathing_height: float = 1.10,
) -> Tuple[float, float, float]:
    """Determine room center at seated human breathing plane (Z = 1.10 m)."""
    bounds = volume_pv.bounds
    cx = (bounds[1] + bounds[0]) / 2.0
    cy = (bounds[3] + bounds[2]) / 2.0
    return (float(cx), float(cy), float(breathing_height))
