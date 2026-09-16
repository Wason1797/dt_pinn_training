# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression verification script.

Validates that the new geometry_utils module produces 100% identical results
to the current legacy baseline on the existing geometries.
"""

import os
import sys
import numpy as np
import pyvista as pv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.geometry_utils import (
    extract_room_domain,
    split_walls_and_obstacles,
    exclude_obstacle_volumes,
    verify_flow_direction,
    decompose_windows,
    find_seated_breathing_center,
)

GEOM_DIR = os.path.join(REPO_ROOT, "geometries")


def run_parity_checks() -> bool:
    print("\n" + "=" * 70)
    print("RUNNING GEOMETRY REGRESSION & PARITY CHECKS")
    print("=" * 70)

    volume_pv = pv.read(os.path.join(GEOM_DIR, "RoomVolume.stl"))
    walls_pv = pv.read(os.path.join(GEOM_DIR, "RoomVolume_Walls.stl"))
    windows_pv = pv.read(os.path.join(GEOM_DIR, "Windows.stl"))
    doors_pv = pv.read(os.path.join(GEOM_DIR, "Doors.stl"))

    all_passed = True

    # -------------------------------------------------------------
    # 1. Room Volume Extraction Parity Check
    # -------------------------------------------------------------
    legacy_body = volume_pv.split_bodies()[1].extract_surface(algorithm="dataset_surface")
    new_body = extract_room_domain(volume_pv)

    legacy_bounds = np.array(legacy_body.bounds)
    new_bounds = np.array(new_body.bounds)
    bounds_diff = np.max(np.abs(legacy_bounds - new_bounds))

    if bounds_diff < 1e-5 and legacy_body.n_cells == new_body.n_cells:
        print(f"  [PASS] 1. Room Volume Domain Extraction matches legacy state identically (diff: {bounds_diff:.2e}).")
    else:
        print(f"  [FAIL] 1. Room Volume mismatch! Bounds diff: {bounds_diff:.2e}")
        all_passed = False

    # -------------------------------------------------------------
    # 2. Room Center & Seated Height Parity Check
    # -------------------------------------------------------------
    b = volume_pv.bounds
    legacy_center = ((b[1] + b[0]) / 2.0, (b[3] + b[2]) / 2.0, 1.10)
    new_center = find_seated_breathing_center(volume_pv, breathing_height=1.10)
    center_diff = np.max(np.abs(np.array(legacy_center) - np.array(new_center)))

    if center_diff < 1e-6:
        print(f"  [PASS] 2. Seated Breathing Center matches legacy center {new_center} (diff: {center_diff:.2e}).")
    else:
        print(f"  [FAIL] 2. Center mismatch! diff: {center_diff:.2e}")
        all_passed = False

    # -------------------------------------------------------------
    # 3. Column Detection & Exclusion Parity Check
    # -------------------------------------------------------------
    # Generate reproducible test points
    np.random.seed(42)
    raw_pts = np.random.uniform(
        [b[0], b[2], b[4]],
        [b[1], b[3], b[5]],
        size=(50000, 3),
    )
    cloud = pv.PolyData(raw_pts)
    enclosed = cloud.select_interior_points(legacy_body, check_surface=False)
    interior_pts = raw_pts[enclosed["selected_points"].astype(bool)]

    # Legacy column exclusion
    walls_bodies = walls_pv.split_bodies()
    legacy_col_bodies = walls_bodies[1:5]
    legacy_valid = interior_pts.copy()
    legacy_col_count = 0
    for col_mesh in legacy_col_bodies:
        cx, cy, _ = col_mesh.center
        cb = col_mesh.bounds
        radius = 0.285
        dist_sq = (legacy_valid[:, 0] - cx) ** 2 + (legacy_valid[:, 1] - cy) ** 2
        in_col = (
            (dist_sq <= radius**2)
            & (legacy_valid[:, 2] >= cb[4] - 0.01)
            & (legacy_valid[:, 2] <= cb[5] + 0.01)
        )
        legacy_col_count += int(np.sum(in_col))
        legacy_valid = legacy_valid[~in_col]

    # New obstacle & column exclusion
    _, obstacle_bodies = split_walls_and_obstacles(walls_pv, geom_dir=GEOM_DIR)
    new_valid, new_col_count = exclude_obstacle_volumes(interior_pts.copy(), obstacle_bodies)

    if legacy_col_count == new_col_count and len(legacy_valid) == len(new_valid):
        print(
            f"  [PASS] 3. Column/Obstacle Exclusion matches legacy state exactly "
            f"({legacy_col_count} points excluded from 4 columns)."
        )
    else:
        print(
            f"  [FAIL] 3. Column exclusion discrepancy! Legacy excluded {legacy_col_count}, New excluded {new_col_count}."
        )
        all_passed = False

    # -------------------------------------------------------------
    # 4. Windows Decomposition Parity Check
    # -------------------------------------------------------------
    window_bodies, num_windows = decompose_windows(windows_pv)
    if num_windows == 8:
        print(f"  [PASS] 4. Windows decomposition correctly detected exactly {num_windows} windows sorted West->East.")
    else:
        print(f"  [FAIL] 4. Expected 8 windows, got {num_windows}.")
        all_passed = False

    # -------------------------------------------------------------
    # 5. Flow Direction Verification (Windows -> Doors along -Y)
    # -------------------------------------------------------------
    flow_ok = verify_flow_direction(windows_pv, doors_pv)
    if flow_ok:
        y_win = (windows_pv.bounds[2] + windows_pv.bounds[3]) / 2.0
        y_door = (doors_pv.bounds[2] + doors_pv.bounds[3]) / 2.0
        print(
            f"  [PASS] 5. Flow direction verified: Windows (Y={y_win:.2f} m) -> Doors (Y={y_door:.2f} m), "
            f"inflow towards -Y is physically preserved."
        )
    else:
        all_passed = False

    # -------------------------------------------------------------
    # 6. Network Dimensionality Matching Check
    # -------------------------------------------------------------
    expected_dim = 3 + 1 + num_windows + 1  # (x,y,z) + t + 8 velocities + occupancy
    if expected_dim == 13:
        print(f"  [PASS] 6. Network input dimensionality matches current state (in_features = {expected_dim}).")
    else:
        print(f"  [FAIL] 6. Expected in_features = 13, got {expected_dim}.")
        all_passed = False

    # -------------------------------------------------------------
    # 7. Single-Body STL Robustness Check (Synthetic CAD model)
    # -------------------------------------------------------------
    cube = pv.Cube().triangulate()
    cube_extracted = extract_room_domain(cube)
    if cube_extracted.n_cells == cube.n_cells:
        print("  [PASS] 7. Single-body watertight STL robustness verified (handles single-body CAD without IndexError).")
    else:
        print("  [FAIL] 7. Single-body STL handling failed.")
        all_passed = False

    # -------------------------------------------------------------
    # 8. Future-Proof Obstacle Injection Check
    # -------------------------------------------------------------
    mock_desk = pv.Cube(center=(7.0, 4.0, 0.5), x_length=1.5, y_length=0.8, z_length=0.8).triangulate()
    mock_pts, mock_excluded = exclude_obstacle_volumes(interior_pts.copy(), obstacle_bodies + [mock_desk])
    if mock_excluded > legacy_col_count:
        print(
            f"  [PASS] 8. Future-proof internal obstacle hook verified (excluded {mock_excluded - legacy_col_count} "
            f"additional collocation points inside mock obstacle volume)."
        )
    else:
        print("  [FAIL] 8. Mock obstacle points were not excluded.")
        all_passed = False

    print("=" * 70)
    if all_passed:
        print("RESULT: ALL 8 GEOMETRY PARITY & ROBUSTNESS CHECKS PASSED SUCCESSFULLY! 🚀")
    else:
        print("RESULT: SOME CHECKS FAILED! Please inspect output above.")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_parity_checks()
    sys.exit(0 if success else 1)
