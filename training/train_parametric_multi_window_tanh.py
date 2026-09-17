# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""13D Parametric 3D Room Navier-Stokes + CO2 PINN training with a tunable architecture.

Same physics, geometry, sampling and exports as train_parametric_multi_window.py, but with:
  - a configurable network: arch.num_layers / arch.layer_size / arch.activation
    (defaults: 5 hidden layers x 128, Tanh - the activation from the AIQ notebook)
  - a FLAT learning rate: this script constructs no lr_scheduler at all
  - per-term loss weights (phy / walls / windows / doors / ic)
  - both the weighted total AND the raw unweighted per-term losses logged, so runs with
    different weights stay directly comparable to the unweighted baseline

Trains a 13D surrogate neural network mapping:
  (x, y, z, t, V_1, ..., V_N, N_people) -> (u, v, w, p, c)
Temperature is deliberately out of scope, so out_features stays at 5.

All configuration lives in config_multi_window_tanh.yaml at the repo root.
"""

import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
GEOM_DIR = os.path.join(REPO_ROOT, "geometries")

import hydra
import numpy as np
import pyvista as pv
import torch
from omegaconf import DictConfig
from torch.optim import Adam

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.sampling import sample_random_points_on_cells
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.nn import get_activation
from physicsnemo.utils.logging import PythonLogger
from training.checkpoint_utils import save_training_checkpoint
from training.geometry_utils import (
    extract_room_domain,
    split_walls_and_obstacles,
    exclude_obstacle_volumes,
    verify_flow_direction,
    decompose_windows,
    find_seated_breathing_center,
)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable HH:MM:SS or MM:SS string."""
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def write_pvd_file(pvd_path: str, entries: list) -> None:
    """Write a ParaView Data (PVD) collection referencing VTU time slices."""
    root = ET.Element("VTKFile", type="Collection", version="0.1")
    collection = ET.SubElement(root, "Collection")
    for timestep, filepath in entries:
        ET.SubElement(collection, "DataSet", timestep=str(timestep), file=filepath)
    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(os.path.abspath(pvd_path)), exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(pvd_path, xml_declaration=True, encoding="utf-8")


def load_checkpoint_local(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
    resume: bool = False,
    expected_arch: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Hoisted copy of checkpoint_utils.load_checkpoint_for_training, with two changes.

    1. All scheduler handling is dropped - this trainer has no scheduler, so there is no
       scheduler state to restore and no legacy lr_lambdas replay branch to worry about.
    2. Architecture validation is extended past in_features/out_features to cover
       num_layers, layer_size, activation, skip_connections and weight_norm. The upstream
       helper checks only the first two, which means a depth/width mismatch surfaces as a
       raw PyTorch shape error, and an activation mismatch is never caught at all: Tanh
       and SiLU are parameterless, so the state_dict keys are identical and
       load_state_dict succeeds silently while every predicted field is wrong.

    Missing keys warn rather than fail, so checkpoints written before a key existed
    still load.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        model: PyTorch model to load weights into.
        optimizer: PyTorch optimizer (required if resume=True).
        device: Target torch.device.
        resume: If True, restores optimizer state and returns iteration + 1.
                If False, only loads model weights and returns 0.
        expected_arch: Mapping of model_config keys to the values configured for this run.

    Returns:
        Tuple of (start_iter, metadata_dict).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    metadata: Dict[str, Any] = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        # Bare state_dict
        state_dict = checkpoint

    # Validate the full architecture, not just in/out features
    model_config = metadata.get("model_config")
    if expected_arch:
        if model_config and isinstance(model_config, dict):
            for key, want in expected_arch.items():
                got = model_config.get(key)
                if got is None:
                    print(f"  WARNING: checkpoint has no model_config['{key}']; assuming {want!r}.")
                elif got != want:
                    raise ValueError(
                        f"Checkpoint {key} ({got!r}) does not match the configured {key} ({want!r}). "
                        f"Re-run with that value, or start a fresh run."
                    )
        else:
            print("  WARNING: checkpoint has no model_config; skipping architecture validation.")

    # Load model parameters
    model.load_state_dict(state_dict)
    print("Successfully loaded model weights.")

    if not resume:
        # Pretrained initialization only (warm start)
        print("Warm-start mode: Starting fresh training run from iteration 0.")
        return 0, metadata

    # Resume training: restore optimizer state (there is no scheduler in this trainer)
    ckpt_iter = int(metadata.get("iteration", 0))
    start_iter = ckpt_iter + 1

    if optimizer is not None and "optimizer_state_dict" in metadata:
        optimizer.load_state_dict(metadata["optimizer_state_dict"])
        # Ensure optimizer state tensors are on the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"Restored optimizer state from checkpoint (checkpoint iteration: {ckpt_iter}).")

    print(f"Resuming training starting at iteration {start_iter}.")
    return start_iter, metadata


def compute_unsteady_pde_residuals_occupancy(
    coords: torch.Tensor,
    t: torch.Tensor,
    n_people: torch.Tensor,
    out: torch.Tensor,
    nu: float = 0.01,
    rho: float = 1.0,
    D: float = 0.005,
    center: Tuple[float, float, float] = (7.79, 4.57, 1.10),
    emission_per_person: float = 1.15e-4,  # g/(m^3*s) per human (0.010 g/s integrated over room)
    sigma: float = 2.5,
) -> Dict[str, torch.Tensor]:
    """Compute unsteady 3D Navier-Stokes and occupancy-dependent CO2 transport residuals via autodiff."""
    u = out[:, 0:1]
    v = out[:, 1:2]
    w = out[:, 2:3]
    p = out[:, 3:4]
    c = out[:, 4:5]

    ones = torch.ones_like(u)

    # 1. Temporal derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=ones, create_graph=True)[0]
    w_t = torch.autograd.grad(w, t, grad_outputs=ones, create_graph=True)[0]
    c_t = torch.autograd.grad(c, t, grad_outputs=ones, create_graph=True)[0]

    # 2. First spatial derivatives
    grad_u = torch.autograd.grad(u, coords, grad_outputs=ones, create_graph=True)[0]
    grad_v = torch.autograd.grad(v, coords, grad_outputs=ones, create_graph=True)[0]
    grad_w = torch.autograd.grad(w, coords, grad_outputs=ones, create_graph=True)[0]
    grad_p = torch.autograd.grad(p, coords, grad_outputs=ones, create_graph=True)[0]
    grad_c = torch.autograd.grad(c, coords, grad_outputs=ones, create_graph=True)[0]

    u_x, u_y, u_z = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
    v_x, v_y, v_z = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
    w_x, w_y, w_z = grad_w[:, 0:1], grad_w[:, 1:2], grad_w[:, 2:3]
    p_x, p_y, p_z = grad_p[:, 0:1], grad_p[:, 1:2], grad_p[:, 2:3]
    c_x, c_y, c_z = grad_c[:, 0:1], grad_c[:, 1:2], grad_c[:, 2:3]

    # 3. Second spatial derivatives (Laplacians)
    u_xx = torch.autograd.grad(u_x, coords, grad_outputs=ones, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, coords, grad_outputs=ones, create_graph=True)[0][:, 1:2]
    u_zz = torch.autograd.grad(u_z, coords, grad_outputs=ones, create_graph=True)[0][:, 2:3]
    laplace_u = u_xx + u_yy + u_zz

    v_xx = torch.autograd.grad(v_x, coords, grad_outputs=ones, create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, coords, grad_outputs=ones, create_graph=True)[0][:, 1:2]
    v_zz = torch.autograd.grad(v_z, coords, grad_outputs=ones, create_graph=True)[0][:, 2:3]
    laplace_v = v_xx + v_yy + v_zz

    w_xx = torch.autograd.grad(w_x, coords, grad_outputs=ones, create_graph=True)[0][:, 0:1]
    w_yy = torch.autograd.grad(w_y, coords, grad_outputs=ones, create_graph=True)[0][:, 1:2]
    w_zz = torch.autograd.grad(w_z, coords, grad_outputs=ones, create_graph=True)[0][:, 2:3]
    laplace_w = w_xx + w_yy + w_zz

    c_xx = torch.autograd.grad(c_x, coords, grad_outputs=ones, create_graph=True)[0][:, 0:1]
    c_yy = torch.autograd.grad(c_y, coords, grad_outputs=ones, create_graph=True)[0][:, 1:2]
    c_zz = torch.autograd.grad(c_z, coords, grad_outputs=ones, create_graph=True)[0][:, 2:3]
    laplace_c = c_xx + c_yy + c_zz

    # 4. Residual definitions
    res_continuity = u_x + v_y + w_z
    res_mom_x = u_t + u * u_x + v * u_y + w * u_z + (1.0 / rho) * p_x - nu * laplace_u
    res_mom_y = v_t + u * v_x + v * v_y + w * v_z + (1.0 / rho) * p_y - nu * laplace_v
    res_mom_z = w_t + u * w_x + v * w_y + w * w_z + (1.0 / rho) * p_z - nu * laplace_w

    # Dynamic CO2 source term scaling linearly with N_people
    x0, y0, z0 = center
    dist_sq = (coords[:, 0:1] - x0) ** 2 + (coords[:, 1:2] - y0) ** 2 + (coords[:, 2:3] - z0) ** 2
    gaussian_shape = torch.exp(-dist_sq / (sigma**2))
    source = (n_people * emission_per_person) * gaussian_shape
    res_transport = c_t + u * c_x + v * c_y + w * c_z - D * laplace_c - source

    return {
        "continuity": res_continuity,
        "momentum_x": res_mom_x,
        "momentum_y": res_mom_y,
        "momentum_z": res_mom_z,
        "transport": res_transport,
    }


@hydra.main(version_base="1.3", config_path="../", config_name="config_multi_window_tanh.yaml")
def room_trainer_multi_window_tanh(cfg: DictConfig) -> None:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Accelerating neural network training with Apple Silicon (MPS) 🚀")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Warning: Running on CPU.")

    # ---------------------------------------------------------------- architecture
    arch_activation = str(cfg.arch.activation).lower()
    try:
        # Fail fast, before any STL I/O, with the full list of valid options
        get_activation(arch_activation)
    except KeyError as exc:
        raise ValueError(f"Unknown arch.activation={arch_activation!r}. {exc}") from exc

    arch_num_layers = int(cfg.arch.num_layers)
    arch_layer_size = int(cfg.arch.layer_size)
    skip_connections = bool(cfg.arch.skip_connections)
    weight_norm = bool(cfg.arch.weight_norm)

    # ------------------------------------------------------------------- optimizer
    lr = float(cfg.optimizer.lr)

    # ----------------------------------------------------------------- loss weights
    w_phy = float(cfg.loss_weights.phy)
    w_walls = float(cfg.loss_weights.walls)
    w_windows = float(cfg.loss_weights.windows)
    w_doors = float(cfg.loss_weights.doors)
    w_ic = float(cfg.loss_weights.ic)

    # ----------------------------------------------------------------- point counts
    pts_int = int(cfg.points.interior)
    pts_walls = int(cfg.points.walls)
    pts_win = int(cfg.points.windows_per_window)
    pts_doors = int(cfg.points.doors)
    pts_ic = int(cfg.points.ic)

    # -------------------------------------------------------------- physics / ranges
    t_max = float(cfg.physics.t_max)
    v_min = float(cfg.physics.v_min)
    v_max = float(cfg.physics.v_max)
    n_people_min = float(cfg.physics.n_people_min)
    n_people_max = float(cfg.physics.n_people_max)
    tau_ramp = float(cfg.physics.tau_ramp)
    nu = float(cfg.physics.nu)
    rho = float(cfg.physics.rho)
    diffusivity = float(cfg.physics.diffusivity)
    emission_pp = float(cfg.physics.emission_per_person)
    sigma = float(cfg.physics.sigma)
    breathing_height = float(cfg.physics.breathing_height)

    # -------------------------------------------------------------------- training
    total_iters = int(cfg.max_iters)
    log_every = int(cfg.log_every)
    snapshot_every = int(cfg.snapshot_every)
    snapshot_resolution = int(cfg.snapshot_resolution)
    log_csv = bool(cfg.log_csv)

    date_str = datetime.now().strftime("%Y-%m-%d")
    subdir = cfg.get("output_subdir") or (
        f"parametric_multi_window_{arch_activation}_L{arch_num_layers}_W{arch_layer_size}"
    )
    output_dir = os.path.join(REPO_ROOT, "outputs", date_str, subdir)
    os.makedirs(output_dir, exist_ok=True)

    log = PythonLogger(name=f"room_13d_mw_{arch_activation}_L{arch_num_layers}_W{arch_layer_size}")
    # Explicit path is REQUIRED here: PythonLogger.file_logging() defaults to ./launch.log
    # in the cwd AND os.removes the existing file first, which would wipe the baseline
    # trainer's log at the repo root.
    log.file_logging(file_name=os.path.join(output_dir, "launch.log"))

    log.info("Loading STL geometries from geometries/ ...")
    volume_pv = pv.read(os.path.join(GEOM_DIR, "RoomVolume.stl"))
    walls_pv = pv.read(os.path.join(GEOM_DIR, "RoomVolume_Walls.stl"))
    windows_pv = pv.read(os.path.join(GEOM_DIR, "Windows.stl"))
    doors_pv = pv.read(os.path.join(GEOM_DIR, "Doors.stl"))

    # Verify physical flow direction: Windows -> Doors along -Y
    verify_flow_direction(windows_pv, doors_pv)

    # Decompose windows and detect window count
    window_bodies, num_windows = decompose_windows(windows_pv)
    log.info(f"Loaded {num_windows} individual window bodies from Windows.stl (sorted West->East along X):")
    mesh_windows = []
    windows_areas = []
    for w_idx, body in enumerate(window_bodies, start=1):
        log.info(
            f"  Window {w_idx}: X ∈ [{body.bounds[0]:.2f}, {body.bounds[1]:.2f}], "
            f"Center=({body.center[0]:.2f}, {body.center[1]:.2f}, {body.center[2]:.2f}), Area={body.area:.2f} m²"
        )
        mesh_windows.append(from_pyvista(body))
        windows_areas.append(torch.tensor(body.compute_cell_sizes().cell_data["Area"], dtype=torch.float32))

    # Identify outer walls and any internal obstacle bodies (columns + furniture)
    walls_pv, obstacle_bodies = split_walls_and_obstacles(walls_pv, geom_dir=GEOM_DIR)
    mesh_walls = from_pyvista(walls_pv)
    mesh_doors = from_pyvista(doors_pv)

    bounds = volume_pv.bounds
    center = find_seated_breathing_center(volume_pv, breathing_height=breathing_height)

    walls_areas = torch.tensor(walls_pv.compute_cell_sizes().cell_data["Area"], dtype=torch.float32)
    doors_areas = torch.tensor(doors_pv.compute_cell_sizes().cell_data["Area"], dtype=torch.float32)

    def sample_surface_13d(surface_mesh, areas, n_points, current_device):
        """Sample spatial boundary points with random t, multi-window velocities, and N_people."""
        cell_indices = torch.multinomial(areas, n_points, replacement=True).to(current_device)
        pts = sample_random_points_on_cells(surface_mesh, cell_indices).to(
            device=current_device, dtype=torch.float32
        )
        t = torch.rand(n_points, 1, device=current_device, dtype=torch.float32) * t_max
        v_param = v_min + torch.rand(n_points, num_windows, device=current_device, dtype=torch.float32) * (v_max - v_min)
        n_param = n_people_min + torch.rand(n_points, 1, device=current_device, dtype=torch.float32) * (n_people_max - n_people_min)
        return torch.cat([pts, t, v_param, n_param], dim=1)

    def sample_windows_13d(n_points_per_window: int = pts_win):
        """Sample boundary points across all windows and compute corresponding target velocities towards doors (-Y)."""
        inps: List[torch.Tensor] = []
        v_targets: List[torch.Tensor] = []

        for k in range(num_windows):
            mesh_k = mesh_windows[k]
            areas_k = windows_areas[k]

            cell_indices = torch.multinomial(areas_k, n_points_per_window, replacement=True).to(device)
            pts_k = sample_random_points_on_cells(mesh_k, cell_indices).to(device=device, dtype=torch.float32)

            t_k = torch.rand(n_points_per_window, 1, device=device, dtype=torch.float32) * t_max
            v_param_k = v_min + torch.rand(n_points_per_window, num_windows, device=device, dtype=torch.float32) * (v_max - v_min)
            n_param_k = n_people_min + torch.rand(n_points_per_window, 1, device=device, dtype=torch.float32) * (n_people_max - n_people_min)

            # Target velocity for Window k: inflow towards negative Y (doors) governed by its specific velocity V_k
            v_k = v_param_k[:, k : k + 1]
            v_target_k = -v_k * torch.tanh(3.0 * t_k / tau_ramp)

            inp_k = torch.cat([pts_k, t_k, v_param_k, n_param_k], dim=1)
            inps.append(inp_k)
            v_targets.append(v_target_k)

        return torch.cat(inps, dim=0), torch.cat(v_targets, dim=0)

    raw_pts = np.random.uniform(
        [bounds[0], bounds[2], bounds[4]],
        [bounds[1], bounds[3], bounds[5]],
        size=(200000, 3),
    )
    cloud = pv.PolyData(raw_pts)
    watertight_domain = extract_room_domain(volume_pv)
    enclosed = cloud.select_interior_points(watertight_domain, check_surface=False)
    mask = enclosed["selected_points"].astype(bool)
    valid_interior_pts = raw_pts[mask]

    # Exclude interior columns and any internal obstacles from fluid domain
    valid_interior_pts, col_excluded_count = exclude_obstacle_volumes(valid_interior_pts, obstacle_bodies)
    log.info(
        f"Excluded {col_excluded_count:,} points from inside {len(obstacle_bodies)} internal obstacles/columns."
    )
    interior_pool = torch.tensor(valid_interior_pts, dtype=torch.float32, device=device)
    log.info(f"Interior point pool ready: {len(interior_pool):,} points inside watertight room (obstacles excluded).")

    def sample_interior_13d(n_points):
        """Sample (x, y, z), t, window velocities, and N_people."""
        idx = torch.randint(0, len(interior_pool), (n_points,), device=device)
        coords = interior_pool[idx].clone().requires_grad_(True)
        t = (torch.rand(n_points, 1, device=device, dtype=torch.float32) * t_max).requires_grad_(True)
        v_param = v_min + torch.rand(n_points, num_windows, device=device, dtype=torch.float32) * (v_max - v_min)
        n_param = n_people_min + torch.rand(n_points, 1, device=device, dtype=torch.float32) * (n_people_max - n_people_min)
        return coords, t, v_param, n_param

    def sample_initial_condition_13d(n_points):
        """Sample points at t = 0 with random window velocities and N_people."""
        idx = torch.randint(0, len(interior_pool), (n_points,), device=device)
        coords = interior_pool[idx].clone()
        t_zero = torch.zeros(n_points, 1, device=device, dtype=torch.float32)
        v_param = v_min + torch.rand(n_points, num_windows, device=device, dtype=torch.float32) * (v_max - v_min)
        n_param = n_people_min + torch.rand(n_points, 1, device=device, dtype=torch.float32) * (n_people_max - n_people_min)
        return torch.cat([coords, t_zero, v_param, n_param], dim=1)

    # Dynamic Multi-Window Model: (x, y, z, t, V_1, ..., V_N, N_people) -> (u, v, w, p, c)
    in_features = 3 + 1 + num_windows + 1
    out_features = 5
    model = FullyConnected(
        in_features=in_features,
        out_features=out_features,
        num_layers=arch_num_layers,        # HIDDEN layers; a separate linear head is added on top
        layer_size=arch_layer_size,
        activation_fn=arch_activation,     # bare lowercase string -> physicsnemo.nn.get_activation
        skip_connections=skip_connections,
        weight_norm=weight_norm,
    ).to(device)

    log.info(
        f"Architecture: FullyConnected({in_features} -> {arch_num_layers} x {arch_layer_size} "
        f"[{arch_activation}] -> {out_features}) | skip_connections={skip_connections}, "
        f"weight_norm={weight_norm} | params={sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = Adam(model.parameters(), lr=lr)
    # Deliberately NO lr_scheduler: the learning rate is flat for the whole run. Saves use
    # the imported save_training_checkpoint with scheduler=None (it stores
    # scheduler_state_dict: None); loads use load_checkpoint_local, which has no scheduler
    # code path at all.

    expected_arch = {
        "in_features": in_features,
        "out_features": out_features,
        "num_layers": arch_num_layers,
        "layer_size": arch_layer_size,
        "activation": arch_activation,
        "skip_connections": skip_connections,
        "weight_norm": weight_norm,
    }

    def build_extra_meta() -> Dict[str, Any]:
        """Checkpoint metadata. Plain Python types only - never pickle a DictConfig, or every
        future loader of this .pth would depend on omegaconf."""
        return {
            "bounds": tuple(float(b) for b in bounds),
            "center": tuple(float(c) for c in center),
            "t_max": t_max,
            "v_min": v_min,
            "v_max": v_max,
            "num_windows": num_windows,
            "window_centers": [[float(x) for x in b.center] for b in window_bodies],
            "window_bounds": [[float(x) for x in b.bounds] for b in window_bodies],
            "n_people_min": n_people_min,
            "n_people_max": n_people_max,
            "tau_ramp": tau_ramp,
            "flat_lr": lr,
            "loss_weights": {
                "phy": w_phy, "walls": w_walls, "windows": w_windows,
                "doors": w_doors, "ic": w_ic,
            },
            "point_counts": {
                "interior": pts_int, "walls": pts_walls,
                "windows_per_window": pts_win, "doors": pts_doors, "ic": pts_ic,
            },
            "model_config": dict(expected_arch),
        }

    # Checkpoint loading / Resumption
    resume_path = cfg.get("resume")
    checkpoint_path = cfg.get("checkpoint")

    start_iter = 0
    if resume_path:
        start_iter, _ = load_checkpoint_local(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            resume=True,
            expected_arch=expected_arch,
        )
        # Adam's optimizer_state_dict carries its own lr, so re-assert the configured flat
        # value: otherwise resuming a run trained at a different lr would silently keep it.
        for group in optimizer.param_groups:
            group["lr"] = lr
        log.info(f"Flat LR re-asserted to {lr:.2e} after resume (no scheduler).")
    elif checkpoint_path:
        _, _ = load_checkpoint_local(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
            resume=False,
            expected_arch=expected_arch,
        )

    if start_iter >= total_iters:
        log.warning(
            f"Start iteration ({start_iter}) is >= max_iters ({total_iters}). "
            f"No training needed. Increase max_iters if you wish to continue further."
        )
        return

    csv_path = os.path.join(output_dir, "loss_history.csv")
    if log_csv and not os.path.exists(csv_path):
        with open(csv_path, "w") as fh:
            fh.write("iter,weighted_total,raw_sum,raw_phy,raw_ic,raw_walls,raw_windows,raw_doors,lr\n")

    log.info(
        f"Starting 13D Multi-Window Parametric PINN training from iteration {start_iter} to {total_iters:,} "
        f"(t ∈ [0, {t_max:.1f}]s, V_k ∈ [{v_min:.1f}, {v_max:.1f}] m/s across {num_windows} windows, "
        f"N_people ∈ [{n_people_min:.0f}, {n_people_max:.0f}])..."
    )
    log.info(
        f"Loss weights: phy={w_phy:g}, walls={w_walls:g}, windows={w_windows:g}, "
        f"doors={w_doors:g}, ic={w_ic:g} | flat LR={lr:.2e} (no scheduler)"
    )
    log.info(
        f"Points per iteration: interior={pts_int}, walls={pts_walls}, "
        f"windows={pts_win}x{num_windows}, doors={pts_doors}, ic={pts_ic}"
    )
    start_time = time.time()
    last_log_time = start_time
    last_log_iter = start_iter

    for i in range(start_iter, total_iters):
        optimizer.zero_grad()

        inp_walls = sample_surface_13d(mesh_walls, walls_areas, pts_walls, device)
        inp_windows, target_windows_v = sample_windows_13d(n_points_per_window=pts_win)
        inp_doors = sample_surface_13d(mesh_doors, doors_areas, pts_doors, device)
        inp_ic = sample_initial_condition_13d(pts_ic)
        coords_int, t_int, v_int, n_int = sample_interior_13d(pts_int)

        out_walls = model(inp_walls)
        out_windows = model(inp_windows)
        out_doors = model(inp_doors)
        out_ic = model(inp_ic)
        out_interior = model(torch.cat([coords_int, t_int, v_int, n_int], dim=1))

        # 1. IC at t = 0 (fluid at rest, zero concentration)
        loss_ic = torch.mean(out_ic[:, 0:5] ** 2)

        # 2. Walls no slip (u=0, v=0, w=0)
        loss_walls = torch.mean(out_walls[:, 0:3] ** 2)

        # 3. Multi-window inflow: u=0, w=0, c=0, and v matches window-specific inflow target
        loss_windows_u = torch.mean(out_windows[:, 0:1] ** 2)
        loss_windows_v = torch.mean((out_windows[:, 1:2] - target_windows_v) ** 2)
        loss_windows_w = torch.mean(out_windows[:, 2:3] ** 2)
        loss_windows_c = torch.mean(out_windows[:, 4:5] ** 2)
        loss_windows = loss_windows_u + loss_windows_v + loss_windows_w + loss_windows_c

        # 4. Doors outlet (p=0)
        loss_doors = torch.mean(out_doors[:, 3] ** 2)

        # 5. Unsteady PDE with dynamic CO2 source scaling with N_people
        res_dict = compute_unsteady_pde_residuals_occupancy(
            coords=coords_int,
            t=t_int,
            n_people=n_int,
            out=out_interior,
            nu=nu,
            rho=rho,
            D=diffusivity,
            center=center,
            emission_per_person=emission_pp,
            sigma=sigma,
        )
        loss_phy = (
            torch.mean(res_dict["continuity"] ** 2)
            + torch.mean(res_dict["momentum_x"] ** 2)
            + torch.mean(res_dict["momentum_y"] ** 2)
            + torch.mean(res_dict["momentum_z"] ** 2)
            + torch.mean(res_dict["transport"] ** 2)
        )

        # Weighted objective: the only thing that is optimized. The per-term formulas above
        # are byte-identical to the baseline trainer's, so the raw values logged below stay
        # directly comparable across weightings.
        total_loss = (
            w_phy * loss_phy
            + w_walls * loss_walls
            + w_windows * loss_windows
            + w_doors * loss_doors
            + w_ic * loss_ic
        )
        total_loss.backward()
        optimizer.step()
        # (no scheduler.step())

        do_short = (i % log_every == 0 and i > 0 and i % snapshot_every != 0)
        do_block = (i % snapshot_every == 0)

        if do_short or do_block:
            # .item() forces a device sync on MPS, so only fetch inside the logging branches
            with torch.no_grad():
                r_phy = loss_phy.item()
                r_walls = loss_walls.item()
                r_win = loss_windows.item()
                r_doors = loss_doors.item()
                r_ic = loss_ic.item()
            raw_sum = r_phy + r_walls + r_win + r_doors + r_ic  # plain floats, no second sync
            w_total = total_loss.item()
            current_lr = optimizer.param_groups[0]["lr"]

            if log_csv:
                with open(csv_path, "a") as fh:
                    fh.write(
                        f"{i},{w_total:.8f},{raw_sum:.8f},{r_phy:.8f},{r_ic:.8f},"
                        f"{r_walls:.8f},{r_win:.8f},{r_doors:.8f},{current_lr:.8e}\n"
                    )

        if do_short:
            now = time.time()
            elapsed = now - start_time
            recent_elapsed = now - last_log_time
            recent_iters = i - last_log_iter
            speed = recent_iters / recent_elapsed if recent_elapsed > 0 else 0.0
            sec_per_it = 1.0 / speed if speed > 0 else 0.0
            eta_seconds = (total_iters - i) / speed if speed > 0 else 0.0
            pct = (i / total_iters) * 100.0
            log.info(
                f"[Iter: {i:05d}/{total_iters} ({pct:4.1f}%)] | "
                f"WLoss: {w_total:.5f} | RawSum: {raw_sum:.5f} | "
                f"raw Phy={r_phy:.5f}, IC={r_ic:.5f}, Walls={r_walls:.5f}, "
                f"Win={r_win:.5f}, Doors={r_doors:.5f} | "
                f"Speed: {sec_per_it:.2f} s/it ({speed:4.2f} it/s) | "
                f"Elapsed: {format_duration(elapsed)} | "
                f"ETA: {format_duration(eta_seconds)}"
            )

        if do_block:
            now = time.time()
            elapsed = now - start_time
            pct = (i / total_iters) * 100.0
            avg_speed = i / elapsed if (i > 0 and elapsed > 0) else 0.0
            eta_seconds = (total_iters - i) / avg_speed if avg_speed > 0 else 0.0
            eta_str = format_duration(eta_seconds) if i > 0 else "estimating..."

            log.info(
                f"\n{'='*72}\n"
                f"[Iter: {i:05d}/{total_iters} ({pct:4.1f}%)] "
                f"Weighted Total: {w_total:.5f} | Unweighted Sum: {raw_sum:.5f}\n"
                f"  Raw (unweighted, comparable to the baseline trainer):\n"
                f"    Phy={r_phy:.5f}, IC={r_ic:.5f}, Walls={r_walls:.5f}, "
                f"Win={r_win:.5f}, Doors={r_doors:.5f}\n"
                f"  Weighted contributions (weight x raw):\n"
                f"    Phy={w_phy*r_phy:.5f} (w={w_phy:g}), IC={w_ic*r_ic:.5f} (w={w_ic:g}), "
                f"Walls={w_walls*r_walls:.5f} (w={w_walls:g}), "
                f"Win={w_windows*r_win:.5f} (w={w_windows:g}), "
                f"Doors={w_doors*r_doors:.5f} (w={w_doors:g})\n"
                f"  Arch: {arch_num_layers}x{arch_layer_size} [{arch_activation}] | "
                f"LR={current_lr:.2e} (flat, no scheduler)\n"
                f"  Time Remaining: ETA={eta_str} | Elapsed={format_duration(elapsed)}\n"
                f"  Speed:          {avg_speed:5.1f} it/s\n"
                f"{'='*72}"
            )
            last_log_time = now
            last_log_iter = i

            # Export validation snapshot with nominal asymmetric ventilation:
            # e.g., Windows 1, 4, 8 open at 1.5 m/s, others closed (0.0 m/s), N_people = 30
            with torch.no_grad():
                res_grid = snapshot_resolution
                grid_x, grid_y, grid_z = np.mgrid[
                    bounds[0] : bounds[1] : complex(0, res_grid),
                    bounds[2] : bounds[3] : complex(0, res_grid),
                    bounds[4] : bounds[5] : complex(0, res_grid),
                ]
                grid_pts = np.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T.astype(np.float32)
                time_slices = np.linspace(0.0, t_max, 5)
                pvd_entries = []
                iter_dir = os.path.join(output_dir, f"snapshots_iter_{i:05d}")
                os.makedirs(iter_dir, exist_ok=True)

                # Validation scenario: Asymmetric cross-ventilation
                v_nom = np.zeros(num_windows, dtype=np.float32)
                if num_windows >= 1:
                    v_nom[0] = 1.5
                if num_windows >= 4:
                    v_nom[3] = 1.2
                if num_windows >= 8:
                    v_nom[7] = 1.5
                n_nom = 30.0
                for t_val in time_slices:
                    t_col = np.full((grid_pts.shape[0], 1), fill_value=t_val, dtype=np.float32)
                    v_mat = np.tile(v_nom, (grid_pts.shape[0], 1))
                    n_col = np.full((grid_pts.shape[0], 1), fill_value=n_nom, dtype=np.float32)
                    eval_inp = torch.tensor(
                        np.hstack([grid_pts, t_col, v_mat, n_col]), dtype=torch.float32, device=device
                    )
                    preds = model(eval_inp).cpu().numpy()

                    vtu = pv.PolyData(grid_pts).cast_to_unstructured_grid()
                    vtu.point_data["velocity_u"] = preds[:, 0]
                    vtu.point_data["velocity_v"] = preds[:, 1]
                    vtu.point_data["velocity_w"] = preds[:, 2]
                    vtu.point_data["velocity_mag"] = np.linalg.norm(preds[:, 0:3], axis=1)
                    vtu.point_data["pressure"] = preds[:, 3]
                    vtu.point_data["pollutant_c"] = preds[:, 4]

                    vtu_name = f"multi_win_n_{n_nom:.0f}_t_{t_val:05.1f}s.vtu"
                    vtu.save(os.path.join(iter_dir, vtu_name))
                    pvd_entries.append((float(t_val), vtu_name))

                pvd_path = os.path.join(iter_dir, "multi_window_timelapse.pvd")
                write_pvd_file(pvd_path, pvd_entries)

            extra_meta = build_extra_meta()
            save_training_checkpoint(
                os.path.join(output_dir, f"model_checkpoint_{i:05d}.pth"),
                iteration=i,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                extra_metadata=extra_meta,
            )
            save_training_checkpoint(
                os.path.join(output_dir, "model_latest.pth"),
                iteration=i,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                extra_metadata=extra_meta,
            )

    extra_meta = build_extra_meta()
    save_training_checkpoint(
        os.path.join(output_dir, "model_final.pth"),
        iteration=total_iters,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        extra_metadata=extra_meta,
    )
    save_training_checkpoint(
        os.path.join(output_dir, "model_latest.pth"),
        iteration=total_iters,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        extra_metadata=extra_meta,
    )
    total_time = time.time() - start_time
    log.info(
        f"13D Multi-Window PINN training complete in {format_duration(total_time)}! "
        f"Saved final model to {os.path.join(output_dir, 'model_final.pth')}"
    )


if __name__ == "__main__":
    room_trainer_multi_window_tanh()
