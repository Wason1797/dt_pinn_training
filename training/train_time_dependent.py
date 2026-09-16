# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Time-dependent 3D Room Navier-Stokes + Pollutant PINN training script.

Solves the unsteady 3D incompressible Navier-Stokes and pollutant advection-diffusion
equations inside a realistic room geometry using physics-informed neural networks.

Flow Direction:
  - Inlet: Windows (Y ≈ 9 m), inflow into the room in -Y direction with smooth ramp-up
  - Outlet: Doors (Y ≈ 0 m), pressure outlet (p = 0)
  - Walls: No-slip boundary (u = 0, v = 0, w = 0)
  - Initial Condition: Fluid initially at rest at t = 0 (u = v = w = p = c = 0)
"""

import os
import time
from datetime import datetime
from typing import Dict, Tuple
from xml.etree import ElementTree as ET

import hydra
import numpy as np
import pyvista as pv
import torch
from omegaconf import DictConfig
from torch.optim import Adam, lr_scheduler

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.sampling import sample_random_points_on_cells
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.utils.logging import PythonLogger
from training.checkpoint_utils import load_checkpoint_for_training, save_training_checkpoint
from training.geometry_utils import (
    extract_room_domain,
    split_walls_and_obstacles,
    exclude_obstacle_volumes,
    verify_flow_direction,
    find_seated_breathing_center,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEOM_DIR = os.path.join(REPO_ROOT, "geometries")


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


def compute_unsteady_pde_residuals(
    coords: torch.Tensor,
    t: torch.Tensor,
    out: torch.Tensor,
    nu: float = 0.01,
    rho: float = 1.0,
    D: float = 0.005,
    center: Tuple[float, float, float] = (7.79, 4.57, 1.10),
    source_intensity: float = 0.00345,
    sigma: float = 2.5,
) -> Dict[str, torch.Tensor]:
    """Compute unsteady 3D Navier-Stokes and pollutant transport residuals via autodiff.

    Governing Equations & Parameters:
    --------------------------------
    1. Incompressible Continuity:
       res_continuity = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0

    2. Unsteady Navier-Stokes Momentum Equations:
       res_mom_x = ∂u/∂t + (u·∇)u + (1/ρ) ∂p/∂x - ν ∇²u = 0
       res_mom_y = ∂v/∂t + (u·∇)v + (1/ρ) ∂p/∂y - ν ∇²v = 0
       res_mom_z = ∂w/∂t + (u·∇)w + (1/ρ) ∂p/∂z - ν ∇²w = 0
       where:
         - u, v, w: Fluid velocity components [m/s]
         - p: Static pressure [Pa]
         - ρ (rho): Fluid density [kg/m³] (default: 1.0 kg/m³)
         - ν (nu): Kinematic viscosity of air [m²/s] (default: 0.01 m²/s)
         - ∇²: Laplacian operator (∂²/∂x² + ∂²/∂y² + ∂²/∂z²)

    3. Unsteady Pollutant Advection-Diffusion-Reaction Equation:
       res_transport = ∂c/∂t + u·∇c - D ∇²c - S(x, y, z, t) = 0
       where:
         - c: Scalar concentration field [dimensionless]
         - D: Molecular/turbulent pollutant diffusion coefficient [m²/s] (default: 0.005 m²/s)
         - S(x, y, z): Gaussian continuous emission source [1/s]
         - center (x0, y0, z0): Physical coordinates of the emission source [m]
         - sigma: Gaussian spatial dispersion width [m] (default: 0.2 m)
         - source_intensity: Peak emission rate S0 [1/s] (default: 10.0 s⁻¹)
    """
    u = out[:, 0:1]
    v = out[:, 1:2]
    w = out[:, 2:3]
    p = out[:, 3:4]
    c = out[:, 4:5]

    ones = torch.ones_like(u)

    # 1. Temporal derivatives: ∂u/∂t, ∂v/∂t, ∂w/∂t, ∂c/∂t
    u_t = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=ones, create_graph=True)[0]
    w_t = torch.autograd.grad(w, t, grad_outputs=ones, create_graph=True)[0]
    c_t = torch.autograd.grad(c, t, grad_outputs=ones, create_graph=True)[0]

    # 2. First spatial derivatives w.r.t coords (N, 3)
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

    # 3. Second spatial derivatives (Laplacian)
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

    # Pollutant Gaussian source
    x0, y0, z0 = center
    dist_sq = (coords[:, 0:1] - x0) ** 2 + (coords[:, 1:2] - y0) ** 2 + (coords[:, 2:3] - z0) ** 2
    source = source_intensity * torch.exp(-dist_sq / (sigma**2))
    res_transport = c_t + u * c_x + v * c_y + w * c_z - D * laplace_c - source

    return {
        "continuity": res_continuity,
        "momentum_x": res_mom_x,
        "momentum_y": res_mom_y,
        "momentum_z": res_mom_z,
        "transport": res_transport,
    }


@hydra.main(version_base="1.3", config_path="../", config_name="config.yaml")
def room_trainer_time_dependent(cfg: DictConfig) -> None:
    # 1. Device Selection optimized for M-series Mac (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Accelerating neural network training with Apple Silicon (MPS) 🚀")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Warning: Running on CPU.")

    log = PythonLogger(name="room_pollutant_unsteady")
    log.file_logging()

    # Time parameters: Default 120 seconds
    t_max = float(getattr(cfg, "t_max", 120.0))
    tau_ramp = 2.0  # Smooth inlet velocity ramp-up time constant

    # Output directory formatted by date
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(REPO_ROOT, "outputs", date_str, "time_dependent")
    os.makedirs(output_dir, exist_ok=True)

    log.info("Loading STL geometries from geometries/ ...")
    volume_pv = pv.read(os.path.join(GEOM_DIR, "RoomVolume.stl"))
    walls_pv = pv.read(os.path.join(GEOM_DIR, "RoomVolume_Walls.stl"))
    windows_pv = pv.read(os.path.join(GEOM_DIR, "Windows.stl"))
    doors_pv = pv.read(os.path.join(GEOM_DIR, "Doors.stl"))

    # Verify physical flow direction: Windows -> Doors along -Y
    verify_flow_direction(windows_pv, doors_pv)

    # Identify outer walls and internal obstacles (columns + furniture)
    walls_pv, obstacle_bodies = split_walls_and_obstacles(walls_pv, geom_dir=GEOM_DIR)
    mesh_walls = from_pyvista(walls_pv)
    mesh_windows = from_pyvista(windows_pv)
    mesh_doors = from_pyvista(doors_pv)

    bounds = volume_pv.bounds
    center = find_seated_breathing_center(volume_pv, breathing_height=1.10)

    # Precompute cell areas for uniform area-weighted surface sampling
    walls_areas = torch.tensor(walls_pv.compute_cell_sizes().cell_data["Area"], dtype=torch.float32)
    doors_areas = torch.tensor(doors_pv.compute_cell_sizes().cell_data["Area"], dtype=torch.float32)
    windows_areas = torch.tensor(windows_pv.compute_cell_sizes().cell_data["Area"], dtype=torch.float32)

    def sample_surface_with_time(surface_mesh, areas, n_points, current_device):
        """Sample spatial points on mesh surface and pair with random t in [0, t_max]."""
        cell_indices = torch.multinomial(areas, n_points, replacement=True).to(current_device)
        pts = sample_random_points_on_cells(surface_mesh, cell_indices).to(
            device=current_device, dtype=torch.float32
        )
        t = torch.rand(n_points, 1, device=current_device, dtype=torch.float32) * t_max
        return torch.cat([pts, t], dim=1)

    # Pre-generate valid interior points using watertight RoomVolume geometry
    log.info("Generating interior collocation point pool inside watertight room volume...")
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

    def sample_interior_with_time(n_points):
        """Sample spatial interior points and time t, both requiring gradients."""
        idx = torch.randint(0, len(interior_pool), (n_points,), device=device)
        coords = interior_pool[idx].clone().requires_grad_(True)
        t = (torch.rand(n_points, 1, device=device, dtype=torch.float32) * t_max).requires_grad_(True)
        return coords, t

    def sample_initial_condition(n_points):
        """Sample points at initial time t = 0 across the interior."""
        idx = torch.randint(0, len(interior_pool), (n_points,), device=device)
        coords = interior_pool[idx].clone()
        t_zero = torch.zeros(n_points, 1, device=device, dtype=torch.float32)
        return torch.cat([coords, t_zero], dim=1)

    # Model: (x, y, z, t) -> (u, v, w, p, c)
    model = FullyConnected(
        in_features=4, out_features=5, num_layers=6, layer_size=512
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.99998717**step)

    # Checkpoint loading / Resumption
    resume_path = cfg.get("resume") or (cfg.get("training", {}).get("resume") if hasattr(cfg, "training") else None)
    checkpoint_path = cfg.get("checkpoint") or (cfg.get("training", {}).get("checkpoint") if hasattr(cfg, "training") else None)

    start_iter = 0
    if resume_path:
        start_iter, _ = load_checkpoint_for_training(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            resume=True,
            expected_in_features=4,
            expected_out_features=5,
        )
    elif checkpoint_path:
        _, _ = load_checkpoint_for_training(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            resume=False,
            expected_in_features=4,
            expected_out_features=5,
        )

    total_iters = getattr(cfg, "max_iters", 30000)
    if start_iter >= total_iters:
        log.warning(
            f"Start iteration ({start_iter}) is >= max_iters ({total_iters}). "
            f"No training needed. Increase max_iters if you wish to continue further."
        )
        return

    log.info(f"Starting time-dependent PINN training from iteration {start_iter} to {total_iters:,} (t ∈ [0, {t_max:.1f}]s)...")
    start_time = time.time()
    last_log_time = start_time
    last_log_iter = start_iter

    for i in range(start_iter, total_iters):
        optimizer.zero_grad()

        # 1. Sample spatiotemporal points
        inp_walls = sample_surface_with_time(mesh_walls, walls_areas, 2000, device)
        inp_windows = sample_surface_with_time(mesh_windows, windows_areas, 1000, device)
        inp_doors = sample_surface_with_time(mesh_doors, doors_areas, 1000, device)
        inp_ic = sample_initial_condition(2000)
        coords_int, t_int = sample_interior_with_time(8000)

        # 2. Forward pass
        out_walls = model(inp_walls)
        out_windows = model(inp_windows)
        out_doors = model(inp_doors)
        out_ic = model(inp_ic)
        out_interior = model(torch.cat([coords_int, t_int], dim=1))

        # 3. Initial Condition Loss at t = 0 (Fluid at rest, clean air, zero pressure)
        loss_ic_u = torch.mean(out_ic[:, 0] ** 2)
        loss_ic_v = torch.mean(out_ic[:, 1] ** 2)
        loss_ic_w = torch.mean(out_ic[:, 2] ** 2)
        loss_ic_p = torch.mean(out_ic[:, 3] ** 2)
        loss_ic_c = torch.mean(out_ic[:, 4] ** 2)
        loss_ic = loss_ic_u + loss_ic_v + loss_ic_w + loss_ic_p + loss_ic_c

        # 4. Walls: No-slip (u = 0, v = 0, w = 0) for all t
        loss_walls = torch.mean(out_walls[:, 0:3] ** 2)

        # 5. Windows (Inlet at Y≈9): Inflow in -Y direction with smooth ramp-up
        t_win = inp_windows[:, 3:4]
        v_target = -1.0 * torch.tanh(3.0 * t_win / tau_ramp)
        loss_windows_u = torch.mean(out_windows[:, 0] ** 2)
        loss_windows_v = torch.mean((out_windows[:, 1] - v_target) ** 2)
        loss_windows_w = torch.mean(out_windows[:, 2] ** 2)
        loss_windows_c = torch.mean(out_windows[:, 4] ** 2)  # Clean incoming air
        loss_windows = loss_windows_u + loss_windows_v + loss_windows_w + loss_windows_c

        # 6. Doors (Outlet at Y≈0): Zero relative pressure (p = 0)
        loss_doors = torch.mean(out_doors[:, 3] ** 2)

        # 7. Unsteady PDE Residuals in interior
        res_dict = compute_unsteady_pde_residuals(
            coords=coords_int,
            t=t_int,
            out=out_interior,
            nu=0.01,
            rho=1.0,
            D=0.005,
            center=center,
        )
        loss_phy = (
            torch.mean(res_dict["continuity"] ** 2)
            + torch.mean(res_dict["momentum_x"] ** 2)
            + torch.mean(res_dict["momentum_y"] ** 2)
            + torch.mean(res_dict["momentum_z"] ** 2)
            + torch.mean(res_dict["transport"] ** 2)
        )

        total_loss = loss_phy + loss_walls + loss_windows + loss_doors + loss_ic
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Progress logging
        if i % 10 == 0 and i > 0 and i % 1000 != 0:
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
                f"Loss: {total_loss.item():.5f} | "
                f"Phy: {loss_phy.item():.5f}, IC: {loss_ic.item():.5f} | "
                f"Speed: {sec_per_it:.2f} s/it ({speed:4.2f} it/s) | "
                f"Elapsed: {format_duration(elapsed)} | "
                f"ETA: {format_duration(eta_seconds)}"
            )

        # Checkpoint and time-slice snapshot export every 1000 iterations (and step 0)
        if i % 1000 == 0:
            now = time.time()
            elapsed = now - start_time
            pct = (i / total_iters) * 100.0
            avg_speed = i / elapsed if (i > 0 and elapsed > 0) else 0.0
            eta_seconds = (total_iters - i) / avg_speed if avg_speed > 0 else 0.0
            eta_str = format_duration(eta_seconds) if i > 0 else "estimating..."

            log.info(
                f"\n{'='*65}\n"
                f"[Iter: {i:05d}/{total_iters} ({pct:4.1f}%)] Total Loss: {total_loss.item():.5f}\n"
                f"  Loss Breakdown: Phy={loss_phy.item():.5f}, IC={loss_ic.item():.5f}, "
                f"Walls={loss_walls.item():.5f}, Win={loss_windows.item():.5f}, Doors={loss_doors.item():.5f}\n"
                f"  Time Remaining: ETA={eta_str} | Elapsed={format_duration(elapsed)}\n"
                f"  Speed & LR:     {avg_speed:5.1f} it/s | LR={optimizer.param_groups[0]['lr']:.2e}\n"
                f"{'='*65}"
            )
            last_log_time = now
            last_log_iter = i

            with torch.no_grad():
                res_grid = 35
                grid_x, grid_y, grid_z = np.mgrid[
                    bounds[0] : bounds[1] : complex(0, res_grid),
                    bounds[2] : bounds[3] : complex(0, res_grid),
                    bounds[4] : bounds[5] : complex(0, res_grid),
                ]
                grid_pts = np.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T.astype(np.float32)

                # Export physical time series snapshots (5 time points across 0 to t_max)
                time_slices = np.linspace(0.0, t_max, 5)
                pvd_entries = []
                iter_dir = os.path.join(output_dir, f"snapshots_iter_{i:05d}")
                os.makedirs(iter_dir, exist_ok=True)

                for t_idx, t_val in enumerate(time_slices):
                    t_col = np.full((grid_pts.shape[0], 1), fill_value=t_val, dtype=np.float32)
                    eval_inp = torch.tensor(np.hstack([grid_pts, t_col]), dtype=torch.float32, device=device)
                    preds = model(eval_inp).cpu().numpy()

                    vtu = pv.PolyData(grid_pts).cast_to_unstructured_grid()
                    vtu.point_data["velocity_u"] = preds[:, 0]
                    vtu.point_data["velocity_v"] = preds[:, 1]
                    vtu.point_data["velocity_w"] = preds[:, 2]
                    vtu.point_data["velocity_mag"] = np.linalg.norm(preds[:, 0:3], axis=1)
                    vtu.point_data["pressure"] = preds[:, 3]
                    vtu.point_data["pollutant_c"] = preds[:, 4]

                    vtu_name = f"time_{t_val:05.1f}s.vtu"
                    vtu.save(os.path.join(iter_dir, vtu_name))
                    pvd_entries.append((float(t_val), vtu_name))

                pvd_path = os.path.join(iter_dir, "timelapse.pvd")
                write_pvd_file(pvd_path, pvd_entries)

            # Checkpoint metadata
            extra_meta = {
                "bounds": bounds,
                "center": center,
                "t_max": t_max,
                "model_config": {
                    "in_features": 4,
                    "out_features": 5,
                    "num_layers": 6,
                    "layer_size": 512,
                },
            }
            save_training_checkpoint(
                os.path.join(output_dir, f"model_checkpoint_{i:05d}.pth"),
                iteration=i,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                extra_metadata=extra_meta,
            )
            save_training_checkpoint(
                os.path.join(output_dir, "model_latest.pth"),
                iteration=i,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                extra_metadata=extra_meta,
            )

    # Save final trained model
    extra_meta = {
        "bounds": bounds,
        "center": center,
        "t_max": t_max,
        "model_config": {
            "in_features": 4,
            "out_features": 5,
            "num_layers": 6,
            "layer_size": 512,
        },
    }
    save_training_checkpoint(
        os.path.join(output_dir, "model_final.pth"),
        iteration=total_iters,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        extra_metadata=extra_meta,
    )
    save_training_checkpoint(
        os.path.join(output_dir, "model_latest.pth"),
        iteration=total_iters,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        extra_metadata=extra_meta,
    )
    total_time = time.time() - start_time
    log.info(
        f"Time-dependent training complete in {format_duration(total_time)}! "
        f"Saved final model to {os.path.join(output_dir, 'model_final.pth')}"
    )


if __name__ == "__main__":
    room_trainer_time_dependent()
