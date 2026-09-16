# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from datetime import datetime
import hydra
import torch
import numpy as np
import pyvista as pv
from omegaconf import DictConfig
import sys
from torch.optim import Adam, lr_scheduler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
GEOM_DIR = os.path.join(REPO_ROOT, "geometries")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable HH:MM:SS or MM:SS string."""
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


from sympy import Function, Number, Symbol, exp
from physicsnemo.utils.logging import PythonLogger
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.mesh.sampling import sample_random_points_on_cells
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.mesh.io import from_pyvista
from training.checkpoint_utils import load_checkpoint_for_training, save_training_checkpoint
from training.geometry_utils import (
    extract_room_domain,
    split_walls_and_obstacles,
    exclude_obstacle_volumes,
    verify_flow_direction,
    find_seated_breathing_center,
)


class NavierStokesPollutant3D(PDE):
    """Incompressible Navier-Stokes + Advection-Diffusion for Pollutant (steady, 3D).

    Governing Equations & Parameters:
    --------------------------------
    1. Continuity Equation (Mass conservation for incompressible fluid):
       ∇ · u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0

    2. Steady Momentum Equations (Navier-Stokes):
       (u · ∇)u = -(1/ρ) ∇p + ν ∇²u
       where:
         - u, v, w: Fluid velocity components in X, Y, Z directions [m/s]
         - p: Static pressure field [Pa] or [N/m²]
         - ρ (rho): Fluid density (default: 1.0 kg/m³)
         - ν (nu): Kinematic viscosity of air (default: 0.01 m²/s)

    3. Pollutant Advection-Diffusion Equation:
       u · ∇c = D ∇²c + S(x, y, z)
       where:
         - c: Pollutant scalar concentration [dimensionless or kg/m³]
         - D: Molecular/turbulent mass diffusion coefficient (default: 0.005 m²/s)
         - S: Continuous Gaussian source emission rate [1/s]
         - center (x0, y0, z0): Physical 3D coordinates of source origin [m]
         - sigma: Gaussian standard deviation / spatial spread of source (default: 0.2 m)
         - source_intensity: Peak emission intensity S0 at source center (default: 10.0 s⁻¹)
    """

    def __init__(
        self,
        nu: float = 0.01,
        rho: float = 1.0,
        D: float = 0.005,
        center: tuple = (7.79, 4.57, 1.10),
        sigma: float = 2.5,
        source_intensity: float = 0.00345,
    ):
        self.dim = 3
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        iv = {"x": x, "y": y, "z": z}

        u = Function("u")(*iv.values())  # Velocity X [m/s]
        v = Function("v")(*iv.values())  # Velocity Y [m/s]
        w = Function("w")(*iv.values())  # Velocity Z [m/s]
        p = Function("p")(*iv.values())  # Pressure [Pa]
        c = Function("c")(*iv.values())  # Pollutant concentration

        nu_sym, rho_sym, D_sym = Number(nu), Number(rho), Number(D)

        x0, y0, z0 = center
        S = Number(source_intensity) * exp(
            -((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / Number(sigma) ** 2
        )

        self.equations = {
            "continuity": u.diff(x) + v.diff(y) + w.diff(z),
            "momentum_x": (
                u * u.diff(x)
                + v * u.diff(y)
                + w * u.diff(z)
                + (1 / rho_sym) * p.diff(x)
                - nu_sym * (u.diff(x, 2) + u.diff(y, 2) + u.diff(z, 2))
            ),
            "momentum_y": (
                u * v.diff(x)
                + v * v.diff(y)
                + w * v.diff(z)
                + (1 / rho_sym) * p.diff(y)
                - nu_sym * (v.diff(x, 2) + v.diff(y, 2) + v.diff(z, 2))
            ),
            "momentum_z": (
                u * w.diff(x)
                + v * w.diff(y)
                + w * w.diff(z)
                + (1 / rho_sym) * p.diff(z)
                - nu_sym * (w.diff(x, 2) + w.diff(y, 2) + w.diff(z, 2))
            ),
            "transport": (
                u * c.diff(x)
                + v * c.diff(y)
                + w * c.diff(z)
                - D_sym * (c.diff(x, 2) + c.diff(y, 2) + c.diff(z, 2))
                - S
            ),
        }


@hydra.main(version_base="1.3", config_path="../", config_name="config.yaml")
def room_trainer(cfg: DictConfig) -> None:
    # 1. Device Selection optimized for M-series Mac (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Accelerating neural network training with Apple Silicon (MPS) 🚀")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Warning: Running on CPU.")

    log = PythonLogger(name="room_pollutant")
    log.file_logging()

    # Output directory formatted by date: ./outputs/YYYY-MM-DD
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(REPO_ROOT, "outputs", date_str)
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

    def sample_surface(surface_mesh, areas, n_points, current_device):
        cell_indices = torch.multinomial(areas, n_points, replacement=True).to(current_device)
        pts = sample_random_points_on_cells(surface_mesh, cell_indices)
        return pts.to(device=current_device, dtype=torch.float32)

    # Pre-generate valid interior points using the watertight RoomVolume geometry
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

    def sample_interior(n_points):
        idx = torch.randint(0, len(interior_pool), (n_points,), device=device)
        return interior_pool[idx].clone().requires_grad_(True)

    model = FullyConnected(
        in_features=3, out_features=5, num_layers=6, layer_size=512
    ).to(device)

    eq = NavierStokesPollutant3D(nu=0.01, rho=1.0, D=0.005, center=center)
    phy_inf = PhysicsInformer(
        required_outputs=["continuity", "momentum_x", "momentum_y", "momentum_z", "transport"],
        equations=eq,
        grad_method="autodiff",
        device=device,
    )

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
            expected_in_features=3,
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
            expected_in_features=3,
            expected_out_features=5,
        )

    total_iters = getattr(cfg, "max_iters", 30000)
    if start_iter >= total_iters:
        log.warning(
            f"Start iteration ({start_iter}) is >= max_iters ({total_iters}). "
            f"No training needed. Increase max_iters if you wish to continue further."
        )
        return

    log.info(f"Starting steady-state training from iteration {start_iter} to {total_iters:,} ({total_iters - start_iter} remaining)...")
    start_time = time.time()
    last_log_time = start_time
    last_log_iter = start_iter

    for i in range(start_iter, total_iters):
        optimizer.zero_grad()

        pts_walls = sample_surface(mesh_walls, walls_areas, 2000, device)
        pts_doors = sample_surface(mesh_doors, doors_areas, 1000, device)
        pts_windows = sample_surface(mesh_windows, windows_areas, 1000, device)
        pts_interior = sample_interior(8000)

        out_walls = model(pts_walls)
        out_doors = model(pts_doors)
        out_windows = model(pts_windows)
        out_interior = model(pts_interior)

        # 1. Walls: No-slip (u=0, v=0, w=0)
        loss_walls = torch.mean(out_walls[:, 0:3] ** 2)

        # 2. Windows (Inlet at Y≈9): Inflow towards negative Y (v = -1.0, u = 0, w = 0), Clean air (c=0)
        loss_windows_u = torch.mean(out_windows[:, 0] ** 2)
        loss_windows_v = torch.mean((out_windows[:, 1] + 1.0) ** 2)
        loss_windows_w = torch.mean(out_windows[:, 2] ** 2)
        loss_windows_c = torch.mean(out_windows[:, 4] ** 2)
        loss_windows = loss_windows_u + loss_windows_v + loss_windows_w + loss_windows_c

        # 3. Doors (Outlet at Y≈0): Zero pressure (p=0)
        loss_doors = torch.mean(out_doors[:, 3] ** 2)

        phy_loss_dict = phy_inf.forward(
            {
                "coordinates": pts_interior,
                "x": pts_interior[:, 0:1],
                "y": pts_interior[:, 1:2],
                "z": pts_interior[:, 2:3],
                "u": out_interior[:, 0:1],
                "v": out_interior[:, 1:2],
                "w": out_interior[:, 2:3],
                "p": out_interior[:, 3:4],
                "c": out_interior[:, 4:5],
            }
        )

        loss_phy = (
            torch.mean(phy_loss_dict["continuity"] ** 2)
            + torch.mean(phy_loss_dict["momentum_x"] ** 2)
            + torch.mean(phy_loss_dict["momentum_y"] ** 2)
            + torch.mean(phy_loss_dict["momentum_z"] ** 2)
            + torch.mean(phy_loss_dict["transport"] ** 2)
        )

        total_loss = loss_phy + loss_walls + loss_doors + loss_windows
        total_loss.backward()
        optimizer.step()
        scheduler.step()

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
                f"Speed: {sec_per_it:.2f} s/it ({speed:4.2f} it/s) | "
                f"Elapsed: {format_duration(elapsed)} | "
                f"ETA: {format_duration(eta_seconds)}"
            )

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
                f"  Loss Breakdown: Phy={loss_phy.item():.5f}, Walls={loss_walls.item():.5f}, "
                f"Doors={loss_doors.item():.5f}, Windows={loss_windows.item():.5f}\n"
                f"  Time Remaining: ETA={eta_str} | Elapsed={format_duration(elapsed)}\n"
                f"  Speed & LR:     {avg_speed:5.1f} it/s | LR={optimizer.param_groups[0]['lr']:.2e}\n"
                f"{'='*65}"
            )
            last_log_time = now
            last_log_iter = i

            with torch.no_grad():
                grid_x, grid_y, grid_z = np.mgrid[
                    bounds[0] : bounds[1] : 50j,
                    bounds[2] : bounds[3] : 50j,
                    bounds[4] : bounds[5] : 50j,
                ]
                grid_pts = np.vstack(
                    (grid_x.flatten(), grid_y.flatten(), grid_z.flatten())
                ).T.astype(np.float32)
                grid_tensor = torch.tensor(grid_pts, dtype=torch.float32, device=device)

                preds = model(grid_tensor).cpu().numpy()

                vtu = pv.PolyData(grid_pts).cast_to_unstructured_grid()
                vtu.point_data["velocity_u"] = preds[:, 0]
                vtu.point_data["velocity_v"] = preds[:, 1]
                vtu.point_data["velocity_w"] = preds[:, 2]
                vtu.point_data["velocity_mag"] = np.linalg.norm(preds[:, 0:3], axis=1)
                vtu.point_data["pressure"] = preds[:, 3]
                vtu.point_data["pollutant_c"] = preds[:, 4]

                vtu.save(os.path.join(output_dir, f"room_inference_{i:05d}.vtu"))

                extra_meta = {
                    "bounds": bounds,
                    "center": center,
                    "model_config": {
                        "in_features": 3,
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
                save_training_checkpoint(
                    os.path.join(REPO_ROOT, "outputs", "model_latest.pth"),
                    iteration=i,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    extra_metadata=extra_meta,
                )

    extra_meta = {
        "bounds": bounds,
        "center": center,
        "model_config": {
            "in_features": 3,
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
        f"Training complete in {format_duration(total_time)}! Saved final model to {os.path.join(output_dir, 'model_final.pth')}"
    )


if __name__ == "__main__":
    room_trainer()