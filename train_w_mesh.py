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
from torch.optim import Adam, lr_scheduler


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


class NavierStokesPollutant3D(PDE):
    """Incompressible Navier-Stokes + Advection-Diffusion for Pollutant (steady, 3D)."""

    def __init__(self, nu=0.01, rho=1.0, D=0.005, center=(0.0, 0.0, 0.0)):
        self.dim = 3
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        iv = {"x": x, "y": y, "z": z}
        
        u = Function("u")(*iv.values())  # type: ignore
        v = Function("v")(*iv.values())  # type: ignore
        w = Function("w")(*iv.values())  # type: ignore
        p = Function("p")(*iv.values())  # type: ignore
        c = Function("c")(*iv.values())  # type: ignore
        
        nu, rho, D = Number(nu), Number(rho), Number(D)
        
        x0, y0, z0 = center
        sigma = 0.2
        source_intensity = 10.0
        S = source_intensity * exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / sigma**2)
        
        self.equations = {
            "continuity": u.diff(x) + v.diff(y) + w.diff(z),
            "momentum_x": (
                u * u.diff(x) + v * u.diff(y) + w * u.diff(z)
                + (1 / rho) * p.diff(x)
                - nu * (u.diff(x, 2) + u.diff(y, 2) + u.diff(z, 2))
            ),
            "momentum_y": (
                u * v.diff(x) + v * v.diff(y) + w * v.diff(z)
                + (1 / rho) * p.diff(y)
                - nu * (v.diff(x, 2) + v.diff(y, 2) + v.diff(z, 2))
            ),
            "momentum_z": (
                u * w.diff(x) + v * w.diff(y) + w * w.diff(z)
                + (1 / rho) * p.diff(z)
                - nu * (w.diff(x, 2) + w.diff(y, 2) + w.diff(z, 2))
            ),
            "transport": (
                u * c.diff(x) + v * c.diff(y) + w * c.diff(z)
                - D * (c.diff(x, 2) + c.diff(y, 2) + c.diff(z, 2))
                - S
            )
        }


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
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
    output_dir = os.path.join("./outputs", date_str)
    os.makedirs(output_dir, exist_ok=True)

    log.info("Loading STL geometries...")
    volume_pv = pv.read("RoomVolume.stl")
    walls_pv = pv.read("RoomVolume_Walls.stl")
    windows_pv = pv.read("Windows.stl")
    doors_pv = pv.read("Doors.stl")

    mesh_volume = from_pyvista(volume_pv)
    mesh_walls = from_pyvista(walls_pv)
    mesh_windows = from_pyvista(windows_pv)
    mesh_doors = from_pyvista(doors_pv)

    bounds = volume_pv.bounds
    center = (
        (bounds[1] + bounds[0]) / 2.0,
        (bounds[3] + bounds[2]) / 2.0,
        (bounds[5] + bounds[4]) / 2.0
    )

    # Precompute cell areas for uniform area-weighted surface sampling
    walls_areas = torch.tensor(walls_pv.compute_cell_sizes().cell_data["Area"], dtype=torch.float32)
    doors_areas = torch.tensor(doors_pv.compute_cell_sizes().cell_data["Area"], dtype=torch.float32)
    windows_areas = torch.tensor(windows_pv.compute_cell_sizes().cell_data["Area"], dtype=torch.float32)

    def sample_surface(surface_mesh, areas, n_points, current_device):
        # Sample cells weighted by actual surface area
        cell_indices = torch.multinomial(areas, n_points, replacement=True).to(current_device)
        pts = sample_random_points_on_cells(surface_mesh, cell_indices)
        return pts.to(device=current_device, dtype=torch.float32)

    # Pre-generate valid interior points using the watertight RoomVolume geometry
    log.info("Generating interior collocation point pool inside watertight room volume...")
    raw_pts = np.random.uniform(
        [bounds[0], bounds[2], bounds[4]],
        [bounds[1], bounds[3], bounds[5]],
        size=(200000, 3)
    )
    cloud = pv.PolyData(raw_pts)
    # Use the main room enclosure
    body1 = volume_pv.split_bodies()[1].extract_surface(algorithm="dataset_surface")
    enclosed = cloud.select_interior_points(body1, check_surface=False)
    mask = enclosed["selected_points"].astype(bool)
    valid_interior_pts = raw_pts[mask]

    # Exclude interior columns/pillars from the fluid domain
    split_vol = volume_pv.split_bodies()
    for col_idx in range(2, len(split_vol)):
        col_b = split_vol[col_idx].bounds
        in_col = (
            (valid_interior_pts[:, 0] >= col_b[0]) & (valid_interior_pts[:, 0] <= col_b[1]) &
            (valid_interior_pts[:, 1] >= col_b[2]) & (valid_interior_pts[:, 1] <= col_b[3]) &
            (valid_interior_pts[:, 2] >= col_b[4]) & (valid_interior_pts[:, 2] <= col_b[5])
        )
        valid_interior_pts = valid_interior_pts[~in_col]

    interior_pool = torch.tensor(valid_interior_pts, dtype=torch.float32, device=device)
    log.info(f"Interior point pool ready: {len(interior_pool):,} points inside watertight room.")

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

    total_iters = 20000
    log.info(f"Starting training for {total_iters:,} iterations...")
    start_time = time.time()
    last_log_time = start_time
    last_log_iter = 0

    for i in range(total_iters):
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
        loss_walls = torch.mean(out_walls[:, 0:3]**2) 

        # 2. Windows (Inlet at Y≈9): Inflow into room towards negative Y (v = -1.0, u = 0, w = 0), Clean air (c=0)
        loss_windows_u = torch.mean(out_windows[:, 0]**2)
        loss_windows_v = torch.mean((out_windows[:, 1] + 1.0)**2)  # Inflow in -Y direction
        loss_windows_w = torch.mean(out_windows[:, 2]**2)
        loss_windows_c = torch.mean(out_windows[:, 4]**2) 
        loss_windows = loss_windows_u + loss_windows_v + loss_windows_w + loss_windows_c

        # 3. Doors (Outlet at Y≈0): Zero pressure (p=0) allows air to naturally exit
        loss_doors = torch.mean(out_doors[:, 3]**2)

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
            torch.mean(phy_loss_dict["continuity"]**2) +
            torch.mean(phy_loss_dict["momentum_x"]**2) +
            torch.mean(phy_loss_dict["momentum_y"]**2) +
            torch.mean(phy_loss_dict["momentum_z"]**2) +
            torch.mean(phy_loss_dict["transport"]**2)
        )

        total_loss = loss_phy + loss_walls + loss_doors + loss_windows
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Regular progress logging every 10 iterations (~20s on Apple Silicon)
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

        # Checkpoint and detailed progress every 1000 iterations (and step 0)
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
                    bounds[0]:bounds[1]:50j, 
                    bounds[2]:bounds[3]:50j, 
                    bounds[4]:bounds[5]:50j
                ]
                # Cast to float32 to prevent MPS backend errors
                grid_pts = np.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T.astype(np.float32)
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

                # Save periodic checkpoint
                checkpoint_data = {
                    "iteration": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "bounds": bounds,
                    "center": center,
                    "model_config": {
                        "in_features": 3,
                        "out_features": 5,
                        "num_layers": 6,
                        "layer_size": 512,
                    },
                }
                torch.save(checkpoint_data, os.path.join(output_dir, f"model_checkpoint_{i:05d}.pth"))
                torch.save(checkpoint_data, os.path.join(output_dir, "model_latest.pth"))
                torch.save(checkpoint_data, "./outputs/model_latest.pth")

    # Save final model
    final_checkpoint = {
        "iteration": 20000,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "bounds": bounds,
        "center": center,
        "model_config": {
            "in_features": 3,
            "out_features": 5,
            "num_layers": 6,
            "layer_size": 512,
        },
    }
    torch.save(final_checkpoint, os.path.join(output_dir, "model_final.pth"))
    torch.save(final_checkpoint, os.path.join(output_dir, "model_latest.pth"))
    total_time = time.time() - start_time
    log.info(f"Training complete in {format_duration(total_time)}! Saved final model to {os.path.join(output_dir, 'model_final.pth')}")

if __name__ == "__main__":
    room_trainer()