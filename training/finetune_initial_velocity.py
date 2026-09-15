# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Fine-tuning script for time-dependent PINN with custom initial velocity conditions.

Loads an existing time-dependent checkpoint and fine-tunes for N iterations
with a user-specified initial velocity field u0 = (u0, v0, w0) at t = 0.
"""

import argparse
from datetime import datetime
import os
import time
from typing import Dict, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pyvista as pv
import torch
from torch.optim import Adam, lr_scheduler

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.sampling import sample_random_points_on_cells
from physicsnemo.models.mlp.fully_connected import FullyConnected

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEOM_DIR = os.path.join(REPO_ROOT, "geometries")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string."""
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def write_pvd_file(pvd_path: str, entries: list) -> None:
    """Write a ParaView Data (PVD) collection file."""
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
         - u, v, w: Velocity field components [m/s]
         - p: Static pressure field [Pa]
         - ρ (rho): Air density [kg/m³] (default: 1.0 kg/m³)
         - ν (nu): Kinematic viscosity of air [m²/s] (default: 0.01 m²/s)

    3. Unsteady Pollutant Transport Equation:
       res_transport = ∂c/∂t + u·∇c - D ∇²c - S(x, y, z, t) = 0
       where:
         - c: Pollutant concentration [dimensionless]
         - D: Pollutant mass diffusivity [m²/s] (default: 0.005 m²/s)
         - S: Gaussian continuous emission source [1/s] (peak S0 = 10.0 s⁻¹, sigma = 0.2 m)
    """
    u = out[:, 0:1]
    v = out[:, 1:2]
    w = out[:, 2:3]
    p = out[:, 3:4]
    c = out[:, 4:5]

    ones = torch.ones_like(u)

    u_t = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=ones, create_graph=True)[0]
    w_t = torch.autograd.grad(w, t, grad_outputs=ones, create_graph=True)[0]
    c_t = torch.autograd.grad(c, t, grad_outputs=ones, create_graph=True)[0]

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

    res_continuity = u_x + v_y + w_z
    res_mom_x = u_t + u * u_x + v * u_y + w * u_z + (1.0 / rho) * p_x - nu * laplace_u
    res_mom_y = v_t + u * v_x + v * v_y + w * v_z + (1.0 / rho) * p_y - nu * laplace_v
    res_mom_z = w_t + u * w_x + v * w_y + w * w_z + (1.0 / rho) * p_z - nu * laplace_w

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune time-dependent PINN for custom initial velocity."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to source trained checkpoint (.pth).",
    )
    parser.add_argument(
        "--initial-velocity",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("U0", "V0", "W0"),
        help="Target initial velocity (u0, v0, w0) at t = 0 (default: 0.0 0.0 0.0).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3000,
        help="Number of fine-tuning iterations N (default: 3000).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Fine-tuning learning rate (default: 2e-4).",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=120.0,
        help="Time horizon in seconds (default: 120.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
    )
    args = parser.parse_args()

    # Device selection
    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (MPS) acceleration 🚀")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    u0, v0, w0 = args.initial_velocity
    t_max = args.t_max
    tau_ramp = 2.0

    date_str = datetime.now().strftime("%Y-%m-%d")
    out_dir = args.output_dir or os.path.join(
        REPO_ROOT, "outputs", date_str, f"finetuned_u0_{u0:.1f}_{v0:.1f}_{w0:.1f}"
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"Time-Dependent PINN Fine-Tuning")
    print(f"  Source Checkpoint: {args.checkpoint}")
    print(f"  Target Initial Velocity at t=0: ({u0:.2f}, {v0:.2f}, {w0:.2f}) m/s")
    print(f"  Iterations: {args.iterations:,} | LR: {args.lr:.2e} | T_max: {t_max:.1f}s")
    print(f"  Output Directory: {out_dir}")
    print(f"{'='*65}\n")

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model_config = ckpt.get("model_config", {"in_features": 4, "out_features": 5, "num_layers": 6, "layer_size": 512})
    model = FullyConnected(
        in_features=model_config.get("in_features", 4),
        out_features=model_config.get("out_features", 5),
        num_layers=model_config.get("num_layers", 6),
        layer_size=model_config.get("layer_size", 512),
    ).to(device)

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.train()

    # Load geometry
    volume_pv = pv.read(os.path.join(GEOM_DIR, "RoomVolume.stl"))
    walls_pv = pv.read(os.path.join(GEOM_DIR, "RoomVolume_Walls.stl"))
    windows_pv = pv.read(os.path.join(GEOM_DIR, "Windows.stl"))
    doors_pv = pv.read(os.path.join(GEOM_DIR, "Doors.stl"))

    mesh_walls = from_pyvista(walls_pv)
    mesh_windows = from_pyvista(windows_pv)
    mesh_doors = from_pyvista(doors_pv)

    bounds = tuple(ckpt.get("bounds", volume_pv.bounds))
    center = tuple(ckpt.get("center", (
        (bounds[1] + bounds[0]) / 2.0,
        (bounds[3] + bounds[2]) / 2.0,
        1.10,
    )))

    walls_areas = torch.tensor(walls_pv.compute_cell_sizes().cell_data["Area"], dtype=torch.float32)
    doors_areas = torch.tensor(doors_pv.compute_cell_sizes().cell_data["Area"], dtype=torch.float32)
    windows_areas = torch.tensor(windows_pv.compute_cell_sizes().cell_data["Area"], dtype=torch.float32)

    def sample_surface_with_time(surface_mesh, areas, n_points, current_device):
        cell_indices = torch.multinomial(areas, n_points, replacement=True).to(current_device)
        pts = sample_random_points_on_cells(surface_mesh, cell_indices).to(
            device=current_device, dtype=torch.float32
        )
        t = torch.rand(n_points, 1, device=current_device, dtype=torch.float32) * t_max
        return torch.cat([pts, t], dim=1)

    raw_pts = np.random.uniform(
        [bounds[0], bounds[2], bounds[4]],
        [bounds[1], bounds[3], bounds[5]],
        size=(200000, 3),
    )
    cloud = pv.PolyData(raw_pts)
    body1 = volume_pv.split_bodies()[1].extract_surface(algorithm="dataset_surface")
    enclosed = cloud.select_interior_points(body1, check_surface=False)
    mask = enclosed["selected_points"].astype(bool)
    valid_interior_pts = raw_pts[mask]

    # Exclude interior cylindrical columns/pillars from fluid domain
    walls_bodies = walls_pv.split_bodies()
    column_bodies = walls_bodies[1:5] if len(walls_bodies) >= 5 else walls_bodies[1:]
    col_excluded_count = 0
    for col_idx, col_mesh in enumerate(column_bodies, start=1):
        cx, cy, cz = col_mesh.center
        cb = col_mesh.bounds
        radius = 0.285  # Conservative radius covering full cylinder
        dist_sq = (valid_interior_pts[:, 0] - cx) ** 2 + (valid_interior_pts[:, 1] - cy) ** 2
        in_cylinder = (
            (dist_sq <= radius**2)
            & (valid_interior_pts[:, 2] >= cb[4] - 0.01)
            & (valid_interior_pts[:, 2] <= cb[5] + 0.01)
        )
        col_excluded_count += int(np.sum(in_cylinder))
        valid_interior_pts = valid_interior_pts[~in_cylinder]

    print(f"Excluded {col_excluded_count:,} points from inside {len(column_bodies)} cylindrical columns.")
    interior_pool = torch.tensor(valid_interior_pts, dtype=torch.float32, device=device)

    def sample_interior_with_time(n_points):
        idx = torch.randint(0, len(interior_pool), (n_points,), device=device)
        coords = interior_pool[idx].clone().requires_grad_(True)
        t = (torch.rand(n_points, 1, device=device, dtype=torch.float32) * t_max).requires_grad_(True)
        return coords, t

    def sample_initial_condition(n_points):
        idx = torch.randint(0, len(interior_pool), (n_points,), device=device)
        coords = interior_pool[idx].clone()
        t_zero = torch.zeros(n_points, 1, device=device, dtype=torch.float32)
        return torch.cat([coords, t_zero], dim=1)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=1e-5)

    start_time = time.time()
    last_log_time = start_time
    last_log_iter = 0

    for i in range(args.iterations):
        optimizer.zero_grad()

        inp_walls = sample_surface_with_time(mesh_walls, walls_areas, 2000, device)
        inp_windows = sample_surface_with_time(mesh_windows, windows_areas, 1000, device)
        inp_doors = sample_surface_with_time(mesh_doors, doors_areas, 1000, device)
        inp_ic = sample_initial_condition(2000)
        coords_int, t_int = sample_interior_with_time(8000)

        out_walls = model(inp_walls)
        out_windows = model(inp_windows)
        out_doors = model(inp_doors)
        out_ic = model(inp_ic)
        out_interior = model(torch.cat([coords_int, t_int], dim=1))

        # Initial condition penalty targeting (u0, v0, w0)
        loss_ic_u = torch.mean((out_ic[:, 0] - u0) ** 2)
        loss_ic_v = torch.mean((out_ic[:, 1] - v0) ** 2)
        loss_ic_w = torch.mean((out_ic[:, 2] - w0) ** 2)
        loss_ic_p = torch.mean(out_ic[:, 3] ** 2)
        loss_ic_c = torch.mean(out_ic[:, 4] ** 2)
        loss_ic = loss_ic_u + loss_ic_v + loss_ic_w + loss_ic_p + loss_ic_c

        # Walls no slip
        loss_walls = torch.mean(out_walls[:, 0:3] ** 2)

        # Windows inlet
        t_win = inp_windows[:, 3:4]
        v_target = -1.0 * torch.tanh(3.0 * t_win / tau_ramp)
        loss_windows_u = torch.mean(out_windows[:, 0] ** 2)
        loss_windows_v = torch.mean((out_windows[:, 1] - v_target) ** 2)
        loss_windows_w = torch.mean(out_windows[:, 2] ** 2)
        loss_windows_c = torch.mean(out_windows[:, 4] ** 2)
        loss_windows = loss_windows_u + loss_windows_v + loss_windows_w + loss_windows_c

        # Doors outlet
        loss_doors = torch.mean(out_doors[:, 3] ** 2)

        # Unsteady PDE
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

        total_loss = loss_phy + loss_walls + loss_windows + loss_doors + 2.0 * loss_ic
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 50 == 0 or i == args.iterations - 1:
            now = time.time()
            elapsed = now - start_time
            pct = ((i + 1) / args.iterations) * 100.0
            print(
                f"[Step {i+1:04d}/{args.iterations} ({pct:4.1f}%)] Loss: {total_loss.item():.5f} | "
                f"IC: {loss_ic.item():.5f}, Phy: {loss_phy.item():.5f} | Elapsed: {format_duration(elapsed)}"
            )

    # Save fine-tuned model and export timelapse snapshots
    with torch.no_grad():
        res_grid = 35
        grid_x, grid_y, grid_z = np.mgrid[
            bounds[0] : bounds[1] : complex(0, res_grid),
            bounds[2] : bounds[3] : complex(0, res_grid),
            bounds[4] : bounds[5] : complex(0, res_grid),
        ]
        grid_pts = np.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T.astype(np.float32)

        time_slices = np.linspace(0.0, t_max, 5)
        pvd_entries = []
        for t_val in time_slices:
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
            vtu.save(os.path.join(out_dir, vtu_name))
            pvd_entries.append((float(t_val), vtu_name))

        pvd_path = os.path.join(out_dir, "finetuned_timelapse.pvd")
        write_pvd_file(pvd_path, pvd_entries)

    final_ckpt = {
        "model_state_dict": model.state_dict(),
        "bounds": bounds,
        "center": center,
        "t_max": t_max,
        "initial_velocity": (u0, v0, w0),
        "model_config": model_config,
    }
    torch.save(final_ckpt, os.path.join(out_dir, "finetuned_model_final.pth"))
    print(f"\nFine-tuning completed in {format_duration(time.time() - start_time)}!")
    print(f"Saved fine-tuned checkpoint to: {os.path.join(out_dir, 'finetuned_model_final.pth')}")


if __name__ == "__main__":
    main()
