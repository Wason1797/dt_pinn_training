# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Inference and animation export script for Parametric 3D Room PINN model.

Evaluates trained 5D model (x, y, z, t, V_inlet) -> (u, v, w, p, c) and exports
physical time-lapse series and particle pathlines for any requested inlet velocity.
"""

import argparse
from datetime import datetime
import os
import sys
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pyvista as pv
import torch

from physicsnemo.models.mlp.fully_connected import FullyConnected

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEOM_DIR = os.path.join(REPO_ROOT, "geometries")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")


def select_device(requested_device: str = "auto") -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.backends.mps.is_available():
        print("Using Apple Silicon (MPS) acceleration 🚀")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using NVIDIA CUDA acceleration 🚀")
        return torch.device("cuda")
    else:
        print("Running on CPU.")
        return torch.device("cpu")


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[FullyConnected, Dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    metadata: Dict = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        model_config = checkpoint.get(
            "model_config",
            {"in_features": 5, "out_features": 5, "num_layers": 6, "layer_size": 512},
        )
    else:
        state_dict = checkpoint
        model_config = {"in_features": 5, "out_features": 5, "num_layers": 6, "layer_size": 512}

    model = FullyConnected(
        in_features=model_config.get("in_features", 5),
        out_features=model_config.get("out_features", 5),
        num_layers=model_config.get("num_layers", 6),
        layer_size=model_config.get("layer_size", 512),
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def predict_in_batches(
    model: FullyConnected,
    coords_5d: np.ndarray,
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    total_pts = coords_5d.shape[0]
    preds = np.empty((total_pts, 5), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, total_pts, batch_size):
            end = min(start + batch_size, total_pts)
            batch = torch.from_numpy(coords_5d[start:end]).to(device=device, dtype=torch.float32)
            out = model(batch)
            preds[start:end] = out.cpu().numpy()

    return preds


def write_pvd_file(pvd_path: str, entries: List[Tuple[float, str]]) -> None:
    root = ET.Element("VTKFile", type="Collection", version="0.1")
    collection = ET.SubElement(root, "Collection")
    for timestep, filepath in entries:
        ET.SubElement(collection, "DataSet", timestep=str(timestep), file=filepath)
    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(os.path.abspath(pvd_path)), exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(pvd_path, xml_declaration=True, encoding="utf-8")
    print(f"  Written PVD collection: {pvd_path}")


def export_parametric_timelapse(
    model: FullyConnected,
    bounds: Tuple[float, float, float, float, float, float],
    t_max: float,
    v_inlet: float,
    device: torch.device,
    output_dir: str,
    n_frames: int = 60,
    resolution: int = 40,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    res = resolution
    print(f"Generating {res}x{res}x{res} grid ({res**3:,} points) for V_inlet = {v_inlet:.2f} m/s across {n_frames} steps...")
    grid_x, grid_y, grid_z = np.mgrid[
        xmin:xmax:complex(0, res),
        ymin:ymax:complex(0, res),
        zmin:zmax:complex(0, res),
    ]
    grid_pts = np.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T.astype(np.float32)
    n_pts = grid_pts.shape[0]

    time_values = np.linspace(0.0, t_max, n_frames)
    entries: List[Tuple[float, str]] = []

    for frame, t_val in enumerate(time_values):
        t_col = np.full((n_pts, 1), fill_value=t_val, dtype=np.float32)
        v_col = np.full((n_pts, 1), fill_value=v_inlet, dtype=np.float32)
        inp = np.hstack([grid_pts, t_col, v_col])
        preds = predict_in_batches(model, inp, device)

        vtu = pv.PolyData(grid_pts).cast_to_unstructured_grid()
        vtu.point_data["velocity_u"] = preds[:, 0]
        vtu.point_data["velocity_v"] = preds[:, 1]
        vtu.point_data["velocity_w"] = preds[:, 2]
        vtu.point_data["velocity_mag"] = np.linalg.norm(preds[:, 0:3], axis=1)
        vtu.point_data["pressure"] = preds[:, 3]
        vtu.point_data["pollutant_c"] = preds[:, 4]

        frame_file = f"v_{v_inlet:.2f}_step_{frame:04d}.vtu"
        vtu.save(os.path.join(output_dir, frame_file))
        entries.append((float(t_val), frame_file))

        if (frame + 1) % 10 == 0 or frame == n_frames - 1:
            print(f"    Exported frame {frame + 1}/{n_frames} at t = {t_val:.2f} s")

    pvd_path = os.path.join(output_dir, f"timelapse_v_{v_inlet:.2f}.pvd")
    write_pvd_file(pvd_path, entries)
    return pvd_path


def export_parametric_particles(
    model: FullyConnected,
    bounds: Tuple[float, float, float, float, float, float],
    t_max: float,
    v_inlet: float,
    device: torch.device,
    output_dir: str,
    n_frames: int = 120,
    n_seeds: int = 250,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    dt = t_max / n_frames

    y_inlet = ymax - 0.05 * (ymax - ymin)
    x_lo = xmin + 0.1 * (xmax - xmin)
    x_hi = xmax - 0.1 * (xmax - xmin)
    z_lo = zmin + 0.1 * (zmax - zmin)
    z_hi = zmax - 0.1 * (zmax - zmin)

    seed_x = np.random.uniform(x_lo, x_hi, n_seeds).astype(np.float32)
    seed_y = np.full(n_seeds, y_inlet, dtype=np.float32)
    seed_z = np.random.uniform(z_lo, z_hi, n_seeds).astype(np.float32)
    positions = np.stack([seed_x, seed_y, seed_z], axis=1)

    entries: List[Tuple[float, str]] = []
    print(f"  Advecting {n_seeds} particles for V_inlet = {v_inlet:.2f} m/s ({n_frames} steps, dt={dt:.4f}s)...")

    for frame in range(n_frames):
        current_t = frame * dt
        t_col = np.full((positions.shape[0], 1), fill_value=current_t, dtype=np.float32)
        v_col = np.full((positions.shape[0], 1), fill_value=v_inlet, dtype=np.float32)
        inp = np.hstack([positions, t_col, v_col])
        preds = predict_in_batches(model, inp, device)
        velocity = preds[:, 0:3]

        pts = pv.PolyData(positions.copy())
        pts.point_data["velocity_u"] = preds[:, 0]
        pts.point_data["velocity_v"] = preds[:, 1]
        pts.point_data["velocity_w"] = preds[:, 2]
        pts.point_data["velocity_mag"] = np.linalg.norm(velocity, axis=1)
        pts.point_data["pressure"] = preds[:, 3]
        pts.point_data["pollutant_c"] = preds[:, 4]

        vtu = pts.cast_to_unstructured_grid()
        frame_file = f"particle_{frame:04d}.vtu"
        vtu.save(os.path.join(output_dir, frame_file))
        entries.append((float(current_t), frame_file))

        positions = positions + dt * velocity

        out_of_bounds = (
            (positions[:, 0] < xmin) | (positions[:, 0] > xmax)
            | (positions[:, 1] < ymin) | (positions[:, 1] > ymax)
            | (positions[:, 2] < zmin) | (positions[:, 2] > zmax)
        )
        n_respawn = int(np.sum(out_of_bounds))
        if n_respawn > 0:
            positions[out_of_bounds, 0] = np.random.uniform(x_lo, x_hi, n_respawn)
            positions[out_of_bounds, 1] = y_inlet
            positions[out_of_bounds, 2] = np.random.uniform(z_lo, z_hi, n_respawn)

        if (frame + 1) % 20 == 0 or frame == n_frames - 1:
            print(f"    Frame {frame + 1}/{n_frames} at t = {current_t:.2f}s")

    pvd_path = os.path.join(output_dir, f"particles_v_{v_inlet:.2f}.pvd")
    write_pvd_file(pvd_path, entries)
    return pvd_path


def main() -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")
    default_dir = os.path.join(OUTPUT_DIR, date_str, "parametric")
    default_ckpt = os.path.join(default_dir, "model_latest.pth")

    parser = argparse.ArgumentParser(
        description="Run inference and animation exports for 5D Parametric PINN model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_ckpt,
        help="Path to trained parametric model checkpoint (.pth).",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=1.0,
        help="Inlet velocity V_inlet in m/s (default: 1.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(default_dir, "animations"),
        help="Directory to save VTU and PVD animation time series.",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Evaluate at a specific time instance (in seconds).",
    )
    parser.add_argument(
        "--probe",
        nargs=5,
        type=float,
        metavar=("X", "Y", "Z", "T", "V_IN"),
        help="Probe model predictions at a single point (x, y, z, t, v_in).",
    )
    parser.add_argument(
        "--animate",
        choices=["timelapse", "particles", "all"],
        default=None,
        help="Export time series animations for ParaView.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of physical time steps (frames).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=40,
        help="Spatial grid resolution per axis.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
    )
    args = parser.parse_args()

    device = select_device(args.device)
    model, metadata = load_model(args.checkpoint, device)
    bounds = tuple(metadata.get("bounds", (0.06, 15.53, -0.01, 9.16, -0.00, 3.13)))
    t_max = float(metadata.get("t_max", 120.0))
    v_min = float(metadata.get("v_min", 0.2))
    v_max = float(metadata.get("v_max", 2.5))

    print(f"Domain bounds: X=[{bounds[0]:.2f}, {bounds[1]:.2f}], Y=[{bounds[2]:.2f}, {bounds[3]:.2f}], Z=[{bounds[4]:.2f}, {bounds[5]:.2f}]")
    print(f"Model physical time horizon: t ∈ [0.0, {t_max:.1f}] s | Trained V_inlet ∈ [{v_min:.1f}, {v_max:.1f}] m/s")

    # Point probe mode
    if args.probe:
        px, py, pz, pt, pv_val = args.probe
        inp = torch.tensor([[px, py, pz, pt, pv_val]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(inp).cpu().numpy()[0]
        v_mag = float(np.linalg.norm(pred[0:3]))
        print(f"\nPrediction at Point ({px:.3f}, {py:.3f}, {pz:.3f}) at t = {pt:.2f}s with V_inlet = {pv_val:.2f} m/s:")
        print(f"  Velocity u:        {pred[0]:.5f} m/s")
        print(f"  Velocity v:        {pred[1]:.5f} m/s")
        print(f"  Velocity w:        {pred[2]:.5f} m/s")
        print(f"  Velocity Magnitude:{v_mag:.5f} m/s")
        print(f"  Pressure p:        {pred[3]:.5f} Pa")
        print(f"  Pollutant c:       {pred[4]:.5f}")
        return

    # Single snapshot at specific time
    if args.time is not None:
        t_eval = float(args.time)
        res = args.resolution
        v_val = args.velocity
        print(f"Exporting 3D volume at t = {t_eval:.2f} s with V_inlet = {v_val:.2f} m/s ({res}x{res}x{res} grid)...")
        grid_x, grid_y, grid_z = np.mgrid[
            bounds[0]:bounds[1]:complex(0, res),
            bounds[2]:bounds[3]:complex(0, res),
            bounds[4]:bounds[5]:complex(0, res),
        ]
        grid_pts = np.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T.astype(np.float32)
        t_col = np.full((grid_pts.shape[0], 1), fill_value=t_eval, dtype=np.float32)
        v_col = np.full((grid_pts.shape[0], 1), fill_value=v_val, dtype=np.float32)
        inp = np.hstack([grid_pts, t_col, v_col])
        preds = predict_in_batches(model, inp, device)

        vtu = pv.PolyData(grid_pts).cast_to_unstructured_grid()
        vtu.point_data["velocity_u"] = preds[:, 0]
        vtu.point_data["velocity_v"] = preds[:, 1]
        vtu.point_data["velocity_w"] = preds[:, 2]
        vtu.point_data["velocity_mag"] = np.linalg.norm(preds[:, 0:3], axis=1)
        vtu.point_data["pressure"] = preds[:, 3]
        vtu.point_data["pollutant_c"] = preds[:, 4]

        os.makedirs(args.output_dir, exist_ok=True)
        out_vtu = os.path.join(args.output_dir, f"snapshot_v_{v_val:.2f}_t_{t_eval:.2f}s.vtu")
        vtu.save(out_vtu)
        print(f"Saved snapshot to: {out_vtu}")
        return

    # Animation export modes
    if args.animate:
        anim_dir = args.output_dir
        v_val = args.velocity
        print(f"\n{'='*60}")
        print(f"Parametric PINN Animation Export: {args.animate} (V_inlet = {v_val:.2f} m/s)")
        print(f"Output directory: {anim_dir}")
        print(f"{'='*60}\n")

        if args.animate in ("timelapse", "all"):
            print("[1/2] Exporting 3D physical time-lapse series...")
            pvd = export_parametric_timelapse(
                model, bounds, t_max, v_val, device,
                os.path.join(anim_dir, f"timelapse_v_{v_val:.2f}"),
                n_frames=args.frames, resolution=args.resolution
            )
            print(f"  ✓ Time-lapse PVD: {pvd}\n")

        if args.animate in ("particles", "all"):
            print("[2/2] Exporting physical unsteady pathline particles...")
            pvd = export_parametric_particles(
                model, bounds, t_max, v_val, device,
                os.path.join(anim_dir, f"particles_v_{v_val:.2f}"),
                n_frames=args.frames
            )
            print(f"  ✓ Unsteady Particles PVD: {pvd}\n")

        print(f"{'='*60}")
        print(f"Parametric animations exported! Open .pvd in ParaView.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
