# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Inference and visualization exporter for 6D Parametric PINN model.

Evaluates trained 6D model:
  (x, y, z, t, V_inlet, N_people) -> (u, v, w, p, c)
for arbitrary combinations of inlet velocity V_inlet and human occupancy N_people.
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
            {"in_features": 6, "out_features": 5, "num_layers": 6, "layer_size": 512},
        )
    else:
        state_dict = checkpoint
        model_config = {"in_features": 6, "out_features": 5, "num_layers": 6, "layer_size": 512}

    model = FullyConnected(
        in_features=model_config.get("in_features", 6),
        out_features=model_config.get("out_features", 5),
        num_layers=model_config.get("num_layers", 6),
        layer_size=model_config.get("layer_size", 512),
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def predict_in_batches(
    model: FullyConnected,
    coords_6d: np.ndarray,
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    total_pts = coords_6d.shape[0]
    preds = np.empty((total_pts, 5), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, total_pts, batch_size):
            end = min(start + batch_size, total_pts)
            batch = torch.from_numpy(coords_6d[start:end]).to(device=device, dtype=torch.float32)
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


def export_occupancy_timelapse(
    model: FullyConnected,
    bounds: Tuple[float, float, float, float, float, float],
    t_max: float,
    v_inlet: float,
    n_people: float,
    device: torch.device,
    output_dir: str,
    n_frames: int = 60,
    resolution: int = 40,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    res = resolution

    print(
        f"Generating {res}x{res}x{res} grid ({res**3:,} points) for V_inlet = {v_inlet:.2f} m/s, "
        f"N_people = {n_people:.0f} across {n_frames} steps..."
    )
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
        n_col = np.full((n_pts, 1), fill_value=n_people, dtype=np.float32)
        inp = np.hstack([grid_pts, t_col, v_col, n_col])
        preds = predict_in_batches(model, inp, device)

        vtu = pv.PolyData(grid_pts).cast_to_unstructured_grid()
        vtu.point_data["velocity_u"] = preds[:, 0]
        vtu.point_data["velocity_v"] = preds[:, 1]
        vtu.point_data["velocity_w"] = preds[:, 2]
        vtu.point_data["velocity_mag"] = np.linalg.norm(preds[:, 0:3], axis=1)
        vtu.point_data["pressure"] = preds[:, 3]
        vtu.point_data["pollutant_c"] = preds[:, 4]

        frame_file = f"v_{v_inlet:.2f}_n_{n_people:.0f}_step_{frame:04d}.vtu"
        vtu.save(os.path.join(output_dir, frame_file))
        entries.append((float(t_val), frame_file))

        if (frame + 1) % 10 == 0 or frame == n_frames - 1:
            print(f"    Exported frame {frame + 1}/{n_frames} at t = {t_val:.2f} s")

    pvd_path = os.path.join(output_dir, f"timelapse_v_{v_inlet:.2f}_n_{n_people:.0f}.pvd")
    write_pvd_file(pvd_path, entries)
    return pvd_path


def export_occupancy_sweep(
    model: FullyConnected,
    bounds: Tuple[float, float, float, float, float, float],
    t_eval: float,
    v_inlet: float,
    occupancy_list: List[float],
    device: torch.device,
    output_dir: str,
    resolution: int = 40,
) -> str:
    """Export comparative steady-state/time-slice snapshots across different occupancy counts."""
    os.makedirs(output_dir, exist_ok=True)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    res = resolution

    print(f"Exporting occupancy comparison across N_people = {occupancy_list} at t = {t_eval:.1f}s...")
    grid_x, grid_y, grid_z = np.mgrid[
        xmin:xmax:complex(0, res),
        ymin:ymax:complex(0, res),
        zmin:zmax:complex(0, res),
    ]
    grid_pts = np.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T.astype(np.float32)
    n_pts = grid_pts.shape[0]

    entries: List[Tuple[float, str]] = []
    for n_people in occupancy_list:
        t_col = np.full((n_pts, 1), fill_value=t_eval, dtype=np.float32)
        v_col = np.full((n_pts, 1), fill_value=v_inlet, dtype=np.float32)
        n_col = np.full((n_pts, 1), fill_value=n_people, dtype=np.float32)
        inp = np.hstack([grid_pts, t_col, v_col, n_col])
        preds = predict_in_batches(model, inp, device)

        vtu = pv.PolyData(grid_pts).cast_to_unstructured_grid()
        vtu.point_data["velocity_u"] = preds[:, 0]
        vtu.point_data["velocity_v"] = preds[:, 1]
        vtu.point_data["velocity_w"] = preds[:, 2]
        vtu.point_data["velocity_mag"] = np.linalg.norm(preds[:, 0:3], axis=1)
        vtu.point_data["pressure"] = preds[:, 3]
        vtu.point_data["pollutant_c"] = preds[:, 4]

        frame_file = f"occupancy_{n_people:02.0f}_people.vtu"
        vtu.save(os.path.join(output_dir, frame_file))
        entries.append((float(n_people), frame_file))
        print(f"    Exported N_people = {n_people:.0f} snapshot (peak CO2 concentration: {np.max(preds[:, 4]):.5f})")

    pvd_path = os.path.join(output_dir, "occupancy_sweep.pvd")
    write_pvd_file(pvd_path, entries)
    return pvd_path


def main() -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")
    default_dir = os.path.join(OUTPUT_DIR, date_str, "parametric_occupancy")
    default_ckpt = os.path.join(default_dir, "model_latest.pth")

    parser = argparse.ArgumentParser(
        description="Run inference and animations for 6D Parametric PINN model (x, y, z, t, V_inlet, N_people)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_ckpt,
        help="Path to trained 6D model checkpoint (.pth).",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=1.0,
        help="Inlet velocity V_inlet in m/s (default: 1.0).",
    )
    parser.add_argument(
        "--occupancy",
        type=float,
        default=30.0,
        help="Human occupant count N_people in [0, 50] (default: 30.0).",
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
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "T", "V_IN", "N_PEOPLE"),
        help="Probe model predictions at a single point (x, y, z, t, v_in, n_people).",
    )
    parser.add_argument(
        "--animate",
        choices=["timelapse", "occupancy_sweep", "all"],
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
    n_min = float(metadata.get("n_people_min", 0.0))
    n_max = float(metadata.get("n_people_max", 50.0))

    print(f"Domain bounds: X=[{bounds[0]:.2f}, {bounds[1]:.2f}], Y=[{bounds[2]:.2f}, {bounds[3]:.2f}], Z=[{bounds[4]:.2f}, {bounds[5]:.2f}]")
    print(f"6D Model: t ∈ [0.0, {t_max:.1f}] s | V_inlet ∈ [{v_min:.1f}, {v_max:.1f}] m/s | N_people ∈ [{n_min:.0f}, {n_max:.0f}]")

    # Point probe mode
    if args.probe:
        px, py, pz, pt, pv_val, pn_val = args.probe
        inp = torch.tensor([[px, py, pz, pt, pv_val, pn_val]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(inp).cpu().numpy()[0]
        v_mag = float(np.linalg.norm(pred[0:3]))
        print(f"\nPrediction at Point ({px:.3f}, {py:.3f}, {pz:.3f}) at t = {pt:.2f}s (V_inlet = {pv_val:.2f} m/s, N_people = {pn_val:.0f}):")
        print(f"  Velocity u:        {pred[0]:.5f} m/s")
        print(f"  Velocity v:        {pred[1]:.5f} m/s")
        print(f"  Velocity w:        {pred[2]:.5f} m/s")
        print(f"  Velocity Magnitude:{v_mag:.5f} m/s")
        print(f"  Pressure p:        {pred[3]:.5f} Pa")
        print(f"  CO2 Pollutant c:   {pred[4]:.5f}")
        return

    # Single snapshot mode
    if args.time is not None:
        t_eval = float(args.time)
        res = args.resolution
        v_val = args.velocity
        n_val = args.occupancy
        print(f"Exporting 3D volume at t = {t_eval:.2f} s (V_inlet = {v_val:.2f} m/s, N_people = {n_val:.0f})...")
        grid_x, grid_y, grid_z = np.mgrid[
            bounds[0]:bounds[1]:complex(0, res),
            bounds[2]:bounds[3]:complex(0, res),
            bounds[4]:bounds[5]:complex(0, res),
        ]
        grid_pts = np.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T.astype(np.float32)
        t_col = np.full((grid_pts.shape[0], 1), fill_value=t_eval, dtype=np.float32)
        v_col = np.full((grid_pts.shape[0], 1), fill_value=v_val, dtype=np.float32)
        n_col = np.full((grid_pts.shape[0], 1), fill_value=n_val, dtype=np.float32)
        inp = np.hstack([grid_pts, t_col, v_col, n_col])
        preds = predict_in_batches(model, inp, device)

        vtu = pv.PolyData(grid_pts).cast_to_unstructured_grid()
        vtu.point_data["velocity_u"] = preds[:, 0]
        vtu.point_data["velocity_v"] = preds[:, 1]
        vtu.point_data["velocity_w"] = preds[:, 2]
        vtu.point_data["velocity_mag"] = np.linalg.norm(preds[:, 0:3], axis=1)
        vtu.point_data["pressure"] = preds[:, 3]
        vtu.point_data["pollutant_c"] = preds[:, 4]

        os.makedirs(args.output_dir, exist_ok=True)
        out_vtu = os.path.join(args.output_dir, f"snapshot_v_{v_val:.2f}_n_{n_val:.0f}_t_{t_eval:.2f}s.vtu")
        vtu.save(out_vtu)
        print(f"Saved snapshot to: {out_vtu}")
        return

    # Animation export modes
    if args.animate:
        anim_dir = args.output_dir
        v_val = args.velocity
        n_val = args.occupancy
        print(f"\n{'='*60}")
        print(f"6D Parametric Occupancy PINN Animation Export: {args.animate}")
        print(f"  V_inlet: {v_val:.2f} m/s | N_people: {n_val:.0f}")
        print(f"  Output directory: {anim_dir}")
        print(f"{'='*60}\n")

        if args.animate in ("timelapse", "all"):
            print("[1/2] Exporting 3D physical time-lapse series...")
            pvd = export_occupancy_timelapse(
                model, bounds, t_max, v_val, n_val, device,
                os.path.join(anim_dir, f"timelapse_v_{v_val:.2f}_n_{n_val:.0f}"),
                n_frames=args.frames, resolution=args.resolution
            )
            print(f"  ✓ Time-lapse PVD: {pvd}\n")

        if args.animate in ("occupancy_sweep", "all"):
            print("[2/2] Exporting occupancy comparison sweep across N_people = [0, 10, 20, 30, 40, 50]...")
            pvd = export_occupancy_sweep(
                model, bounds, t_eval=t_max, v_inlet=v_val,
                occupancy_list=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
                device=device,
                output_dir=os.path.join(anim_dir, f"occupancy_sweep_v_{v_val:.2f}"),
                resolution=args.resolution
            )
            print(f"  ✓ Occupancy Sweep PVD: {pvd}\n")

        print(f"{'='*60}")
        print(f"6D Parametric animations exported! Open .pvd in ParaView.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
