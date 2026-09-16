# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Inference and visualization exporter for 13D Multi-Window Parametric PINN model.

Evaluates trained 13D model:
  (x, y, z, t, V_1, V_2, ..., V_8, N_people) -> (u, v, w, p, c)
for arbitrary combinations of independent window velocities V_1...V_8 and occupancy N_people.
"""

import argparse
from datetime import datetime
import os
import sys
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
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
            {"in_features": 13, "out_features": 5, "num_layers": 6, "layer_size": 512},
        )
    else:
        state_dict = checkpoint
        model_config = {"in_features": 13, "out_features": 5, "num_layers": 6, "layer_size": 512}

    in_features = model_config.get("in_features", 13)
    out_features = model_config.get("out_features", 5)
    num_layers = model_config.get("num_layers", 6)
    layer_size = model_config.get("layer_size", 512)

    if in_features != 13:
        raise ValueError(
            f"Checkpoint in_features is {in_features}; expected 13 for 8-window model "
            f"(3 coords + 1 time + 8 windows + 1 occupancy)."
        )

    model = FullyConnected(
        in_features=in_features,
        out_features=out_features,
        num_layers=num_layers,
        layer_size=layer_size,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def predict_in_batches(
    model: FullyConnected,
    coords_13d: np.ndarray,
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    total_pts = coords_13d.shape[0]
    preds = np.empty((total_pts, 5), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, total_pts, batch_size):
            end = min(start + batch_size, total_pts)
            batch = torch.from_numpy(coords_13d[start:end]).to(device=device, dtype=torch.float32)
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


def export_multi_window_snapshot(
    model: FullyConnected,
    bounds: Tuple[float, float, float, float, float, float],
    t_eval: float,
    velocities: np.ndarray,
    n_people: float,
    device: torch.device,
    output_vtu: str,
    resolution: int = 40,
) -> pv.UnstructuredGrid:
    """Evaluate 13D model on 3D grid for specific time, window velocities, and occupancy."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    res = resolution
    print(
        f"Evaluating 3D grid ({res}x{res}x{res} = {res**3:,} pts) at t={t_eval:.1f}s, "
        f"velocities={np.round(velocities, 2).tolist()}, N_people={n_people:.0f}..."
    )

    grid_x, grid_y, grid_z = np.mgrid[
        xmin:xmax:complex(0, res),
        ymin:ymax:complex(0, res),
        zmin:zmax:complex(0, res),
    ]
    grid_pts = np.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T.astype(np.float32)
    n_pts = grid_pts.shape[0]

    t_col = np.full((n_pts, 1), fill_value=t_eval, dtype=np.float32)
    v_mat = np.tile(velocities.reshape(1, 8), (n_pts, 1)).astype(np.float32)
    n_col = np.full((n_pts, 1), fill_value=n_people, dtype=np.float32)

    inp = np.hstack([grid_pts, t_col, v_mat, n_col])
    preds = predict_in_batches(model, inp, device)

    vtu = pv.PolyData(grid_pts).cast_to_unstructured_grid()
    vtu.point_data["velocity_u"] = preds[:, 0]
    vtu.point_data["velocity_v"] = preds[:, 1]
    vtu.point_data["velocity_w"] = preds[:, 2]
    vtu.point_data["velocity_mag"] = np.linalg.norm(preds[:, 0:3], axis=1)
    vtu.point_data["pressure"] = preds[:, 3]
    vtu.point_data["pollutant_c"] = preds[:, 4]

    os.makedirs(os.path.dirname(os.path.abspath(output_vtu)), exist_ok=True)
    vtu.save(output_vtu)
    print(f"Saved 3D VTU snapshot to: {output_vtu}")
    return vtu


def export_multi_window_timelapse(
    model: FullyConnected,
    bounds: Tuple[float, float, float, float, float, float],
    t_max: float,
    velocities: np.ndarray,
    n_people: float,
    device: torch.device,
    output_dir: str,
    n_frames: int = 60,
    resolution: int = 40,
) -> str:
    """Export animated time sequence PVD + VTU files for given velocities and occupancy."""
    os.makedirs(output_dir, exist_ok=True)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    res = resolution

    print(
        f"Generating timelapse ({n_frames} frames, {res}x{res}x{res}) across t ∈ [0, {t_max:.1f}]s, "
        f"V_k={np.round(velocities, 2).tolist()}, N_people={n_people:.0f}..."
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

    v_mat = np.tile(velocities.reshape(1, 8), (n_pts, 1)).astype(np.float32)
    n_col = np.full((n_pts, 1), fill_value=n_people, dtype=np.float32)

    for frame, t_val in enumerate(time_values):
        t_col = np.full((n_pts, 1), fill_value=t_val, dtype=np.float32)
        inp = np.hstack([grid_pts, t_col, v_mat, n_col])
        preds = predict_in_batches(model, inp, device)

        vtu = pv.PolyData(grid_pts).cast_to_unstructured_grid()
        vtu.point_data["velocity_u"] = preds[:, 0]
        vtu.point_data["velocity_v"] = preds[:, 1]
        vtu.point_data["velocity_w"] = preds[:, 2]
        vtu.point_data["velocity_mag"] = np.linalg.norm(preds[:, 0:3], axis=1)
        vtu.point_data["pressure"] = preds[:, 3]
        vtu.point_data["pollutant_c"] = preds[:, 4]

        frame_file = f"step_{frame:04d}_t_{t_val:05.1f}s.vtu"
        vtu.save(os.path.join(output_dir, frame_file))
        entries.append((float(t_val), frame_file))

        if (frame + 1) % 10 == 0 or frame == n_frames - 1:
            print(f"    Exported frame {frame + 1}/{n_frames} at t = {t_val:.2f} s")

    pvd_path = os.path.join(output_dir, "timelapse_multi_window.pvd")
    write_pvd_file(pvd_path, entries)
    return pvd_path


def plot_2d_slice(
    model: FullyConnected,
    bounds: Tuple[float, float, float, float, float, float],
    t_eval: float,
    velocities: np.ndarray,
    n_people: float,
    device: torch.device,
    output_png: str,
    z_eval: float = 1.10,
    resolution: int = 100,
) -> None:
    """Generate 2D XY-plane slice at specified height Z (default seated breathing height: 1.10 m)."""
    xmin, xmax, ymin, ymax, _, _ = bounds

    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, fill_value=z_eval)

    slice_pts = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T.astype(np.float32)
    n_pts = slice_pts.shape[0]

    t_col = np.full((n_pts, 1), fill_value=t_eval, dtype=np.float32)
    v_mat = np.tile(velocities.reshape(1, 8), (n_pts, 1)).astype(np.float32)
    n_col = np.full((n_pts, 1), fill_value=n_people, dtype=np.float32)

    inp = np.hstack([slice_pts, t_col, v_mat, n_col])
    preds = predict_in_batches(model, inp, device)

    u = preds[:, 0].reshape(resolution, resolution)
    v = preds[:, 1].reshape(resolution, resolution)
    w = preds[:, 2].reshape(resolution, resolution)
    v_mag = np.linalg.norm(preds[:, 0:3], axis=1).reshape(resolution, resolution)
    p = preds[:, 3].reshape(resolution, resolution)
    c = preds[:, 4].reshape(resolution, resolution)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Velocity Magnitude + Streamlines
    im0 = axes[0, 0].imshow(v_mag, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="viridis")
    axes[0, 0].streamplot(x, y, u, v, color="white", density=0.8, linewidth=0.7, arrowsize=0.8)
    fig.colorbar(im0, ax=axes[0, 0], label="Velocity Magnitude (m/s)")
    axes[0, 0].set_title(f"Velocity Field & Streamlines (Z = {z_eval:.2f} m, t = {t_eval:.1f} s)")
    axes[0, 0].set_xlabel("X (m)")
    axes[0, 0].set_ylabel("Y (m)")

    # Pressure
    im1 = axes[0, 1].imshow(p, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="coolwarm")
    fig.colorbar(im1, ax=axes[0, 1], label="Pressure (Pa)")
    axes[0, 1].set_title("Pressure Field")
    axes[0, 1].set_xlabel("X (m)")
    axes[0, 1].set_ylabel("Y (m)")

    # CO2 Concentration
    im2 = axes[1, 0].imshow(c, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="inferno")
    fig.colorbar(im2, ax=axes[1, 0], label="CO2 Concentration (g/m³)")
    axes[1, 0].set_title(f"CO2 Concentration Field (Occupancy = {n_people:.0f})")
    axes[1, 0].set_xlabel("X (m)")
    axes[1, 0].set_ylabel("Y (m)")

    # Vertical Velocity W
    im3 = axes[1, 1].imshow(w, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="bwr")
    fig.colorbar(im3, ax=axes[1, 1], label="Vertical Velocity w (m/s)")
    axes[1, 1].set_title("Vertical Velocity (w)")
    axes[1, 1].set_xlabel("X (m)")
    axes[1, 1].set_ylabel("Y (m)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Exported 2D slice visualization to: {output_png}")


def main() -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")
    default_dir = os.path.join(OUTPUT_DIR, date_str, "parametric_multi_window")
    default_ckpt = os.path.join(default_dir, "model_latest.pth")

    parser = argparse.ArgumentParser(
        description="Inference and visualization for 13D Multi-Window Parametric PINN model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_ckpt,
        help="Path to trained 13D multi-window model checkpoint (.pth).",
    )
    parser.add_argument(
        "--velocities",
        type=float,
        nargs="+",
        default=None,
        help="List of 8 inlet velocities for Windows 1..8 in m/s (e.g., --velocities 1.5 0.0 0.0 1.2 0.0 0.0 0.0 1.5).",
    )
    parser.add_argument(
        "--v-all",
        type=float,
        default=None,
        help="Set a uniform inlet velocity in m/s across all 8 windows (e.g., --v-all 1.0).",
    )
    parser.add_argument(
        "--open-windows",
        type=int,
        nargs="+",
        default=None,
        help="List of 1-based window indices to open (e.g., --open-windows 1 4 8), setting others to 0.0.",
    )
    parser.add_argument(
        "--open-velocity",
        type=float,
        default=1.0,
        help="Inlet velocity for windows specified by --open-windows (default: 1.0 m/s).",
    )
    parser.add_argument(
        "--occupancy",
        type=float,
        default=30.0,
        help="Human occupant count N_people in [0, 50] (default: 30.0).",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=60.0,
        help="Physical evaluation time in seconds (default: 60.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(default_dir, "inference"),
        help="Directory to save VTU, PVD, or PNG outputs.",
    )
    parser.add_argument(
        "--timelapse",
        action="store_true",
        help="Export full time-series animation sequence as PVD and VTU files.",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=60,
        help="Number of animation frames for timelapse (default: 60).",
    )
    parser.add_argument(
        "--save-slice",
        action="store_true",
        help="Export 2D slice visualization (PNG) at seated breathing height Z=1.10 m.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=40,
        help="Spatial grid resolution along each axis (default: 40).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Computing device (default: auto).",
    )

    args = parser.parse_args()
    device = select_device(args.device)

    model, metadata = load_model(args.checkpoint, device)

    # Determine 8 window velocities from arguments
    velocities = np.zeros(8, dtype=np.float32)
    if args.velocities is not None:
        if len(args.velocities) != 8:
            raise ValueError(f"Expected exactly 8 window velocities for --velocities, got {len(args.velocities)}.")
        velocities = np.array(args.velocities, dtype=np.float32)
    elif args.open_windows is not None:
        for idx in args.open_windows:
            if 1 <= idx <= 8:
                velocities[idx - 1] = args.open_velocity
            else:
                raise ValueError(f"Window index {idx} out of range (1..8).")
    elif args.v_all is not None:
        velocities = np.full(8, fill_value=args.v_all, dtype=np.float32)
    else:
        # Default: All windows open at 1.0 m/s
        velocities = np.ones(8, dtype=np.float32)

    print("\n" + "=" * 60)
    print("Multi-Window Configuration (8 Windows):")
    for w_idx, vel in enumerate(velocities, start=1):
        status = "CLOSED" if vel == 0.0 else f"OPEN ({vel:.2f} m/s)"
        print(f"  Window {w_idx}: {status}")
    print(f"Occupancy: {args.occupancy:.0f} occupants")
    print(f"Evaluation Time: {args.time:.1f} s")
    print("=" * 60 + "\n")

    # Load bounds from metadata or fallback to RoomVolume.stl
    if "bounds" in metadata:
        bounds = tuple(metadata["bounds"])
    else:
        volume_pv = pv.read(os.path.join(GEOM_DIR, "RoomVolume.stl"))
        bounds = volume_pv.bounds

    t_max = float(metadata.get("t_max", 120.0))
    os.makedirs(args.output_dir, exist_ok=True)

    if args.timelapse:
        pvd_path = export_multi_window_timelapse(
            model=model,
            bounds=bounds,
            t_max=t_max,
            velocities=velocities,
            n_people=args.occupancy,
            device=device,
            output_dir=os.path.join(args.output_dir, "timelapse"),
            n_frames=args.n_frames,
            resolution=args.resolution,
        )
        print(f"Exported timelapse to {pvd_path}")

    # Single snapshot export
    snapshot_vtu = os.path.join(args.output_dir, f"snapshot_t_{args.time:.1f}s.vtu")
    export_multi_window_snapshot(
        model=model,
        bounds=bounds,
        t_eval=args.time,
        velocities=velocities,
        n_people=args.occupancy,
        device=device,
        output_vtu=snapshot_vtu,
        resolution=args.resolution,
    )

    if args.save_slice:
        slice_png = os.path.join(args.output_dir, f"slice_t_{args.time:.1f}s.png")
        plot_2d_slice(
            model=model,
            bounds=bounds,
            t_eval=args.time,
            velocities=velocities,
            n_people=args.occupancy,
            device=device,
            output_png=slice_png,
            z_eval=1.10,
            resolution=max(args.resolution, 80),
        )


if __name__ == "__main__":
    main()
