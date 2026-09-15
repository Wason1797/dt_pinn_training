# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Inference script for 3D Room Navier-Stokes + Pollutant PINN model."""

import argparse
from datetime import datetime
import glob
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
    """Select the most optimal hardware device available."""
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
    default_config: Optional[Dict] = None,
) -> Tuple[FullyConnected, Dict]:
    """Load the FullyConnected PINN model and metadata from a checkpoint."""
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
            default_config or {"in_features": 3, "out_features": 5, "num_layers": 6, "layer_size": 512},
        )
    else:
        # Bare state_dict
        state_dict = checkpoint
        model_config = default_config or {"in_features": 3, "out_features": 5, "num_layers": 6, "layer_size": 512}

    model = FullyConnected(
        in_features=model_config.get("in_features", 3),
        out_features=model_config.get("out_features", 5),
        num_layers=model_config.get("num_layers", 6),
        layer_size=model_config.get("layer_size", 512),
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def predict_in_batches(
    model: FullyConnected,
    coords: np.ndarray,
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    """Run model inference in chunks to avoid memory bottlenecks on MPS/GPU."""
    total_pts = coords.shape[0]
    preds = np.empty((total_pts, 5), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, total_pts, batch_size):
            end = min(start + batch_size, total_pts)
            batch_pts = torch.from_numpy(coords[start:end]).to(device=device, dtype=torch.float32)
            out = model(batch_pts)
            preds[start:end] = out.cpu().numpy()

    return preds


def get_domain_bounds(
    stl_path: Optional[str],
    metadata: Dict,
    user_bounds: Optional[list],
) -> Tuple[float, float, float, float, float, float]:
    """Determine domain bounding box [xmin, xmax, ymin, ymax, zmin, zmax]."""
    if user_bounds and len(user_bounds) == 6:
        return tuple(user_bounds)
    if "bounds" in metadata:
        return tuple(metadata["bounds"])
    if stl_path and os.path.exists(stl_path):
        mesh = pv.read(stl_path)
        return tuple(mesh.bounds)
    # Default room fallback
    return (0.0, 15.5, 0.0, 9.2, 0.0, 3.1)


def export_grid_vtu(
    grid_pts: np.ndarray,
    preds: np.ndarray,
    output_vtu: str,
) -> None:
    """Save predicted fields to an UnstructuredGrid VTU file for ParaView."""
    os.makedirs(os.path.dirname(os.path.abspath(output_vtu)), exist_ok=True)

    vtu = pv.PolyData(grid_pts).cast_to_unstructured_grid()
    vtu.point_data["velocity_u"] = preds[:, 0]
    vtu.point_data["velocity_v"] = preds[:, 1]
    vtu.point_data["velocity_w"] = preds[:, 2]
    vtu.point_data["velocity_mag"] = np.linalg.norm(preds[:, 0:3], axis=1)
    vtu.point_data["pressure"] = preds[:, 3]
    vtu.point_data["pollutant_c"] = preds[:, 4]

    vtu.save(output_vtu)
    print(f"Exported 3D field to VTU file: {output_vtu}")


def write_pvd_file(pvd_path: str, entries: List[Tuple[float, str]]) -> None:
    """Write a ParaView Data (PVD) collection file referencing VTU files with time steps."""
    root = ET.Element("VTKFile", type="Collection", version="0.1")
    collection = ET.SubElement(root, "Collection")
    for timestep, filepath in entries:
        ET.SubElement(collection, "DataSet", timestep=str(timestep), file=filepath)
    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(os.path.abspath(pvd_path)), exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(pvd_path, xml_declaration=True, encoding="utf-8")
    print(f"  Written PVD collection: {pvd_path}")


def export_convergence_series(checkpoint_dir: str, output_dir: str) -> str:
    """Create a PVD collection file from existing training VTU snapshots.

    The training script (train_w_mesh.py) exports room_inference_NNNNN.vtu
    at every 1000 iterations. This function writes a .pvd file that ParaView
    can load as a time series, mapping iteration index to time step.

    Returns: path to the .pvd file.
    """
    vtu_files = sorted(glob.glob(os.path.join(checkpoint_dir, "room_inference_*.vtu")))
    if not vtu_files:
        raise FileNotFoundError(
            f"No room_inference_*.vtu files found in: {checkpoint_dir}"
        )

    os.makedirs(output_dir, exist_ok=True)
    entries: List[Tuple[float, str]] = []
    for idx, vtu_path in enumerate(vtu_files):
        basename = os.path.basename(vtu_path)
        # Symlink into output dir so PVD references are local
        link_path = os.path.join(output_dir, basename)
        rel_source = os.path.relpath(vtu_path, output_dir)
        if not os.path.exists(link_path):
            os.symlink(rel_source, link_path)
        entries.append((float(idx), basename))

    pvd_path = os.path.join(output_dir, "convergence.pvd")
    write_pvd_file(pvd_path, entries)
    print(f"  Convergence series: {len(entries)} timesteps from training snapshots")
    return pvd_path


def export_streamline_series(
    model: FullyConnected,
    bounds: Tuple[float, float, float, float, float, float],
    device: torch.device,
    output_dir: str,
    n_frames: int = 120,
    n_seeds: int = 200,
) -> str:
    """Export VTU series of particles advected through the steady-state velocity field.

    Performs forward-Euler advection of seed particles placed near the inlet
    (door region, Y ≈ ymin). Particles that leave the domain are respawned
    at the inlet. Each frame is a point cloud with velocity and scalar data.

    Returns: path to the .pvd file.
    """
    os.makedirs(output_dir, exist_ok=True)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    # Auto-compute dt so particles roughly traverse the domain in n_frames
    dt = (ymax - ymin) / n_frames

    # Determine flow direction along Y to place inlet seeds
    test_pts = np.array([
        [(xmin + xmax) / 2.0, ymin + 0.1 * (ymax - ymin), (zmin + zmax) / 2.0],
        [(xmin + xmax) / 2.0, ymax - 0.1 * (ymax - ymin), (zmin + zmax) / 2.0],
    ], dtype=np.float32)
    test_preds = predict_in_batches(model, test_pts, device)
    mean_v = float(np.mean(test_preds[:, 1]))

    # If mean_v < 0, inflow is at Windows (ymax) blowing towards Doors (ymin)
    if mean_v < 0:
        y_inlet = ymax - 0.05 * (ymax - ymin)
        flow_desc = "Windows (Y≈ymax) -> Doors (Y≈ymin)"
    else:
        y_inlet = ymin + 0.05 * (ymax - ymin)
        flow_desc = "Doors (Y≈ymin) -> Windows (Y≈ymax)"

    x_lo = xmin + 0.1 * (xmax - xmin)
    x_hi = xmax - 0.1 * (xmax - xmin)
    z_lo = zmin + 0.1 * (zmax - zmin)
    z_hi = zmax - 0.1 * (zmax - zmin)

    seed_x = np.random.uniform(x_lo, x_hi, n_seeds).astype(np.float32)
    seed_y = np.full(n_seeds, y_inlet, dtype=np.float32)
    seed_z = np.random.uniform(z_lo, z_hi, n_seeds).astype(np.float32)
    positions = np.stack([seed_x, seed_y, seed_z], axis=1)

    entries: List[Tuple[float, str]] = []
    print(
        f"  Advecting {n_seeds} particles for {n_frames} frames (dt={dt:.4f})...\n"
        f"  Inlet seeds at Y={y_inlet:.2f}m [{flow_desc}]"
    )

    for frame in range(n_frames):
        # Predict all fields at current positions
        preds = predict_in_batches(model, positions, device)
        velocity = preds[:, 0:3]

        # Export current frame as a point cloud VTU
        pts = pv.PolyData(positions.copy())
        pts.point_data["velocity_u"] = preds[:, 0]
        pts.point_data["velocity_v"] = preds[:, 1]
        pts.point_data["velocity_w"] = preds[:, 2]
        pts.point_data["velocity_mag"] = np.linalg.norm(velocity, axis=1)
        pts.point_data["pressure"] = preds[:, 3]
        pts.point_data["pollutant_c"] = preds[:, 4]

        vtu = pts.cast_to_unstructured_grid()
        frame_file = f"frame_{frame:04d}.vtu"
        vtu.save(os.path.join(output_dir, frame_file))
        entries.append((float(frame), frame_file))

        # Advect particles: simple forward Euler integration
        positions = positions + dt * velocity

        # Respawn particles that leave the domain back at the inlet
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
            print(
                f"    Frame {frame + 1}/{n_frames} "
                f"({n_respawn} particles respawned)"
            )

    pvd_path = os.path.join(output_dir, "streamlines.pvd")
    write_pvd_file(pvd_path, entries)
    return pvd_path


def export_sweep_series(
    model: FullyConnected,
    bounds: Tuple[float, float, float, float, float, float],
    device: torch.device,
    output_dir: str,
    n_frames: int = 40,
    resolution: int = 100,
) -> str:
    """Export VTU series of XZ-plane slices sweeping along the Y axis.

    Sweeps along the primary flow direction: from Windows (ymax) to Doors (ymin)
    if flow is towards -Y, or ymin to ymax otherwise.

    Returns: path to the .pvd file.
    """
    os.makedirs(output_dir, exist_ok=True)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    # Check flow direction to sweep along the flow
    test_pts = np.array([
        [(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0],
    ], dtype=np.float32)
    mean_v = float(predict_in_batches(model, test_pts, device)[0, 1])
    if mean_v < 0:
        y_values = np.linspace(ymax, ymin, n_frames)
        print(f"  Sweeping {n_frames} slices from Windows (Y={ymax:.2f}m) to Doors (Y={ymin:.2f}m)...")
    else:
        y_values = np.linspace(ymin, ymax, n_frames)
        print(f"  Sweeping {n_frames} slices from Doors (Y={ymin:.2f}m) to Windows (Y={ymax:.2f}m)...")
    entries: List[Tuple[float, str]] = []

    for frame, y_val in enumerate(y_values):
        # Generate XZ grid at this Y position
        x = np.linspace(xmin, xmax, resolution)
        z = np.linspace(zmin, zmax, resolution)
        xx, zz = np.meshgrid(x, z)
        yy = np.full_like(xx, fill_value=y_val)

        slice_pts = np.vstack(
            (xx.flatten(), yy.flatten(), zz.flatten())
        ).T.astype(np.float32)
        preds = predict_in_batches(model, slice_pts, device)

        vtu = pv.PolyData(slice_pts).cast_to_unstructured_grid()
        vtu.point_data["velocity_u"] = preds[:, 0]
        vtu.point_data["velocity_v"] = preds[:, 1]
        vtu.point_data["velocity_w"] = preds[:, 2]
        vtu.point_data["velocity_mag"] = np.linalg.norm(preds[:, 0:3], axis=1)
        vtu.point_data["pressure"] = preds[:, 3]
        vtu.point_data["pollutant_c"] = preds[:, 4]

        frame_file = f"slice_{frame:04d}.vtu"
        vtu.save(os.path.join(output_dir, frame_file))
        entries.append((float(frame), frame_file))

        if (frame + 1) % 10 == 0 or frame == n_frames - 1:
            print(f"    Slice {frame + 1}/{n_frames} at Y={y_val:.2f}m")

    pvd_path = os.path.join(output_dir, "sweep.pvd")
    write_pvd_file(pvd_path, entries)
    return pvd_path


def plot_2d_slices(
    model: FullyConnected,
    bounds: Tuple[float, float, float, float, float, float],
    device: torch.device,
    output_png: str,
    resolution: int = 100,
) -> None:
    """Generate 2D slice visualizations (XY plane at mid-height Z) of model outputs."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    z_mid = (zmin + zmax) / 2.0

    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, fill_value=z_mid)

    slice_pts = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T.astype(np.float32)
    preds = predict_in_batches(model, slice_pts, device)

    u = preds[:, 0].reshape(resolution, resolution)
    v = preds[:, 1].reshape(resolution, resolution)
    w = preds[:, 2].reshape(resolution, resolution)
    v_mag = np.linalg.norm(preds[:, 0:3], axis=1).reshape(resolution, resolution)
    p = preds[:, 3].reshape(resolution, resolution)
    c = preds[:, 4].reshape(resolution, resolution)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Velocity Magnitude + Streamlines
    im0 = axes[0, 0].imshow(v_mag, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="viridis")
    axes[0, 0].streamplot(x, y, u, v, color="white", density=0.8, linewidth=0.7, arrowsize=0.8)
    fig.colorbar(im0, ax=axes[0, 0], label="Velocity Magnitude (m/s)")
    axes[0, 0].set_title(f"Velocity Field (Mid-Z = {z_mid:.2f} m)")
    axes[0, 0].set_xlabel("X (m)")
    axes[0, 0].set_ylabel("Y (m)")

    # Pressure
    im1 = axes[0, 1].imshow(p, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="coolwarm")
    fig.colorbar(im1, ax=axes[0, 1], label="Pressure (Pa)")
    axes[0, 1].set_title("Pressure Field")
    axes[0, 1].set_xlabel("X (m)")
    axes[0, 1].set_ylabel("Y (m)")

    # Pollutant Concentration
    im2 = axes[1, 0].imshow(c, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="inferno")
    fig.colorbar(im2, ax=axes[1, 0], label="Concentration c")
    axes[1, 0].set_title("Pollutant Concentration Field")
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
    date_dir = os.path.join(OUTPUT_DIR, date_str)

    # Find the most relevant default checkpoint
    date_ckpt = os.path.join(date_dir, "model_latest.pth")
    root_ckpt = os.path.join(OUTPUT_DIR, "model_latest.pth")
    default_ckpt = date_ckpt if os.path.exists(date_ckpt) else root_ckpt

    parser = argparse.ArgumentParser(
        description="Run inference with trained PhysicsNeMo room pollutant model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_ckpt,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-vtu",
        type=str,
        default=os.path.join(date_dir, "inference_results.vtu"),
        help="Path to save 3D VTU output for ParaView.",
    )
    parser.add_argument(
        "--save-slices",
        action="store_true",
        help="Also export 2D cross-section visualization image (PNG).",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default=os.path.join(date_dir, "inference_slice.png"),
        help="Path to save 2D slice plot if --save-slices is set.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=50,
        help="Grid resolution per axis (N x N x N). Default is 50.",
    )
    parser.add_argument(
        "--stl",
        type=str,
        default=os.path.join(GEOM_DIR, "RoomVolume.stl"),
        help="Path to RoomVolume STL to determine domain bounds.",
    )
    parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Explicit domain bounding box. Overrides STL/metadata bounds.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Compute device. Default auto selects MPS on Apple Silicon.",
    )
    parser.add_argument(
        "--probe",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Probe model predictions at a single point (x, y, z).",
    )

    # --- Animation export modes ---
    parser.add_argument(
        "--animate",
        choices=["convergence", "streamlines", "sweep", "all"],
        default=None,
        help="Export VTU time series for ParaView animation.",
    )
    parser.add_argument(
        "--anim-frames",
        type=int,
        default=120,
        help="Number of animation frames (for streamlines/sweep). Default: 120.",
    )
    parser.add_argument(
        "--anim-output-dir",
        type=str,
        default=None,
        help="Directory for animation VTU series. Default: <output-dir>/animation/",
    )
    args = parser.parse_args()

    device = select_device(args.device)

    # 1. Load model
    model, metadata = load_model(args.checkpoint, device)
    if "iteration" in metadata:
        print(f"Checkpoint iteration: {metadata['iteration']}")

    # 2. Point probe mode (if requested)
    if args.probe:
        px, py, pz = args.probe
        pt_tensor = torch.tensor([[px, py, pz]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(pt_tensor).cpu().numpy()[0]
        v_mag = float(np.linalg.norm(pred[0:3]))
        print(f"\nPrediction at Point ({px:.3f}, {py:.3f}, {pz:.3f}):")
        print(f"  Velocity u:        {pred[0]:.5f} m/s")
        print(f"  Velocity v:        {pred[1]:.5f} m/s")
        print(f"  Velocity w:        {pred[2]:.5f} m/s")
        print(f"  Velocity Magnitude:{v_mag:.5f} m/s")
        print(f"  Pressure p:        {pred[3]:.5f} Pa")
        print(f"  Pollutant c:       {pred[4]:.5f}")
        return

    # 3. Determine domain bounds
    bounds = get_domain_bounds(args.stl, metadata, args.bounds)
    print(
        f"Domain bounds: X=[{bounds[0]:.2f}, {bounds[1]:.2f}], "
        f"Y=[{bounds[2]:.2f}, {bounds[3]:.2f}], "
        f"Z=[{bounds[4]:.2f}, {bounds[5]:.2f}]"
    )

    # 4. Animation export mode
    if args.animate:
        anim_dir = args.anim_output_dir or os.path.join(date_dir, "animation")
        print(f"\n{'='*60}")
        print(f"Animation Export Mode: {args.animate}")
        print(f"Output directory: {anim_dir}")
        print(f"{'='*60}\n")

        if args.animate in ("convergence", "all"):
            print("[1/3] Exporting convergence series...")
            try:
                pvd = export_convergence_series(
                    date_dir, os.path.join(anim_dir, "convergence")
                )
                print(f"  ✓ Convergence PVD: {pvd}\n")
            except FileNotFoundError as e:
                print(f"  ⚠ Skipped convergence: {e}\n")

        if args.animate in ("streamlines", "all"):
            print("[2/3] Exporting streamline particle series...")
            pvd = export_streamline_series(
                model,
                bounds,
                device,
                os.path.join(anim_dir, "streamlines"),
                n_frames=args.anim_frames,
            )
            print(f"  ✓ Streamlines PVD: {pvd}\n")

        if args.animate in ("sweep", "all"):
            print("[3/3] Exporting spatial sweep series...")
            pvd = export_sweep_series(
                model,
                bounds,
                device,
                os.path.join(anim_dir, "sweep"),
                n_frames=args.anim_frames,
                resolution=max(args.resolution, 80),
            )
            print(f"  ✓ Sweep PVD: {pvd}\n")

        print(f"{'='*60}")
        print(f"Animation data exported to: {anim_dir}")
        print(f"Render with: pvpython paraview_animate.py --input-dir {anim_dir}")
        print(f"{'='*60}")
        return

    # 5. 3D Grid inference
    res = args.resolution
    print(f"Generating {res}x{res}x{res} grid ({res**3:,} points)...")
    grid_x, grid_y, grid_z = np.mgrid[
        bounds[0]:bounds[1]:complex(0, res),
        bounds[2]:bounds[3]:complex(0, res),
        bounds[4]:bounds[5]:complex(0, res),
    ]
    grid_pts = np.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T.astype(np.float32)

    print("Running forward inference...")
    preds = predict_in_batches(model, grid_pts, device)

    # Export VTU for 3D visualization
    export_grid_vtu(grid_pts, preds, args.output_vtu)

    # Optional 2D slice visualization
    if args.save_slices:
        plot_2d_slices(model, bounds, device, args.output_png, resolution=max(res, 80))

    print("Inference completed successfully! 🎉")


if __name__ == "__main__":
    main()
