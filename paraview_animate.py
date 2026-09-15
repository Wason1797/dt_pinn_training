#!/usr/bin/env pvpython
"""ParaView animation renderer for room airflow PINN results.

Usage:
    pvpython paraview_animate.py --type convergence --input convergence/convergence.pvd
    pvpython paraview_animate.py --type streamlines --input streamlines/streamlines.pvd
    pvpython paraview_animate.py --type sweep --input sweep/sweep.pvd
    /Applications/ParaView-6.1.1.app/Contents/bin/pvpython paraview_animate.py --type all --input-dir ./outputs/2026-09-14/animation/

Requires ParaView's pvpython (not regular Python).
Install ParaView from: https://www.paraview.org/download/
On macOS: /Applications/ParaView-6.1.1.app/Contents/bin/pvpython
"""

import argparse
import os
import sys
import time

from paraview.simple import *

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIELD_COLOR_MAPS = {
    "velocity_mag": "Viridis",
    "pressure": "Cool to Warm",
    "pollutant_c": "Inferno",
    "velocity_u": "Cool to Warm",
    "velocity_v": "Cool to Warm",
    "velocity_w": "Cool to Warm",
}

FIELD_LABELS = {
    "velocity_mag": "Velocity Magnitude [m/s]",
    "pressure": "Pressure [Pa]",
    "pollutant_c": "Pollutant Concentration",
    "velocity_u": "Velocity U [m/s]",
    "velocity_v": "Velocity V [m/s]",
    "velocity_w": "Velocity W [m/s]",
}

ANIMATION_SUBDIRS = {
    "convergence": "convergence",
    "streamlines": "streamlines",
    "sweep": "sweep",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_resolution(res_str: str) -> tuple:
    """Parse 'WxH' resolution string into (width, height) ints."""
    parts = res_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid resolution format '{res_str}', expected WxH (e.g. 1920x1080)")
    return int(parts[0]), int(parts[1])


def find_pvd_file(directory: str, anim_type: str) -> str | None:
    """Find the PVD collection file for a given animation type inside *directory*.

    Looks for ``<anim_type>/<anim_type>.pvd`` first, then falls back to any
    ``.pvd`` file inside ``<anim_type>/``.
    """
    subdir = os.path.join(directory, ANIMATION_SUBDIRS[anim_type])
    if not os.path.isdir(subdir):
        return None

    # Canonical name first
    canonical = os.path.join(subdir, f"{anim_type}.pvd")
    if os.path.isfile(canonical):
        return canonical

    # Fallback: first .pvd found
    for fname in sorted(os.listdir(subdir)):
        if fname.lower().endswith(".pvd"):
            return os.path.join(subdir, fname)
    return None


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# View & camera setup
# ---------------------------------------------------------------------------


def setup_view(width: int, height: int) -> object:
    """Create (or get) a render view and configure it for offscreen rendering."""
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [width, height]

    # Dark gradient background
    view.UseColorPaletteForBackground = 0
    view.Background = [0.12, 0.12, 0.15]
    view.Background2 = [0.22, 0.22, 0.28]
    view.BackgroundColorMode = "Gradient"

    # Anti-aliasing for quality
    view.EnableRayTracing = 0
    view.OrientationAxesVisibility = 0

    return view


def set_camera(view, preset: str, bounds: tuple) -> None:
    """Position the camera based on the data *bounds* and a named *preset*.

    Parameters
    ----------
    view : RenderView proxy
    preset : one of 'isometric', 'front', 'top', 'side'
    bounds : (xmin, xmax, ymin, ymax, zmin, zmax) of the data
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    diag = (dx**2 + dy**2 + dz**2) ** 0.5
    # Distance factor so the whole scene fits comfortably
    dist = diag * 1.8

    focal = [cx, cy, cz]
    view_up = [0, 0, 1]

    if preset == "front":
        # Looking along -Y
        position = [cx, cy - dist, cz]
    elif preset == "top":
        # Looking down along -Z
        position = [cx, cy, cz + dist]
        view_up = [0, -1, 0]
    elif preset == "side":
        # Looking along -X
        position = [cx - dist, cy, cz]
    else:
        # isometric – 45° azimuth, ~35° elevation
        import math

        az = math.radians(45)
        el = math.radians(35)
        position = [
            cx + dist * math.cos(el) * math.cos(az),
            cy - dist * math.cos(el) * math.sin(az),
            cz + dist * math.sin(el),
        ]

    camera = view.GetActiveCamera()
    camera.SetPosition(*position)
    camera.SetFocalPoint(*focal)
    camera.SetViewUp(*view_up)
    view.ResetCamera()
    # Slight zoom in after reset to tighten framing
    camera.Dolly(1.15)


def get_bounds(source) -> tuple:
    """Return the spatial bounds of a pipeline *source*."""
    source.UpdatePipeline()
    info = source.GetDataInformation()
    return info.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)


# ---------------------------------------------------------------------------
# STL overlay helper
# ---------------------------------------------------------------------------


def load_stl_overlay(stl_path: str, view) -> object | None:
    """Load an STL file and display it as a translucent wireframe.

    Returns the display proxy, or ``None`` if the file is missing.
    """
    if not os.path.isfile(stl_path):
        print(f"  [WARN] STL file not found, skipping overlay: {stl_path}")
        return None

    stl = STLReader(FileNames=[stl_path])
    stl.UpdatePipeline()

    display = Show(stl, view)
    display.Representation = "Wireframe"
    display.AmbientColor = [0.6, 0.6, 0.6]
    display.DiffuseColor = [0.6, 0.6, 0.6]
    display.Opacity = 0.25
    display.LineWidth = 1.0
    return display


# ---------------------------------------------------------------------------
# Color-mapping helpers
# ---------------------------------------------------------------------------


def apply_color_map(display, source, field: str, view) -> object:
    """Color a *display* by *field* using the appropriate LUT and add a color bar.

    Returns the scalar-bar (color-bar) proxy.
    """
    ColorBy(display, ("POINTS", field))
    display.RescaleTransferFunctionToDataRange(True, False)

    lut = GetColorTransferFunction(field)
    cmap_name = FIELD_COLOR_MAPS.get(field, "Viridis (matplotlib)")
    lut.ApplyPreset(cmap_name, True)

    display.SetScalarBarVisibility(view, True)

    scalar_bar = GetScalarBar(lut, view)
    scalar_bar.Title = FIELD_LABELS.get(field, field)
    scalar_bar.ComponentTitle = ""
    scalar_bar.TitleFontSize = 14
    scalar_bar.LabelFontSize = 12
    scalar_bar.ScalarBarLength = 0.35
    scalar_bar.WindowLocation = "Upper Right Corner"
    return scalar_bar


# ---------------------------------------------------------------------------
# Text annotation helpers
# ---------------------------------------------------------------------------


def add_text(view, text: str, location: str = "Upper Left Corner", font_size: int = 14) -> object:
    """Add a text annotation to the view. Returns the text source proxy."""
    txt = Text(Text=text)
    txt_display = Show(txt, view)
    txt_display.WindowLocation = location
    txt_display.FontSize = font_size
    txt_display.Color = [1.0, 1.0, 1.0]
    return txt


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _output_path(output_dir: str, anim_type: str, frame: int, fmt: str) -> str:
    """Build the output file path for a single frame."""
    subdir = os.path.join(output_dir, anim_type)
    ensure_dir(subdir)
    return os.path.join(subdir, f"{anim_type}_{frame:04d}.{fmt}")


def render_frame(view, path: str) -> None:
    """Save a single screenshot."""
    SaveScreenshot(
        path,
        view,
        ImageResolution=view.ViewSize,
        TransparentBackground=0,
    )


def render_as_avi(view, output_dir: str, anim_type: str) -> None:
    """Save the full animation as an AVI using ParaView's SaveAnimation."""
    ensure_dir(os.path.join(output_dir, anim_type))
    avi_path = os.path.join(output_dir, anim_type, f"{anim_type}.avi")
    print(f"  Saving AVI to {avi_path} ...")
    SaveAnimation(
        avi_path,
        view,
        ImageResolution=view.ViewSize,
        FrameRate=24,
    )
    print(f"  AVI saved: {avi_path}")


# ---------------------------------------------------------------------------
# Animation renderers
# ---------------------------------------------------------------------------


def render_convergence(
    pvd_path: str,
    output_dir: str,
    fmt: str,
    field: str,
    stl_path: str | None,
    camera_preset: str,
    resolution: tuple,
) -> None:
    """Render convergence animation: iso-surfaces colored by *field* at each timestep."""
    print(f"\n{'='*60}")
    print(f"  Convergence animation")
    print(f"  Input : {pvd_path}")
    print(f"  Field : {field}")
    print(f"{'='*60}")

    if not os.path.isfile(pvd_path):
        print(f"  [ERROR] PVD file not found: {pvd_path}")
        return

    ResetSession()
    width, height = resolution
    view = setup_view(width, height)

    # Load PVD data
    reader = PVDReader(FileName=pvd_path)
    reader.UpdatePipeline()

    timesteps = reader.TimestepValues if hasattr(reader, "TimestepValues") else []
    if not timesteps:
        # Try extracting from the animation scene
        scene = GetAnimationScene()
        scene.UpdateAnimationUsingDataTimeSteps()
        timesteps = scene.TimeKeeper.TimestepValues
    n_steps = len(timesteps) if timesteps else 1
    print(f"  Timesteps found: {n_steps}")

    bounds = get_bounds(reader)

    # Create a threshold to visualise the volume (keeps cells where the field
    # has finite values, effectively an iso-volume of the whole domain).
    threshold = Threshold(Input=reader, Scalars=["POINTS", field])
    info = reader.GetDataInformation()
    arr_info = info.GetPointDataInformation().GetArrayInformation(field)
    if arr_info:
        data_range = arr_info.GetComponentRange(0)
        threshold.LowerThreshold = data_range[0]
        threshold.UpperThreshold = data_range[1]
    else:
        threshold.LowerThreshold = 0.0
        threshold.UpperThreshold = 1e6
    threshold.UpdatePipeline()

    display = Show(threshold, view)
    display.Representation = "Surface"
    apply_color_map(display, threshold, field, view)

    # STL overlay
    if stl_path:
        load_stl_overlay(stl_path, view)

    # Camera
    set_camera(view, camera_preset, bounds)

    # Annotation
    annotation = add_text(view, "Iteration: 0", location="Upper Left Corner", font_size=16)

    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()

    t0 = time.time()

    if fmt == "avi":
        render_as_avi(view, output_dir, "convergence")
    else:
        for i, ts in enumerate(timesteps if timesteps else [0]):
            scene.AnimationTime = ts
            reader.UpdatePipeline(ts)
            threshold.UpdatePipeline(ts)

            # Update annotation
            iter_num = int(ts) if ts == int(ts) else ts
            annotation.Text = f"Iteration: {iter_num}"

            Render()
            out_path = _output_path(output_dir, "convergence", i, "png")
            render_frame(view, out_path)

            elapsed = time.time() - t0
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Frame {i+1}/{n_steps}  ({fps:.1f} fps)  -> {os.path.basename(out_path)}")

    print(f"  Convergence done – {n_steps} frames in {time.time()-t0:.1f}s")


def render_streamlines(
    pvd_path: str,
    output_dir: str,
    fmt: str,
    field: str,
    stl_path: str | None,
    camera_preset: str,
    resolution: tuple,
) -> None:
    """Render streamline/particle animation: glyphs colored by velocity_mag."""
    print(f"\n{'='*60}")
    print(f"  Streamline / particle animation")
    print(f"  Input : {pvd_path}")
    print(f"  Field : {field}")
    print(f"{'='*60}")

    if not os.path.isfile(pvd_path):
        print(f"  [ERROR] PVD file not found: {pvd_path}")
        return

    ResetSession()
    width, height = resolution
    view = setup_view(width, height)

    reader = PVDReader(FileName=pvd_path)
    reader.UpdatePipeline()

    timesteps = reader.TimestepValues if hasattr(reader, "TimestepValues") else []
    if not timesteps:
        scene = GetAnimationScene()
        scene.UpdateAnimationUsingDataTimeSteps()
        timesteps = scene.TimeKeeper.TimestepValues
    n_steps = len(timesteps) if timesteps else 1
    print(f"  Timesteps found: {n_steps}")

    bounds = get_bounds(reader)

    # Glyph – small spheres at each particle/point
    glyph = Glyph(Input=reader, GlyphType="Sphere")
    glyph.ScaleArray = ["POINTS", "No scale array"]
    glyph.ScaleFactor = 0.05
    glyph.GlyphMode = "All Points"
    glyph.GlyphType.Radius = 0.05
    glyph.GlyphType.ThetaResolution = 12
    glyph.GlyphType.PhiResolution = 12
    glyph.UpdatePipeline()

    display = Show(glyph, view)
    display.Representation = "Surface"
    # Color by the requested field (default velocity_mag for streamlines)
    color_field = field if field != "velocity_mag" else "velocity_mag"
    apply_color_map(display, glyph, color_field, view)

    # STL overlay
    if stl_path:
        load_stl_overlay(stl_path, view)

    set_camera(view, camera_preset, bounds)

    annotation = add_text(view, "Frame: 0", location="Upper Left Corner", font_size=16)

    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()

    t0 = time.time()

    if fmt == "avi":
        render_as_avi(view, output_dir, "streamlines")
    else:
        for i, ts in enumerate(timesteps if timesteps else [0]):
            scene.AnimationTime = ts
            reader.UpdatePipeline(ts)
            glyph.UpdatePipeline(ts)

            annotation.Text = f"Frame: {i}"

            Render()
            out_path = _output_path(output_dir, "streamlines", i, "png")
            render_frame(view, out_path)

            elapsed = time.time() - t0
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Frame {i+1}/{n_steps}  ({fps:.1f} fps)  -> {os.path.basename(out_path)}")

    print(f"  Streamlines done – {n_steps} frames in {time.time()-t0:.1f}s")


def render_sweep(
    pvd_path: str,
    output_dir: str,
    fmt: str,
    field: str,
    stl_path: str | None,
    camera_preset: str,
    resolution: tuple,
) -> None:
    """Render sweep animation: moving slice plane colored by *field*."""
    print(f"\n{'='*60}")
    print(f"  Sweep (slice) animation")
    print(f"  Input : {pvd_path}")
    print(f"  Field : {field}")
    print(f"{'='*60}")

    if not os.path.isfile(pvd_path):
        print(f"  [ERROR] PVD file not found: {pvd_path}")
        return

    ResetSession()
    width, height = resolution
    view = setup_view(width, height)

    reader = PVDReader(FileName=pvd_path)
    reader.UpdatePipeline()

    timesteps = reader.TimestepValues if hasattr(reader, "TimestepValues") else []
    if not timesteps:
        scene = GetAnimationScene()
        scene.UpdateAnimationUsingDataTimeSteps()
        timesteps = scene.TimeKeeper.TimestepValues
    n_steps = len(timesteps) if timesteps else 1
    print(f"  Timesteps found: {n_steps}")

    bounds = get_bounds(reader)

    # The sweep PVD already contains per-frame slice data, so we just
    # display the surface directly (each timestep is a different slice
    # position exported by inference.py).
    display = Show(reader, view)
    display.Representation = "Surface"
    apply_color_map(display, reader, field, view)

    # STL overlay
    if stl_path:
        load_stl_overlay(stl_path, view)

    set_camera(view, camera_preset, bounds)

    annotation = add_text(view, "Y = 0.00", location="Upper Left Corner", font_size=16)

    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()

    # Compute Y range from bounds for labelling
    y_min, y_max = bounds[2], bounds[3]

    t0 = time.time()

    if fmt == "avi":
        render_as_avi(view, output_dir, "sweep")
    else:
        for i, ts in enumerate(timesteps if timesteps else [0]):
            scene.AnimationTime = ts
            reader.UpdatePipeline(ts)

            # Get exact Y position from slice data bounds
            info = reader.GetDataInformation()
            if info:
                b = info.GetBounds()
                y_pos = 0.5 * (b[2] + b[3])
            else:
                frac = i / (n_steps - 1) if n_steps > 1 else 0.0
                y_pos = y_min + frac * (y_max - y_min)
            annotation.Text = f"Y = {y_pos:.2f} m"

            Render()
            out_path = _output_path(output_dir, "sweep", i, "png")
            render_frame(view, out_path)

            elapsed = time.time() - t0
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Frame {i+1}/{n_steps}  ({fps:.1f} fps)  -> {os.path.basename(out_path)}")

    print(f"  Sweep done – {n_steps} frames in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render animations from room-airflow PINN VTU data using ParaView.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  pvpython paraview_animate.py --type convergence --input convergence/convergence.pvd\n"
            "  pvpython paraview_animate.py --type all --input-dir ./outputs/2026-09-14/animation/\n"
            "  pvpython paraview_animate.py --type sweep --input sweep.pvd --field pollutant_c --format avi\n"
        ),
    )
    parser.add_argument(
        "--type",
        choices=["convergence", "streamlines", "sweep", "all"],
        default="all",
        help="Animation type to render (default: all)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to a specific .pvd file (required when --type is not 'all')",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing animation subdirs (convergence/, streamlines/, sweep/)",
    )
    parser.add_argument(
        "--output-dir",
        default="./animation_renders/",
        help="Output directory for rendered frames (default: ./animation_renders/)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "avi"],
        default="png",
        help="Output format: png frame sequence or avi video (default: png)",
    )
    parser.add_argument(
        "--resolution",
        default="1920x1080",
        help="Render resolution as WxH (default: 1920x1080)",
    )
    parser.add_argument(
        "--field",
        default="velocity_mag",
        choices=list(FIELD_COLOR_MAPS.keys()),
        help="Field to color by (default: velocity_mag)",
    )
    parser.add_argument(
        "--stl",
        default="RoomVolume_Walls.stl",
        help="Room geometry STL for wireframe overlay (default: RoomVolume_Walls.stl)",
    )
    parser.add_argument(
        "--no-stl",
        action="store_true",
        default=False,
        help="Disable STL overlay entirely",
    )
    parser.add_argument(
        "--camera",
        choices=["isometric", "front", "top", "side"],
        default="isometric",
        help="Camera preset (default: isometric)",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve resolution
    try:
        resolution = parse_resolution(args.resolution)
    except ValueError as exc:
        parser.error(str(exc))

    # STL path
    stl_path: str | None = None if args.no_stl else args.stl

    # Determine which animations to render and their PVD paths
    render_jobs: list[tuple[str, str]] = []  # (anim_type, pvd_path)

    if args.type == "all":
        if not args.input_dir:
            parser.error("--input-dir is required when --type is 'all'")
        for atype in ("convergence", "streamlines", "sweep"):
            pvd = find_pvd_file(args.input_dir, atype)
            if pvd:
                render_jobs.append((atype, pvd))
            else:
                print(f"  [WARN] No PVD file found for '{atype}' in {args.input_dir}")
    else:
        if not args.input:
            # Try to auto-discover from --input-dir
            if args.input_dir:
                pvd = find_pvd_file(args.input_dir, args.type)
                if pvd:
                    render_jobs.append((args.type, pvd))
                else:
                    parser.error(
                        f"No PVD file found for '{args.type}' in {args.input_dir}"
                    )
            else:
                parser.error("Either --input or --input-dir must be provided")
        else:
            if not os.path.isfile(args.input):
                parser.error(f"Input file not found: {args.input}")
            render_jobs.append((args.type, args.input))

    if not render_jobs:
        print("Nothing to render. Exiting.")
        sys.exit(0)

    ensure_dir(args.output_dir)

    print(f"\nParaView Animation Renderer")
    print(f"  Resolution : {resolution[0]}x{resolution[1]}")
    print(f"  Format     : {args.format}")
    print(f"  Field      : {args.field}")
    print(f"  Camera     : {args.camera}")
    print(f"  STL overlay: {stl_path or 'disabled'}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Jobs       : {len(render_jobs)}")

    dispatch = {
        "convergence": render_convergence,
        "streamlines": render_streamlines,
        "sweep": render_sweep,
    }

    t_total = time.time()

    for atype, pvd_path in render_jobs:
        dispatch[atype](
            pvd_path=pvd_path,
            output_dir=args.output_dir,
            fmt=args.format,
            field=args.field,
            stl_path=stl_path,
            camera_preset=args.camera,
            resolution=resolution,
        )

    elapsed = time.time() - t_total
    print(f"\nAll done – total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
