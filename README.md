# 3D Room Navier-Stokes + Pollutant PINN Framework

Physics-Informed Neural Network (PINN) framework for simulating 3D room airflow and pollutant dispersion inside complex geometries using NVIDIA PhysicsNeMo, PyTorch, and ParaView.

---

## 📁 Repository Structure

```text
pinn_training/
├── geometries/                     # 3D STL boundary & domain files
│   ├── Doors.stl                   # Pressure outlet openings (Y ≈ 0 m)
│   ├── Windows.stl                 # Inflow inlet openings (Y ≈ 9 m)
│   ├── RoomVolume.stl              # Watertight domain mesh for collocation
│   └── RoomVolume_Walls.stl        # No-slip boundary walls
├── training/                       # Neural network training pipelines
│   ├── train_steady.py             # 3D Steady Navier-Stokes + pollutant (30k iters)
│   ├── train_time_dependent.py     # 4D (x,y,z,t) Unsteady PINN (T_max=120s, 30k iters)
│   ├── finetune_initial_velocity.py# Fine-tune checkpoint for custom IC (u0, v0, w0)
│   ├── train_parametric.py         # 5D (x,y,z,t,V_inlet) Parametric PINN (120s, 30k iters)
│   ├── train_parametric_occupancy.py # 6D (x,y,z,t,V_inlet,N_people) Parametric PINN
│   ├── train_parametric_multi_window.py # 13D (x,y,z,t,V_1..V_8,N_people) Multi-Window PINN
│   ├── train_parametric_multi_window_tanh.py # 13D Multi-Window, tunable arch (Tanh, flat LR, weighted loss)
│   ├── checkpoint_utils.py         # Checkpoint save/load, resume & warm-start helpers
│   ├── geometry_utils.py           # STL domain extraction, window decomposition, obstacle masking
│   └── verify_geometry_match.py    # Regression check: geometry_utils vs legacy baseline
├── inference/                      # Forward evaluation & visualization exporters
│   ├── inference_steady.py         # Steady-state field, streamline & sweep exporter
│   ├── inference_time_dependent.py # Physical time-lapse & unsteady pathline exporter (120s)
│   ├── inference_parametric.py     # Real-time query for arbitrary V_inlet (0.2 - 2.5 m/s)
│   ├── inference_parametric_occupancy.py # Real-time query for arbitrary V_inlet & N_people
│   ├── inference_parametric_multi_window.py # Real-time query for independent window velocities (V1..V8)
│   └── inference_parametric_multi_window_tanh.py # Tunable-arch query (rebuilds the trained activation)
├── rendering/                      # Headless visualization & video rendering
│   └── paraview_animate.py         # ParaView / pvpython batch animation renderer
├── outputs/                        # Checkpoints (.pth), snapshots (.vtu), and collections (.pvd)
├── config.yaml                     # Model architecture and scheduler config
├── config_multi_window_tanh.yaml   # Live config for the tunable-architecture trainer
└── README.md
```

---

## 🚀 Installation

### 1. Install `uv`


```bash
brew install uv
```


### 2. Create the virtual environment and install

```bash
# From the repo root (pinn_training/)
uv sync
```

### 3. Run scripts

Use `.venv/bin/python` to invoke any script directly (no need to activate the environment):

```bash
.venv/bin/python training/train_steady.py
```

---

## 🔬 Physics & Governing Equations

The framework solves the **3D Incompressible Navier-Stokes Equations** coupled with a **Pollutant Advection-Diffusion-Reaction Equation**:

### 1. Mathematical Equations

$$\begin{aligned}
\text{1. Continuity (Mass Conservation):} \quad & \nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0 \\
\text{2. Momentum (Navier-Stokes):} \quad & \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} \\
\text{3. Pollutant Transport:} \quad & \frac{\partial c}{\partial t} + \mathbf{u} \cdot \nabla c = D \nabla^2 c + S(\mathbf{x}, t)
\end{aligned}$$

### 2. Parameter Definitions & Values

| Parameter | Symbol | Default Value | Physical Meaning |
| :--- | :---: | :---: | :--- |
| **Velocity Vector** | $\mathbf{u} = (u, v, w)$ | Variable | Air velocity field components in X, Y, Z directions $[\text{m/s}]$ |
| **Static Pressure** | $p$ | Variable | Relative fluid pressure field $[\text{Pa}]$ or $[\text{N/m}^2]$ |
| **Pollutant Concentration** | $c$ | Variable | Normalized scalar concentration field $[\text{dimensionless}]$ |
| **Fluid Density** | $\rho$ (`rho`) | $1.0\,\text{kg/m}^3$ | Ambient air density |
| **Kinematic Viscosity** | $\nu$ (`nu`) | $0.01\,\text{m}^2/\text{s}$ | Fluid kinematic viscosity (governs viscous momentum dissipation) |
| **Pollutant Mass Diffusivity** | $D$ | $0.005\,\text{m}^2/\text{s}$ | Molecular and turbulent diffusion coefficient of the pollutant |
| **Schmidt Number** | $\text{Sc} = \nu / D$ | $2.0$ | Ratio of momentum diffusivity to mass diffusivity |
| **Emission Source Function** | $S(\mathbf{x}, t)$ | Continuous | Gaussian emitter modeling **30 human occupants**: $S(\mathbf{x}) = S_0 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}_0\|^2}{\sigma^2}\right)$ |
| **Total $\text{CO}_2$ Emission Rate** | $\dot{Q}_{\text{total}}$ | $0.30\,\text{g/s}$ ($540\,\text{L/h}$) | 30 people emitting $\sim 0.010\,\text{g/s}$ ($18\,\text{L/h}$) each during light activity |
| **Peak Source Intensity** | $S_0$ (`source_intensity`) | $0.00345\,\text{g}/(\text{m}^3\cdot\text{s})$ | Integrated source amplitude: $\iiint S(\mathbf{x}) dV = S_0 \pi^{3/2} \sigma^3 = 0.30\,\text{g/s}$ |
| **Occupancy Area Spread** | $\sigma$ (`sigma`) | $2.5\,\text{m}$ | Characteristic Gaussian spread across classroom/office seating zone |
| **Breathing Zone Origin** | $\mathbf{x}_0$ (`center`) | $(7.79, 4.57, 1.10)\,\text{m}$ | Center of seating area at seated human breathing plane ($Z = 1.10\,\text{m}$) |
| **Inflow Ramp Time Constant** | $\tau$ (`tau_ramp`) | $2.0\,\text{s}$ | Smooth hyperbolic tangent inlet velocity ramp-up scale: $\tanh(3t/\tau)$ |

### 3. Boundary & Initial Conditions

- **Flow Direction:** Air enters through the **Windows** ($Y \approx 8.92\,\text{m}$) into the room with negative Y-velocity ($v < 0$) and exits naturally through the **Doors** ($Y \approx 0.05\,\text{m}$).
- **Windows ($Y \approx 9\,\text{m}$):** Inflow $v(t) = -V_{\text{inlet}} \cdot \tanh(3t/\tau)$ ($u=0, w=0$) with clean incoming air ($c=0$).
- **Doors ($Y \approx 0\,\text{m}$):** Zero relative pressure outlet ($p = 0\,\text{Pa}$).
- **Walls & Column Cylinders:** No-slip boundary ($u = v = w = 0\,\text{m/s}$).
- **Initial Condition ($t = 0$):** Fluid initially at rest ($\mathbf{u} = \mathbf{0}, p = 0, c = 0$) or customized via fine-tuning.

---

## 📖 Step-by-Step Workflow & Command Reference

---

### Step 1: Training the Models

#### 1.1 Steady-State PINN
*Solves the time-independent airflow and pollutant equilibrium field.*
```bash
.venv/bin/python training/train_steady.py
```
- **What it does:** Samples interior points from `RoomVolume.stl` and surface points on `Walls`, `Windows`, and `Doors`. Trains a 6-layer $\times$ 512 MLP $(x, y, z) \to (u, v, w, p, c)$ for 30,000 iterations.
- **Outputs:** Saves checkpoint to `outputs/YYYY-MM-DD/model_final.pth` and snapshots `room_inference_NNNNN.vtu`.

---

#### 1.2 Time-Dependent PINN ($T_{\max} = 120\,\text{s}$)
*Solves transient development from rest to fully-developed flow over physical time.*
```bash
.venv/bin/python training/train_time_dependent.py
```
- **What it does:** Samples spatiotemporal points $(x, y, z, t)$ where $t \in [0,\, 120\,\text{s}]$. Computes temporal acceleration $\frac{\partial \mathbf{u}}{\partial t}, \frac{\partial c}{\partial t}$ and enforces initial conditions at $t = 0$.
- **Outputs:** Saves checkpoint to `outputs/YYYY-MM-DD/time_dependent/model_final.pth` and multi-timestep validation snapshots `timelapse.pvd`.

---

#### 1.3 Fine-Tuning for Custom Initial Velocity
*Adapts a trained time-dependent model to a custom initial velocity in minutes.*
```bash
.venv/bin/python training/finetune_initial_velocity.py \
  --checkpoint outputs/2026-09-15/time_dependent/model_final.pth \
  --initial-velocity 0.0 -0.2 0.0 \
  --iterations 3000 \
  --lr 2e-4
```
- **What it does:** Resumes from the pre-trained weights, updates the initial condition loss $\mathcal{L}_{IC}$ to target $(u_0, v_0, w_0) = (0.0, -0.2, 0.0)\,\text{m/s}$ at $t = 0$, and optimizes for $N$ iterations.
- **Outputs:** Saves `finetuned_model_final.pth` and `finetuned_timelapse.pvd`.

---

#### 1.4 5D Parametric PINN ($V_{\text{inlet}} \in [0.2,\, 2.5]\,\text{m/s}$)
*Trains a single surrogate network capable of predicting any inlet velocity on the fly.*
```bash
.venv/bin/python training/train_parametric.py
```
- **What it does:** Trains a 5D MLP $(x, y, z, t, V_{\text{inlet}}) \to (u, v, w, p, c)$ sampling continuous velocities during optimization.
- **Outputs:** Saves `outputs/YYYY-MM-DD/parametric/model_final.pth`.

---

#### 1.5 6D Parametric PINN with Occupancy ($V_{\text{inlet}} \in [0.2,\, 2.5]\,\text{m/s},\; N_{\text{people}} \in [0,\, 50]$)
*Trains a 6D surrogate model predicting airflow and $\text{CO}_2$ concentration for any velocity and any occupant count.*
```bash
.venv/bin/python training/train_parametric_occupancy.py
```
- **What it does:** Trains a 6D MLP $(x, y, z, t, V_{\text{inlet}}, N_{\text{people}}) \to (u, v, w, p, c)$ where the $\text{CO}_2$ source rate scales directly with the number of people ($S_0 = N_{\text{people}} \times 1.15 \times 10^{-4}\,\text{g}/(\text{m}^3\cdot\text{s})$).
- **Outputs:** Saves `outputs/YYYY-MM-DD/parametric_occupancy/model_final.pth`.

---

#### 1.6 13D Multi-Window Parametric PINN ($V_1 \dots V_8 \in [0,\, 2.5]\,\text{m/s},\; N_{\text{people}} \in [0,\, 50]$)
*Trains a 13D surrogate model predicting airflow and $\text{CO}_2$ concentration for independent window velocities and occupant count.*
```bash
.venv/bin/python training/train_parametric_multi_window.py
```
- **What it does:** Trains a 13D MLP $(x, y, z, t, V_1, V_2, V_3, V_4, V_5, V_6, V_7, V_8, N_{\text{people}}) \to (u, v, w, p, c)$ where each $V_i$ represents the independent inlet velocity for window $i$, and the $\text{CO}_2$ source rate scales with occupancy ($S_0 = N_{\text{people}} \times 1.15 \times 10^{-4}\,\text{g}/(\text{m}^3\cdot\text{s})$).
- **Outputs:** Saves `outputs/YYYY-MM-DD/parametric_multi_window/model_final.pth`.

---

#### 1.7 Tunable-Architecture 13D Multi-Window PINN ($V_1 \dots V_8 \in [0,\, 2.5]\,\text{m/s},\; N_{\text{people}} \in [0,\, 50]$)
*Identical 13D physics, geometry and sampling to 1.6, but with a configurable network, a flat learning rate, and per-term loss weights — built for finding which architecture actually converges.*
```bash
# Train with the defaults in config_multi_window_tanh.yaml: 5 hidden layers x 128, Tanh
.venv/bin/python training/train_parametric_multi_window_tanh.py
```

- **What it does:** Trains the same 13D MLP $(x, y, z, t, V_1, \dots, V_8, N_{\text{people}}) \to (u, v, w, p, c)$ as 1.6. Only the network and the optimization recipe differ:

  | | 1.6 `train_parametric_multi_window.py` | 1.7 `train_parametric_multi_window_tanh.py` |
  | :--- | :--- | :--- |
  | Depth / width | 6 hidden $\times$ 512 (hardcoded) | configurable, default 5 hidden $\times$ 128 |
  | Activation | SiLU (PhysicsNeMo default) | configurable, default **Tanh** |
  | Learning rate | `LambdaLR` exponential decay | **flat, no scheduler** |
  | Loss | unweighted sum of all 5 terms | per-term weights, default `walls=windows=100` |
  | Logging | weighted total only | weighted total **and** raw unweighted per-term |

- **Configuration:** every knob lives in [config_multi_window_tanh.yaml](config_multi_window_tanh.yaml) and is overridable on the command line. Valid activations are any PhysicsNeMo `ACT2FN` key (`tanh`, `silu`, `gelu`, `sin`, `softplus`, `elu`, ...); do **not** use `stan`, which is registered but broken in physicsnemo 2.2.1.
- **Outputs:** Saves `outputs/YYYY-MM-DD/parametric_multi_window_<activation>_L<num_layers>_W<layer_size>/model_final.pth`. The run directory is auto-named from the architecture, so sweeps never overwrite each other. Each run also writes its own `launch.log` and a `loss_history.csv` for comparing runs.

Overriding the architecture and recipe:
```bash
# A wider, deeper network
.venv/bin/python training/train_parametric_multi_window_tanh.py \
  arch.num_layers=6 arch.layer_size=256

# Exact parity with the AIQ notebook architecture (4 hidden x 64, Tanh -> 13,701 params)
.venv/bin/python training/train_parametric_multi_window_tanh.py \
  arch.num_layers=4 arch.layer_size=64 arch.activation=tanh

# Reproduce 1.6's unweighted loss, to isolate the effect of the architecture alone
.venv/bin/python training/train_parametric_multi_window_tanh.py \
  loss_weights.phy=1.0 loss_weights.walls=1.0 loss_weights.windows=1.0 \
  loss_weights.doors=1.0 loss_weights.ic=1.0

# Halved collocation counts for faster iteration on a MacBook (MPS)
.venv/bin/python training/train_parametric_multi_window_tanh.py \
  points.interior=4000 points.walls=1000 points.windows_per_window=75 \
  points.doors=500 points.ic=1000
```

Sweeping to find what converges — `--multirun` is safe because run directories are auto-named:
```bash
# 1. Sweep depth x width (9 runs), short runs to compare early convergence slope
.venv/bin/python training/train_parametric_multi_window_tanh.py --multirun \
  arch.num_layers=4,5,6 arch.layer_size=64,128,256 \
  max_iters=3000 snapshot_every=1000

# 2. Sweep the activation function (4 runs)
.venv/bin/python training/train_parametric_multi_window_tanh.py --multirun \
  arch.activation=tanh,silu,gelu,sin \
  max_iters=3000 snapshot_every=1000

# 3. Compare the RAW (unweighted) losses across every run in the sweep
for d in outputs/$(date +%F)/parametric_multi_window_*_L*_W*/; do
  printf '%-52s %s\n' "$(basename "$d")" "$(tail -1 "$d/loss_history.csv")"
done
```
> Compare runs on the **raw** columns (`raw_phy`, `raw_walls`, `raw_windows`), not `weighted_total` — the raw values use formulas identical to 1.6, so they stay comparable both across weightings and against a 1.6 baseline run. Note that `--multirun` creates a `multirun/` directory at the repo root for Hydra's own job logs; you may want to add it to `.gitignore`.

Resuming works the same as every other trainer, and the architecture is validated on load — a depth, width or activation mismatch fails immediately with a clear message instead of a raw tensor-shape error:
```bash
.venv/bin/python training/train_parametric_multi_window_tanh.py \
  resume=outputs/2026-09-17/parametric_multi_window_tanh_L5_W128/model_checkpoint_10000.pth
```

### Resuming Training & Loading Checkpoints

All training pipelines support continuing an existing run or warm-starting a new run from a checkpoint:

1. **Continue / Resume Training (`resume=...` or `--resume`):**
   Restores model parameters, optimizer momentum, scheduler step progression, and continues training from `iteration + 1` up to `max_iters`:
   ```bash
   # Continue time-dependent PINN from checkpoint up to 30k iterations
   .venv/bin/python training/train_time_dependent.py resume=outputs/2026-09-14/time_dependent/model_checkpoint_10000.pth

   # Continue steady-state training
   .venv/bin/python training/train_steady.py resume=outputs/2026-09-14/model_checkpoint_10000.pth

   # Resume fine-tuning run
   .venv/bin/python training/finetune_initial_velocity.py --resume outputs/2026-09-15/finetuned_u0_0.0_-0.2_0.0/finetuned_model_final.pth --iterations 5000
   ```

2. **Warm-Start / Pretrained Model Initialization (`checkpoint=...` or `--checkpoint`):**
   Initializes model weights from an existing checkpoint but starts a fresh run at iteration 0 with brand new optimizer and learning rate states:
   ```bash
   # Initialize steady PINN weights from pretrained model but start at iteration 0
   .venv/bin/python training/train_steady.py checkpoint=outputs/2026-09-14/model_latest.pth max_iters=10000

   # Warm-start parametric PINN
   .venv/bin/python training/train_parametric.py checkpoint=outputs/2026-09-14/parametric/model_latest.pth
   ```

---

### Step 2: Inference & Animation Generation

The inference scripts evaluate the neural network over 3D spatial grids and particle systems, generating `.vtu` and `.pvd` collections.

---

#### 2.1 Generating Time-Dependent Visualizations ([inference/inference_time_dependent.py](inference/inference_time_dependent.py))

```bash
# 1. Spatiotemporal point probe at (x=7.0, y=4.5, z=1.5) at t=30s
.venv/bin/python inference/inference_time_dependent.py \
  --checkpoint outputs/2026-09-15/time_dependent/model_final.pth \
  --probe 7.0 4.5 1.5 30.0

# 2. Export 3D Physical Time-Lapse Grid (60 frames over 0 to 120s)
.venv/bin/python inference/inference_time_dependent.py \
  --checkpoint outputs/2026-09-15/time_dependent/model_final.pth \
  --animate timelapse \
  --frames 60 \
  --resolution 40 \
  --output-dir outputs/2026-09-15/time_dependent/animations

# 3. Export Unsteady Particle Pathlines (250 particles advected through time)
.venv/bin/python inference/inference_time_dependent.py \
  --checkpoint outputs/2026-09-15/time_dependent/model_final.pth \
  --animate particles \
  --frames 120 \
  --output-dir outputs/2026-09-15/time_dependent/animations

# 4. Export Both Time-Lapse and Particles at once
.venv/bin/python inference/inference_time_dependent.py \
  --checkpoint outputs/2026-09-15/time_dependent/model_final.pth \
  --animate all \
  --frames 60 \
  --resolution 40 \
  --output-dir outputs/2026-09-15/time_dependent/animations
```

- **What `timelapse` does:** Evaluates a $40\times 40\times 40$ grid (64,000 points) at discrete time steps $t_0, t_1, \dots, t_{59}$. Writes `timelapse/timelapse.pvd` mapping simulation seconds directly to ParaView's time slider.
- **What `particles` does:** Seeds 250 particles near the windows and numerically integrates forward in time: $\mathbf{x}(t + \Delta t) = \mathbf{x}(t) + \Delta t \cdot \mathbf{u}(\mathbf{x}(t), t)$. Writes `particles/particles.pvd`.

---

#### 2.2 Generating 5D Parametric Visualizations ([inference/inference_parametric.py](inference/inference_parametric.py))

```bash
# 1. Probe for V_inlet = 1.5 m/s at t = 20s
.venv/bin/python inference/inference_parametric.py \
  --checkpoint outputs/2026-09-15/parametric/model_final.pth \
  --probe 7.0 4.5 1.5 20.0 1.5

# 2. Export physical time-lapse and pathlines for V_inlet = 1.8 m/s
.venv/bin/python inference/inference_parametric.py \
  --checkpoint outputs/2026-09-15/parametric/model_final.pth \
  --velocity 1.8 \
  --animate all \
  --frames 60 \
  --resolution 40 \
  --output-dir outputs/2026-09-15/parametric/animations/v_1.80
```

> [!NOTE]
> **What does `v_1.80` mean in the output?**
> - `v_1.80` indicates that the simulation was evaluated with an **Inlet Velocity of $1.8\,\text{m/s}$** ($V_{\text{inlet}} = 1.8$).
> - Because the 5D Parametric model $(x, y, z, t, V_{\text{inlet}})$ was trained over a continuous range of window velocities ($V_{\text{inlet}} \in [0.2,\, 2.5]\,\text{m/s}$), you can query any desired inflow speed on the fly:
>   - `--velocity 0.5` $\longrightarrow$ outputs to `v_0.50/` (gentle breeze scenario)
>   - `--velocity 1.8` $\longrightarrow$ outputs to `v_1.80/` (moderate ventilation)
>   - `--velocity 2.5` $\longrightarrow$ outputs to `v_2.50/` (strong ventilation)
> - This allows you to compare multiple ventilation conditions side-by-side in ParaView using the exact same trained neural network without retraining.

---

#### 2.3 Generating 6D Parametric Occupancy Visualizations ([inference/inference_parametric_occupancy.py](inference/inference_parametric_occupancy.py))

```bash
# 1. Probe point (x=7.0, y=4.5, z=1.1) at t=30s for V_inlet = 1.5 m/s and N_people = 30
.venv/bin/python inference/inference_parametric_occupancy.py \
  --checkpoint outputs/2026-09-15/parametric_occupancy/model_final.pth \
  --probe 7.0 4.5 1.1 30.0 1.5 30.0

# 2. Export physical time-lapse for V_inlet = 1.2 m/s with N_people = 25 occupants
.venv/bin/python inference/inference_parametric_occupancy.py \
  --checkpoint outputs/2026-09-15/parametric_occupancy/model_final.pth \
  --velocity 1.2 \
  --occupancy 25.0 \
  --animate timelapse \
  --frames 60 \
  --resolution 40 \
  --output-dir outputs/2026-09-15/parametric_occupancy/animations

# 3. Export Occupancy Comparison Sweep comparing N_people = [0, 10, 20, 30, 40, 50]
.venv/bin/python inference/inference_parametric_occupancy.py \
  --checkpoint outputs/2026-09-15/parametric_occupancy/model_final.pth \
  --velocity 1.5 \
  --animate occupancy_sweep \
  --resolution 40 \
  --output-dir outputs/2026-09-15/parametric_occupancy/animations
```

---

#### 2.4 Generating Multi-Window Visualizations ([inference/inference_parametric_multi_window.py](inference/inference_parametric_multi_window.py))

Evaluate arbitrary combinations of independent window velocities across all 8 window panels:

```bash
# 1. Open specific windows (e.g., Windows 1, 4, and 8 at 1.5 m/s; others closed) with 2D slice
.venv/bin/python inference/inference_parametric_multi_window.py \
  --checkpoint outputs/2026-09-16/parametric_multi_window/model_final.pth \
  --open-windows 1 4 8 \
  --open-velocity 1.5 \
  --occupancy 25.0 \
  --time 60.0 \
  --save-slice

# 2. Specify independent velocities for all 8 windows explicitly:
.venv/bin/python inference/inference_parametric_multi_window.py \
  --checkpoint outputs/2026-09-16/parametric_multi_window/model_final.pth \
  --velocities 1.5 0.0 0.0 1.2 0.0 0.0 0.0 1.5 \
  --occupancy 30.0 \
  --time 30.0

# 3. Export full animation time-lapse sequence (PVD + VTU):
.venv/bin/python inference/inference_parametric_multi_window.py \
  --checkpoint outputs/2026-09-16/parametric_multi_window/model_final.pth \
  --open-windows 1 4 8 \
  --open-velocity 1.5 \
  --occupancy 30.0 \
  --timelapse \
  --n-frames 60
```

---

#### 2.5 Generating Tunable-Architecture Multi-Window Visualizations ([inference/inference_parametric_multi_window_tanh.py](inference/inference_parametric_multi_window_tanh.py))

Companion to the 1.7 trainer. The CLI is identical to 2.4; two things behave differently, and both matter:

- It **rebuilds the network with the activation recorded in the checkpoint.** Tanh, SiLU and GELU are parameterless, so a Tanh-trained checkpoint loads into a SiLU-built network with the same weights and *no error at all*, silently producing wrong fields. This script reads `model_config["activation"]` and rebuilds correctly.
- The **window count is read from the checkpoint** instead of being hardcoded to 8, so it works for any `Windows.stl` decomposition.

It stays backwards compatible: pointed at a pre-1.7 checkpoint it prints a note and falls back to PhysicsNeMo's `silu` default, which is the correct activation for those runs.

```bash
# 0. Point this at whichever run you want to evaluate
RUN=outputs/2026-09-17/parametric_multi_window_tanh_L5_W128

# 1. Open Windows 1, 4 and 8 at 1.5 m/s (others closed) and export a 2D slice PNG
.venv/bin/python inference/inference_parametric_multi_window_tanh.py \
  --checkpoint $RUN/model_final.pth \
  --open-windows 1 4 8 \
  --open-velocity 1.5 \
  --occupancy 25.0 \
  --time 60.0 \
  --save-slice

# 2. Specify independent velocities for all 8 windows explicitly
.venv/bin/python inference/inference_parametric_multi_window_tanh.py \
  --checkpoint $RUN/model_final.pth \
  --velocities 1.5 0.0 0.0 1.2 0.0 0.0 0.0 1.5 \
  --occupancy 30.0 \
  --time 30.0

# 3. Export the full animation time-lapse sequence (PVD + VTU) for ParaView
.venv/bin/python inference/inference_parametric_multi_window_tanh.py \
  --checkpoint $RUN/model_final.pth \
  --open-windows 1 4 8 \
  --open-velocity 1.5 \
  --occupancy 30.0 \
  --timelapse \
  --n-frames 60
```

On load it prints the architecture it reconstructed, so an unexpected checkpoint is obvious straight away rather than after a confusing field:
```text
Rebuilding FullyConnected(13 -> 5 x 128 [tanh] -> 5), skip_connections=False, weight_norm=False
Multi-window checkpoint with 8 windows (in_features=13).
```

Unlike 2.4, outputs default to an `inference/` directory **next to the checkpoint**, so results stay with the run that produced them (override with `--output-dir`). The ParaView-loadable artifacts are:

| Artifact | Path | Open in ParaView as |
| :--- | :--- | :--- |
| Inference time-lapse | `$RUN/inference/timelapse/timelapse_multi_window.pvd` | animated collection ($t = 0 \to 120\,\text{s}$) |
| Inference snapshot | `$RUN/inference/snapshot_t_60.0s.vtu` | single unstructured grid |
| Inference 2D slice | `$RUN/inference/slice_t_60.0s.png` | image, no ParaView needed |
| In-training validation snapshot | `$RUN/snapshots_iter_10000/multi_window_timelapse.pvd` | animated collection, 5 time slices |

All carry the same point-data arrays as every other exporter in this repo — `velocity_u`, `velocity_v`, `velocity_w`, `velocity_mag`, `pressure`, `pollutant_c` — so they render with the existing ParaView tooling in Step 3 unchanged:

```bash
# Render the inference time-lapse straight to an AVI, colored by pollutant concentration
/Applications/ParaView-6.1.1.app/Contents/bin/pvpython rendering/paraview_animate.py \
  --type timelapse \
  --input $RUN/inference/timelapse/timelapse_multi_window.pvd \
  --field pollutant_c \
  --format avi \
  --resolution 1920x1080

# Render the in-training validation snapshot as PNG frames, top-down camera
/Applications/ParaView-6.1.1.app/Contents/bin/pvpython rendering/paraview_animate.py \
  --type timelapse \
  --input $RUN/snapshots_iter_10000/multi_window_timelapse.pvd \
  --field velocity_mag \
  --camera top \
  --format png
```

---

#### 2.6 Generating Steady-State Visualizations ([inference/inference_steady.py](inference/inference_steady.py))

```bash
# Export 3D volume, steady streamlines, and spatial slice sweep
.venv/bin/python inference/inference_steady.py \
  --checkpoint outputs/2026-09-15/model_final.pth \
  --animate all \
  --anim-frames 120 \
  --resolution 50 \
  --anim-output-dir outputs/2026-09-15/steady_animations
```

---

### Step 3: Viewing & Rendering in ParaView

---

#### 3.1 Interactive Playback in ParaView GUI (Recommended)

1. Open **/Applications/ParaView-6.1.1.app**
2. Click **File → Open** $\to$ navigate to the generated `.pvd` file:
   - For Time-Lapse Grid: `outputs/2026-09-15/time_dependent/animations/timelapse/timelapse.pvd`
   - For Particle Pathlines: `outputs/2026-09-15/time_dependent/animations/particles/particles.pvd`
   - For Parametric: `outputs/2026-09-15/parametric/animations/v_1.80/timelapse_v_1.80/timelapse_v_1.80.pvd`
   - For Multi-Window inference: `outputs/2026-09-16/parametric_multi_window/inference/timelapse/timelapse_multi_window.pvd`
   - For Tunable-Architecture inference: `outputs/2026-09-17/parametric_multi_window_tanh_L5_W128/inference/timelapse/timelapse_multi_window.pvd`
   - For in-training validation snapshots: `outputs/2026-09-17/parametric_multi_window_tanh_L5_W128/snapshots_iter_10000/multi_window_timelapse.pvd`
3. In the **Properties** panel on the left, click **Apply**.
4. In the top toolbar, change **Solid Color** $\to$ **`velocity_mag`** or **`pollutant_c`**.
5. Press the **▶ Play** button in the animation toolbar at the top. ParaView will scrub through physical simulation seconds ($t = 0 \to 120\,\text{s}$).

---

#### 3.2 Headless Batch Rendering with `pvpython` ([rendering/paraview_animate.py](rendering/paraview_animate.py))

You can render full HD PNG image sequences or AVI videos directly from the command line:

```bash
# 1. Render all animations in a directory to PNG frame sequences
/Applications/ParaView-6.1.1.app/Contents/bin/pvpython rendering/paraview_animate.py \
  --type all \
  --input-dir outputs/2026-09-15/time_dependent/animations \
  --resolution 1920x1080

# 2. Render specific time-lapse PVD directly to AVI video
/Applications/ParaView-6.1.1.app/Contents/bin/pvpython rendering/paraview_animate.py \
  --type timelapse \
  --input outputs/2026-09-15/time_dependent/animations/timelapse/timelapse.pvd \
  --field velocity_mag \
  --format avi \
  --output-dir animation_renders

# 3. Render particle pathlines colored by pollutant concentration
/Applications/ParaView-6.1.1.app/Contents/bin/pvpython rendering/paraview_animate.py \
  --type particles \
  --input outputs/2026-09-15/time_dependent/animations/particles/particles.pvd \
  --field pollutant_c \
  --format png \
  --resolution 1920x1080

# 4. Change camera preset (isometric, front, top, side)
/Applications/ParaView-6.1.1.app/Contents/bin/pvpython rendering/paraview_animate.py \
  --type timelapse \
  --input outputs/2026-09-15/time_dependent/animations/timelapse/timelapse.pvd \
  --camera top \
  --format png
```

---

## 🛠️ Summary Table of Commands

| Task | Script | Primary Command |
| :--- | :--- | :--- |
| **Train Steady** | `training/train_steady.py` | `.venv/bin/python training/train_steady.py` |
| **Train Unsteady** | `training/train_time_dependent.py` | `.venv/bin/python training/train_time_dependent.py` |
| **Fine-Tune IC** | `training/finetune_initial_velocity.py` | `.venv/bin/python training/finetune_initial_velocity.py --checkpoint <ckpt> --initial-velocity 0 -0.2 0` |
| **Train Parametric** | `training/train_parametric.py` | `.venv/bin/python training/train_parametric.py` |
| **Train Occupancy** | `training/train_parametric_occupancy.py` | `.venv/bin/python training/train_parametric_occupancy.py` |
| **Train Multi-Window** | `training/train_parametric_multi_window.py` | `.venv/bin/python training/train_parametric_multi_window.py` |
| **Train Multi-Window (tunable arch)** | `training/train_parametric_multi_window_tanh.py` | `.venv/bin/python training/train_parametric_multi_window_tanh.py arch.num_layers=5 arch.layer_size=128 arch.activation=tanh` |
| **Unsteady Inference**| `inference/inference_time_dependent.py`| `.venv/bin/python inference/inference_time_dependent.py --checkpoint <ckpt> --animate all` |
| **Parametric Query** | `inference/inference_parametric.py` | `.venv/bin/python inference/inference_parametric.py --checkpoint <ckpt> --velocity 1.8 --animate all` |
| **Occupancy Query** | `inference/inference_parametric_occupancy.py` | `.venv/bin/python inference/inference_parametric_occupancy.py --checkpoint <ckpt> --velocity 1.5 --occupancy 30` |
| **Multi-Window Query**| `inference/inference_parametric_multi_window.py` | `.venv/bin/python inference/inference_parametric_multi_window.py --checkpoint <ckpt> --open-windows 1 4 8 --open-velocity 1.5` |
| **Multi-Window Query (tunable arch)**| `inference/inference_parametric_multi_window_tanh.py` | `.venv/bin/python inference/inference_parametric_multi_window_tanh.py --checkpoint <ckpt> --open-windows 1 4 8 --open-velocity 1.5 --timelapse` |
| **Render Video/PNG** | `rendering/paraview_animate.py` | `/Applications/ParaView-6.1.1.app/Contents/bin/pvpython rendering/paraview_animate.py --input <pvd>` |
