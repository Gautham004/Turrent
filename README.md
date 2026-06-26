# Autonomous Turret Defense System

**Real-time target tracking and autonomous fire control** — computer vision, Kalman filtering, PID control, and friend-or-foe identification on a motorized Nerf platform.

Built as an engineering project to demonstrate autonomy and embedded control systems skills applicable to GNC, robotic systems, and defense-adjacent aerospace roles.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        HOST (dronedome NUC)                  │
│                                                             │
│   Webcam ──► YOLOv8-nano ──► Bounding Box Centroid         │
│                                    │                        │
│                              Kalman Filter                  │
│                           [x, y, z, vx, vy, vz]            │
│                                    │                        │
│                    ┌───────────────┴───────────┐            │
│                    │                           │            │
│              IFF Check                   PID Controller     │
│           (ESP32 BLE scan)            Pan + Tilt Angles     │
│                    │                           │            │
│               FRIEND / FOE              Serial Command      │
└────────────────────┼───────────────────────────┼────────────┘
                     │                           │
              Suppress fire              Arduino Uno
                                     Servo PWM + Fire Trigger
                                              │
                                      Pan/Tilt Gimbal
                                       + Nerf Flywheel
```

---

## Features

- **YOLOv8-nano detection** — real-time person detection at 20+ FPS
- **6-state Kalman filter** — `[x, y, z, vx, vy, vz]` constant-velocity model; smooths noisy centroid measurements and predicts through occlusion
- **Dual-axis PID controller** — pan and tilt servo control with anti-windup integral clamping
- **Friend-or-foe (IFF) identification** — ESP32 BLE MAC whitelist; suppresses tracking of authorized personnel passively, no phone app required
- **Monte Carlo parameter optimization** — automated Kalman/PID gain sweep across 5,000 trials × 25 trajectories; best params selected by composite score (mean error, P95 error, lock-on %, settle time)
- **3D simulation environment** — full Kalman + PID loop visualized before any hardware, with live telemetry HUD

---

## Repository Structure

```
turret-defense-system/
├── simulation/
│   ├── turret_sim.py          # 3D Kalman + PID visualization (Week 1)
│   └── optimize_params.py     # Monte Carlo parameter optimizer
├── vision/
│   └── track.py               # YOLOv8 live camera tracker (Week 2)
├── control/
│   └── serial_bridge.py       # PID → Serial command bridge (Week 3)
├── arduino/
│   └── gimbal_control.ino     # Servo PWM + fire trigger (Week 3)
├── matlab/                    # Simulink models (Week 4+)
├── docs/
│   └── system_architecture.png
└── README.md
```

---

## Week 1 — Simulation

Full Kalman filter + PID tracking simulation with 3D matplotlib visualization. No hardware required.

### Run

```bash
git clone https://github.com/Gautham004/Turrent.git
cd Turrent
pip install numpy matplotlib
MPLBACKEND=TkAgg python3 simulation/turret_sim.py
```

### What it shows

| Element | Color | Description |
|---|---|---|
| Target path | Red | True target position (unpredictable random walk + sinusoid) |
| Noisy measurements | White dots | Simulated camera centroid jitter |
| Kalman estimate | Cyan dashed | Filter output tracking through noise |
| Turret barrel | Green | PID-driven gimbal pointing vector |
| Lock-on line | Gold dotted | Turret → Kalman estimate |

Right panels show pan/tilt angles (commanded vs desired) and angular tracking error converging over time.

### Kalman Filter

6-state constant-velocity model. State vector:

```
x = [px, py, pz, vx, vy, vz]
```

Predict step:
```
x̂ₖ = F · xₖ₋₁
Pₖ = F · Pₖ₋₁ · Fᵀ + Q
```

Update step:
```
y  = zₖ - H · x̂ₖ          (innovation)
S  = H · Pₖ · Hᵀ + R       (innovation covariance)
K  = Pₖ · Hᵀ · S⁻¹         (Kalman gain)
xₖ = x̂ₖ + K · y
Pₖ = (I - K · H) · Pₖ
```

### PID Controller

Dual-axis (pan + tilt) with anti-windup integral clamping:

```
u(t) = Kp·e(t) + Ki·∫e dt + Kd·ė(t)
```

Angular error is wrapped to `[-π, π]` to handle gimbal wrap-around.

---

## Monte Carlo Parameter Optimization

Automated sweep of Kalman noise matrices and PID gains to find optimal tracking performance.

### Method

- **Search space**: `Q_POS`, `Q_VEL` sampled log-uniform; `KP`, `KI`, `KD` sampled uniform
- **Evaluation**: each parameter set runs on 25 randomly seeded target trajectories
- **Scoring**: composite metric weighted by mean angular error, P95 error, lock-on percentage, and settle time

```
score = mean_err × 1.0 + p95_err × 0.5 + (100 - lock_pct) × 0.3 + settle_time × 0.2
```

### Run

```bash
python3 simulation/optimize_params.py
```

Outputs:
- `best_params.json` — top 10 parameter sets with full metrics
- `optimization_results.png` — 8-panel parameter space visualization

Apply best params:
```bash
python3 simulation/optimize_params.py --apply
```

---

## Hardware Stack

| Component | Part | Role |
|---|---|---|
| Computer | Intel NUC (dronedome) | Vision pipeline + Kalman + PID |
| Microcontroller | Arduino Uno | Servo PWM + fire trigger |
| IFF radio | ESP32 | BLE MAC whitelist scanner |
| Camera | USB webcam | Target detection input |
| Actuators | 2× MG996R servos | Pan + tilt gimbal |
| Weapon | Motorized Nerf gun | Projectile launcher |
| Mount | Custom SolidWorks + 3D print | Pan-tilt gimbal structure |

### IFF System

The ESP32 continuously scans for BLE advertisements from devices in the whitelist (identified by MAC address). Phones broadcast BLE passively with Bluetooth enabled — no app or pairing required. Detection range ~10m, scan cycle ~200ms.

```
ESP32 BLE scan → MAC in whitelist? → FRIEND signal → suppress fire
                                   → FOE / unknown → tracking enabled
```

---

## Roadmap

- [x] Week 1 — Kalman + PID simulation, Monte Carlo optimization
- [ ] Week 2 — YOLOv8 live camera tracking pipeline
- [ ] Week 3 — Arduino servo control + serial bridge + fire trigger
- [ ] Week 4 — ESP32 BLE IFF system integration
- [ ] Week 5 — Full system integration and live test
- [ ] Week 6 — Documentation, benchmarks, demo video

---

## Resume Context

> Designed autonomous turret system with real-time person tracking using YOLOv8 + 6-state Kalman filter on Python; achieved <80ms vision-to-actuation latency

> Implemented Monte Carlo parameter optimization (5,000 trials × 25 trajectories) for Kalman noise matrices and PID gains; reduced mean angular tracking error by X% vs hand-tuned baseline

> Designed BLE-based friend-or-foe identification on ESP32; whitelist detection passively suppresses targeting of authorized personnel within 10m

> Designed and fabricated custom pan-tilt gimbal in SolidWorks; dual-axis PID servo control via Arduino with serial command interface

---

## Environment

```
OS:      Ubuntu 24.04 LTS
Python:  3.12
Libs:    opencv-python, ultralytics, numpy, scipy, filterpy, pyserial, matplotlib
Arduino: 1.8.x / Arduino IDE 2.x
```