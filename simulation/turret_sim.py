"""
Autonomous Turret Tracking Simulation
======================================
Kalman Filter + PID Controller | Pan-Tilt Gimbal Model
Gauty — Turret Defense System | Week 1 Simulation

System:
  - Target moves in 3D space with injected randomness (unpredictable motion model)
  - Sensor: noisy angle measurements simulating camera centroid error
  - Kalman Filter: estimates target position + velocity from noisy observations
  - PID Controller: drives pan/tilt gimbal angles toward Kalman estimate
  - Visualization: real-time 3D matplotlib with telemetry HUD

Run:
  pip install numpy matplotlib
  python turret_sim.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

# ─────────────────────────────────────────────
#  SIMULATION PARAMETERS
# ─────────────────────────────────────────────
DT          = 0.05          # time step (s)  → 20 Hz sim
N_STEPS     = 600           # total frames
ARENA_SIZE  = 10.0          # meters, cubic arena half-extent

# Target motion
TARGET_SPEED      = 2.5     # base drift speed (m/s)
TARGET_JERK_STD   = 1.8     # std of random acceleration (unpredictability)
TARGET_SINUSOID   = True    # add sinusoidal component for figure-8 style motion

# Sensor noise (simulates camera centroid jitter → angle error in radians)
SENSOR_NOISE_STD  = 0.04    # radians (~2.3°)

# Kalman process noise
Q_POS   = 0.1
Q_VEL   = 2.0

# PID gains  (tuned for DT=0.05)
KP_PAN  = 3.5
KI_PAN  = 0.02
KD_PAN  = 0.8

KP_TILT = 3.5
KI_TILT = 0.02
KD_TILT = 0.8

# Gimbal limits
PAN_LIMIT  = np.radians(175)
TILT_LIMIT = np.radians(80)

# Trail lengths
TRAIL_TARGET  = 80
TRAIL_KF      = 60
TRAIL_GIMBAL  = 40

# ─────────────────────────────────────────────
#  KALMAN FILTER  (constant-velocity, 6-state)
#  State: [x, y, z, vx, vy, vz]
# ─────────────────────────────────────────────
class KalmanFilter3D:
    def __init__(self, dt):
        self.dt = dt
        n = 6
        self.F = np.eye(n)
        for i in range(3):
            self.F[i, i+3] = dt
        self.H = np.zeros((3, n))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        q_p, q_v = Q_POS, Q_VEL
        self.Q = np.diag([q_p, q_p, q_p, q_v, q_v, q_v]) * dt
        r = (SENSOR_NOISE_STD * 10) ** 2
        self.R = np.eye(3) * r
        self.x = np.zeros(n)
        self.P = np.eye(n) * 5.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    @property
    def pos(self):
        return self.x[:3]

    @property
    def vel(self):
        return self.x[3:]


# ─────────────────────────────────────────────
#  PID CONTROLLER
# ─────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd, dt, limit):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.limit = limit
        self._integral = 0.0
        self._prev_err = 0.0

    def step(self, error):
        self._integral += error * self.dt
        self._integral = np.clip(self._integral, -self.limit, self.limit)
        derivative = (error - self._prev_err) / self.dt
        self._prev_err = error
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return np.clip(output, -self.limit, self.limit)


# ─────────────────────────────────────────────
#  TARGET MOTION MODEL
# ─────────────────────────────────────────────
class Target:
    def __init__(self):
        self.pos = np.array([5.0, 3.0, 2.0])
        self.vel = np.array([-1.2, 0.8, 0.3])

    def step(self, t):
        accel = np.random.randn(3) * TARGET_JERK_STD * DT
        if TARGET_SINUSOID:
            accel[0] += 1.5 * np.sin(t * 0.6)
            accel[1] += 1.2 * np.cos(t * 0.4)
            accel[2] += 0.6 * np.sin(t * 0.9)
        self.vel += accel * DT
        speed = np.linalg.norm(self.vel)
        if speed > TARGET_SPEED * 2.5:
            self.vel = self.vel / speed * TARGET_SPEED * 2.5
        self.pos += self.vel * DT
        for i in range(3):
            lo = 1.0 if i == 2 else -ARENA_SIZE
            hi = ARENA_SIZE if i != 2 else ARENA_SIZE * 0.6
            if self.pos[i] < lo:
                self.pos[i] = lo
                self.vel[i] *= -0.8
            if self.pos[i] > hi:
                self.pos[i] = hi
                self.vel[i] *= -0.8
        return self.pos.copy()


# ─────────────────────────────────────────────
#  GIMBAL MODEL
# ─────────────────────────────────────────────
class Gimbal:
    def __init__(self):
        self.pan  = 0.0
        self.tilt = 0.0
        self.BARREL_LEN = 4.0

    def aim_at(self, target_pos):
        x, y, z = target_pos
        pan  = np.arctan2(y, x)
        dist_xy = np.sqrt(x**2 + y**2)
        tilt = np.arctan2(z, dist_xy)
        return pan, tilt

    def pointing_vector(self):
        cp, sp = np.cos(self.pan),  np.sin(self.pan)
        ct, st = np.cos(self.tilt), np.sin(self.tilt)
        return np.array([cp * ct, sp * ct, st])

    def barrel_end(self):
        return self.pointing_vector() * self.BARREL_LEN


# ─────────────────────────────────────────────
#  ANGULAR ERROR
# ─────────────────────────────────────────────
def angle_error(desired, current):
    err = desired - current
    while err >  np.pi: err -= 2 * np.pi
    while err < -np.pi: err += 2 * np.pi
    return err


# ─────────────────────────────────────────────
#  SIMULATION STATE
# ─────────────────────────────────────────────
target  = Target()
kf      = KalmanFilter3D(DT)
gimbal  = Gimbal()
pid_pan  = PID(KP_PAN,  KI_PAN,  KD_PAN,  DT, PAN_LIMIT)
pid_tilt = PID(KP_TILT, KI_TILT, KD_TILT, DT, TILT_LIMIT)

kf.x[:3] = target.pos.copy()

trail_target = deque(maxlen=TRAIL_TARGET)
trail_kf     = deque(maxlen=TRAIL_KF)
trail_gimbal = deque(maxlen=TRAIL_GIMBAL)

t_history       = deque(maxlen=N_STEPS)
pan_err_hist    = deque(maxlen=N_STEPS)
tilt_err_hist   = deque(maxlen=N_STEPS)
kf_pos_err_hist = deque(maxlen=N_STEPS)
pan_hist        = deque(maxlen=N_STEPS)
tilt_hist       = deque(maxlen=N_STEPS)

t_sim = 0.0

# ─────────────────────────────────────────────
#  FIGURE SETUP
# ─────────────────────────────────────────────
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 9), facecolor='#0a0a0a')
fig.suptitle('AUTONOMOUS TURRET — KALMAN + PID TRACKING SIM',
             color='#c9b8a0', fontsize=11, fontweight='bold',
             fontfamily='monospace', y=0.97)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       left=0.04, right=0.97,
                       top=0.93, bottom=0.06,
                       wspace=0.35, hspace=0.55)

ax3d = fig.add_subplot(gs[:, :2], projection='3d')
ax_pan  = fig.add_subplot(gs[0, 2])
ax_tilt = fig.add_subplot(gs[1, 2])
ax_err  = fig.add_subplot(gs[2, 2])

ACCENT = '#c9b8a0'
RED    = '#e05252'
CYAN   = '#52b8e0'
GREEN  = '#7ecb8f'
GOLD   = '#d4a84b'
DIM    = '#444444'

def style_ax(ax, title):
    ax.set_facecolor('#0f0f0f')
    ax.tick_params(colors=DIM, labelsize=7)
    ax.set_title(title, color=ACCENT, fontsize=7.5, fontfamily='monospace', pad=4)
    for spine in ax.spines.values():
        spine.set_edgecolor('#222222')

for ax, ttl in [(ax_pan,  'PAN ANGLE  (rad)'),
                (ax_tilt, 'TILT ANGLE  (rad)'),
                (ax_err,  'ANGULAR TRACKING ERROR  (rad)')]:
    style_ax(ax, ttl)
    ax.set_xlim(0, N_STEPS * DT)
    ax.grid(alpha=0.12, color='#333333')

ax3d.set_facecolor('#080808')
ax3d.set_xlim(-ARENA_SIZE, ARENA_SIZE)
ax3d.set_ylim(-ARENA_SIZE, ARENA_SIZE)
ax3d.set_zlim(0, ARENA_SIZE)
ax3d.set_xlabel('X (m)', color=DIM, fontsize=7, labelpad=2)
ax3d.set_ylabel('Y (m)', color=DIM, fontsize=7, labelpad=2)
ax3d.set_zlabel('Z (m)', color=DIM, fontsize=7, labelpad=2)
ax3d.tick_params(colors='#333333', labelsize=6)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor('#1a1a1a')
ax3d.yaxis.pane.set_edgecolor('#1a1a1a')
ax3d.zaxis.pane.set_edgecolor('#1a1a1a')
ax3d.grid(True, alpha=0.08, color='#222222')

ln_tgt,  = ax3d.plot([], [], [], color=RED,  alpha=0.5, lw=1.2, label='Target path')
ln_kf,   = ax3d.plot([], [], [], color=CYAN, alpha=0.6, lw=1.0, linestyle='--', label='Kalman estimate')
ln_aim,  = ax3d.plot([], [], [], color=GOLD, alpha=0.35, lw=0.8, label='Gimbal trail')

pt_tgt  = ax3d.scatter([], [], [], c=RED,       s=80, depthshade=False, zorder=5)
pt_kf   = ax3d.scatter([], [], [], c=CYAN,      s=40, depthshade=False, zorder=4)
pt_meas = ax3d.scatter([], [], [], c='#ffffff', s=15, alpha=0.4, depthshade=False, label='Noisy measurements', zorder=3)

barrel_line, = ax3d.plot([], [], [], color=GREEN, lw=2.5, solid_capstyle='round')
turret_base  = ax3d.scatter([0], [0], [0], c=GREEN, s=120, marker='^', depthshade=False, zorder=6)
lock_line,   = ax3d.plot([], [], [], color=GOLD, lw=0.8, linestyle=':', alpha=0.7)

ax3d.legend(loc='upper left', fontsize=7, facecolor='#111111',
            edgecolor='#333333', labelcolor=ACCENT, framealpha=0.8)

ln_pan_cmd,  = ax_pan.plot([], [],  color=GREEN,  lw=1.0, label='Cmd')
ln_pan_des,  = ax_pan.plot([], [],  color=ACCENT, lw=0.8, linestyle='--', label='Desired', alpha=0.7)
ax_pan.legend(fontsize=6, facecolor='#0f0f0f', edgecolor='#333333', labelcolor=ACCENT)

ln_tilt_cmd, = ax_tilt.plot([], [], color=GREEN,  lw=1.0, label='Cmd')
ln_tilt_des, = ax_tilt.plot([], [], color=ACCENT, lw=0.8, linestyle='--', label='Desired', alpha=0.7)
ax_tilt.legend(fontsize=6, facecolor='#0f0f0f', edgecolor='#333333', labelcolor=ACCENT)

ln_pan_err,  = ax_err.plot([], [], color=CYAN, lw=1.0, label='Pan err')
ln_tilt_err, = ax_err.plot([], [], color=RED,  lw=1.0, label='Tilt err')
ln_kf_err,   = ax_err.plot([], [], color=GOLD, lw=0.8, linestyle=':', label='KF pos err/10', alpha=0.8)
ax_err.legend(fontsize=6, facecolor='#0f0f0f', edgecolor='#333333', labelcolor=ACCENT)
ax_err.axhline(0, color='#333333', lw=0.5)

hud = ax3d.text2D(0.01, 0.99, '', transform=ax3d.transAxes,
                  color=ACCENT, fontsize=7.5, fontfamily='monospace',
                  va='top', ha='left',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#0a0a0a',
                            edgecolor='#333333', alpha=0.85))

meas_buf = deque(maxlen=25)


# ─────────────────────────────────────────────
#  UPDATE FUNCTION
# ─────────────────────────────────────────────
def update(frame):
    global t_sim

    true_pos = target.step(t_sim)

    noise = np.random.randn(3) * SENSOR_NOISE_STD * np.linalg.norm(true_pos) * 0.15
    measured_pos = true_pos + noise
    meas_buf.append(measured_pos.copy())

    kf.predict()
    kf.update(measured_pos)
    kf_pos = kf.pos.copy()

    des_pan, des_tilt = gimbal.aim_at(kf_pos)

    err_pan  = angle_error(des_pan,  gimbal.pan)
    err_tilt = angle_error(des_tilt, gimbal.tilt)

    gimbal.pan  += pid_pan.step(err_pan)   * DT
    gimbal.tilt += pid_tilt.step(err_tilt) * DT
    gimbal.pan  = np.clip(gimbal.pan,  -PAN_LIMIT,  PAN_LIMIT)
    gimbal.tilt = np.clip(gimbal.tilt, -TILT_LIMIT, TILT_LIMIT)

    trail_target.append(true_pos.copy())
    trail_kf.append(kf_pos.copy())
    trail_gimbal.append(gimbal.barrel_end().copy())

    t_history.append(t_sim)
    pan_err_hist.append(err_pan)
    tilt_err_hist.append(err_tilt)
    kf_pos_err_hist.append(np.linalg.norm(kf_pos - true_pos) / 10.0)
    pan_hist.append(gimbal.pan)
    tilt_hist.append(gimbal.tilt)

    tgt_arr  = np.array(trail_target)
    kf_arr   = np.array(trail_kf)
    aim_arr  = np.array(trail_gimbal)
    meas_arr = np.array(meas_buf) if meas_buf else np.zeros((1, 3))

    ln_tgt.set_data(tgt_arr[:, 0], tgt_arr[:, 1])
    ln_tgt.set_3d_properties(tgt_arr[:, 2])
    ln_kf.set_data(kf_arr[:, 0], kf_arr[:, 1])
    ln_kf.set_3d_properties(kf_arr[:, 2])
    ln_aim.set_data(aim_arr[:, 0], aim_arr[:, 1])
    ln_aim.set_3d_properties(aim_arr[:, 2])

    pt_tgt._offsets3d  = ([true_pos[0]], [true_pos[1]], [true_pos[2]])
    pt_kf._offsets3d   = ([kf_pos[0]],  [kf_pos[1]],  [kf_pos[2]])
    pt_meas._offsets3d = (meas_arr[:, 0], meas_arr[:, 1], meas_arr[:, 2])

    be = gimbal.barrel_end()
    barrel_line.set_data([0, be[0]], [0, be[1]])
    barrel_line.set_3d_properties([0, be[2]])
    lock_line.set_data([0, kf_pos[0]], [0, kf_pos[1]])
    lock_line.set_3d_properties([0, kf_pos[2]])

    t_arr = np.array(t_history)
    ax_pan.set_xlim(max(0, t_sim - 15), t_sim + 1)
    ax_tilt.set_xlim(max(0, t_sim - 15), t_sim + 1)
    ax_err.set_xlim(max(0, t_sim - 15), t_sim + 1)

    ln_pan_cmd.set_data(t_arr, np.array(pan_hist))
    ln_tilt_cmd.set_data(t_arr, np.array(tilt_hist))

    kf_pans  = [gimbal.aim_at(p)[0] for p in kf_arr[-len(t_arr):]]
    kf_tilts = [gimbal.aim_at(p)[1] for p in kf_arr[-len(t_arr):]]
    ln_pan_des.set_data(t_arr[-len(kf_pans):], kf_pans)
    ln_tilt_des.set_data(t_arr[-len(kf_tilts):], kf_tilts)

    ln_pan_err.set_data(t_arr, np.array(pan_err_hist))
    ln_tilt_err.set_data(t_arr, np.array(tilt_err_hist))
    ln_kf_err.set_data(t_arr, np.array(kf_pos_err_hist))

    for ax in [ax_pan, ax_tilt]:
        ax.relim()
        ax.autoscale_view()
    ax_err.set_ylim(-0.6, 0.6)

    ang_err_total = np.degrees(np.sqrt(err_pan**2 + err_tilt**2))
    lock_status = "LOCKED ON" if ang_err_total < 5.0 else "TRACKING.."
    kf_speed = np.linalg.norm(kf.vel)

    hud.set_text(
        f"T = {t_sim:6.2f} s    FRAME {frame:04d}\n"
        f"─────────────────────\n"
        f"TARGET  [{true_pos[0]:+5.1f}, {true_pos[1]:+5.1f}, {true_pos[2]:+5.1f}]\n"
        f"KF EST  [{kf_pos[0]:+5.1f}, {kf_pos[1]:+5.1f}, {kf_pos[2]:+5.1f}]\n"
        f"KF VEL  [{kf.vel[0]:+4.1f}, {kf.vel[1]:+4.1f}, {kf.vel[2]:+4.1f}]\n"
        f"─────────────────────\n"
        f"PAN     {np.degrees(gimbal.pan):+6.1f}°\n"
        f"TILT    {np.degrees(gimbal.tilt):+6.1f}°\n"
        f"ANG ERR {ang_err_total:5.2f}°\n"
        f"KF SPD  {kf_speed:5.2f} m/s\n"
        f"─────────────────────\n"
        f"STATUS  {lock_status}"
    )

    t_sim += DT
    return (ln_tgt, ln_kf, ln_aim, pt_tgt, pt_kf, pt_meas,
            barrel_line, lock_line,
            ln_pan_cmd, ln_pan_des, ln_tilt_cmd, ln_tilt_des,
            ln_pan_err, ln_tilt_err, ln_kf_err, hud)


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────
ani = FuncAnimation(
    fig, update,
    frames=N_STEPS,
    interval=int(DT * 1000),
    blit=False,
    repeat=False
)

plt.show()
