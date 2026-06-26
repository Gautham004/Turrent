"""
Turret Tracking — Monte Carlo Parameter Optimizer
===================================================
Kalman Filter + PID Gain Sweep | Automated Accuracy Study
Gauty — Turret Defense System

What this does:
  - Runs N_TRIALS simulations with randomly sampled Kalman/PID parameters
  - Each trial uses the same target trajectory (seeded) for fair comparison
  - Scores each run: mean angular error, lock-on %, settling time
  - Saves top-K parameter sets to JSON
  - Plots heatmaps and scatter plots of the parameter space

Run:
  python optimize_params.py

Output:
  best_params.json        — top 10 parameter sets
  optimization_results.png — full visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import json
import time

# ─────────────────────────────────────────────
#  SEARCH SPACE
# ─────────────────────────────────────────────
SEARCH_SPACE = {
    "Q_POS":  (0.01, 2.0),
    "Q_VEL":  (0.5,  8.0),
    "KP":     (1.0,  8.0),
    "KI":     (0.0,  0.1),
    "KD":     (0.1,  2.0),
}

N_TRIALS        = 10000
N_STEPS         = 1000
DT              = 0.05
TOP_K           = 10
TRAJ_SEED       = 42
N_TRAJECTORIES  = 50

ARENA_SIZE       = 10.0
TARGET_SPEED     = 2.5
TARGET_JERK_STD  = 1.8
SENSOR_NOISE_STD = 0.04
PAN_LIMIT        = np.radians(175)
TILT_LIMIT       = np.radians(80)


def run_sim(params, traj_seed=42):
    rng = np.random.RandomState(traj_seed)
    Q_POS = params["Q_POS"]
    Q_VEL = params["Q_VEL"]
    KP    = params["KP"]
    KI    = params["KI"]
    KD    = params["KD"]

    n = 6
    F = np.eye(n)
    for i in range(3):
        F[i, i+3] = DT
    H = np.zeros((3, n))
    H[0,0] = H[1,1] = H[2,2] = 1.0
    Q = np.diag([Q_POS]*3 + [Q_VEL]*3) * DT
    R = np.eye(3) * (SENSOR_NOISE_STD * 10) ** 2
    kf_x = np.array([5.0, 3.0, 2.0, -1.2, 0.8, 0.3])
    kf_P = np.eye(n) * 5.0

    tgt_pos = np.array([5.0, 3.0, 2.0])
    tgt_vel = np.array([-1.2, 0.8, 0.3])
    pan  = 0.0
    tilt = 0.0
    pan_integral  = 0.0
    tilt_integral = 0.0
    pan_prev_err  = 0.0
    tilt_prev_err = 0.0

    ang_errors       = []
    locked_frames    = 0
    settled_frame    = None
    LOCK_THRESHOLD   = np.radians(5.0)
    SETTLE_WINDOW    = 20
    consecutive_locked = 0

    for step in range(N_STEPS):
        t = step * DT
        accel = rng.randn(3) * TARGET_JERK_STD * DT
        accel[0] += 1.5 * np.sin(t * 0.6)
        accel[1] += 1.2 * np.cos(t * 0.4)
        accel[2] += 0.6 * np.sin(t * 0.9)
        tgt_vel += accel * DT
        speed = np.linalg.norm(tgt_vel)
        if speed > TARGET_SPEED * 2.5:
            tgt_vel = tgt_vel / speed * TARGET_SPEED * 2.5
        tgt_pos += tgt_vel * DT
        for i in range(3):
            lo = 1.0 if i == 2 else -ARENA_SIZE
            hi = ARENA_SIZE if i != 2 else ARENA_SIZE * 0.6
            if tgt_pos[i] < lo: tgt_pos[i] = lo; tgt_vel[i] *= -0.8
            if tgt_pos[i] > hi: tgt_pos[i] = hi; tgt_vel[i] *= -0.8

        noise = rng.randn(3) * SENSOR_NOISE_STD * np.linalg.norm(tgt_pos) * 0.15
        meas  = tgt_pos + noise

        kf_x = F @ kf_x
        kf_P = F @ kf_P @ F.T + Q
        y = meas - H @ kf_x
        S = H @ kf_P @ H.T + R
        K = kf_P @ H.T @ np.linalg.inv(S)
        kf_x = kf_x + K @ y
        kf_P = (np.eye(n) - K @ H) @ kf_P
        kf_pos = kf_x[:3]

        x, y_pos, z = kf_pos
        des_pan  = np.arctan2(y_pos, x)
        dist_xy  = np.sqrt(x**2 + y_pos**2)
        des_tilt = np.arctan2(z, dist_xy)

        def wrap(e):
            while e >  np.pi: e -= 2*np.pi
            while e < -np.pi: e += 2*np.pi
            return e

        err_pan  = wrap(des_pan  - pan)
        err_tilt = wrap(des_tilt - tilt)

        pan_integral  = np.clip(pan_integral  + err_pan  * DT, -PAN_LIMIT,  PAN_LIMIT)
        tilt_integral = np.clip(tilt_integral + err_tilt * DT, -TILT_LIMIT, TILT_LIMIT)
        pan_d  = (err_pan  - pan_prev_err)  / DT
        tilt_d = (err_tilt - tilt_prev_err) / DT

        pan  += np.clip(KP*err_pan  + KI*pan_integral  + KD*pan_d,  -PAN_LIMIT,  PAN_LIMIT) * DT
        tilt += np.clip(KP*err_tilt + KI*tilt_integral + KD*tilt_d, -TILT_LIMIT, TILT_LIMIT) * DT
        pan  = np.clip(pan,  -PAN_LIMIT,  PAN_LIMIT)
        tilt = np.clip(tilt, -TILT_LIMIT, TILT_LIMIT)
        pan_prev_err  = err_pan
        tilt_prev_err = err_tilt

        ang_err = np.sqrt(err_pan**2 + err_tilt**2)
        ang_errors.append(ang_err)
        if ang_err < LOCK_THRESHOLD:
            locked_frames += 1
            consecutive_locked += 1
            if consecutive_locked >= SETTLE_WINDOW and settled_frame is None:
                settled_frame = step
        else:
            consecutive_locked = 0

    ang_errors = np.array(ang_errors)
    return {
        "mean_err_deg":   float(np.degrees(np.mean(ang_errors))),
        "median_err_deg": float(np.degrees(np.median(ang_errors))),
        "p95_err_deg":    float(np.degrees(np.percentile(ang_errors, 95))),
        "lock_pct":       float(locked_frames / N_STEPS * 100),
        "settle_frame":   int(settled_frame) if settled_frame else N_STEPS,
        "settle_time_s":  float(settled_frame * DT) if settled_frame else float(N_STEPS * DT),
    }


def score(metrics):
    return (
        metrics["mean_err_deg"]     * 1.0 +
        metrics["p95_err_deg"]      * 0.5 +
        (100 - metrics["lock_pct"]) * 0.3 +
        metrics["settle_time_s"]    * 0.2
    )


def random_search():
    rng = np.random.RandomState(0)
    results = []
    print(f"\n{'─'*60}")
    print(f"  TURRET OPTIMIZER — Monte Carlo Parameter Search")
    print(f"  {N_TRIALS} trials x {N_TRAJECTORIES} trajectories x {N_STEPS} steps")
    print(f"{'─'*60}\n")
    t0 = time.time()

    for trial in range(N_TRIALS):
        params = {
            "Q_POS": float(10 ** rng.uniform(np.log10(SEARCH_SPACE["Q_POS"][0]),
                                              np.log10(SEARCH_SPACE["Q_POS"][1]))),
            "Q_VEL": float(10 ** rng.uniform(np.log10(SEARCH_SPACE["Q_VEL"][0]),
                                              np.log10(SEARCH_SPACE["Q_VEL"][1]))),
            "KP":    float(rng.uniform(*SEARCH_SPACE["KP"])),
            "KI":    float(rng.uniform(*SEARCH_SPACE["KI"])),
            "KD":    float(rng.uniform(*SEARCH_SPACE["KD"])),
        }

        trial_metrics = []
        for seed in range(TRAJ_SEED, TRAJ_SEED + N_TRAJECTORIES):
            trial_metrics.append(run_sim(params, traj_seed=seed))

        avg_metrics = {k: float(np.mean([m[k] for m in trial_metrics]))
                       for k in trial_metrics[0]}
        s = score(avg_metrics)
        results.append({"trial": trial, "params": params, "metrics": avg_metrics, "score": s})

        if (trial + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (trial+1) * (N_TRIALS - trial - 1)
            best = min(results, key=lambda r: r["score"])
            print(f"  Trial {trial+1:4d}/{N_TRIALS} | "
                  f"Best score: {best['score']:6.2f} | "
                  f"Mean err: {best['metrics']['mean_err_deg']:5.2f}deg | "
                  f"Lock: {best['metrics']['lock_pct']:5.1f}% | "
                  f"ETA: {eta:.0f}s")

    print(f"\n  Done in {time.time()-t0:.1f}s")
    return results


def plot_results(results):
    results_sorted = sorted(results, key=lambda r: r["score"])
    top   = results_sorted[:TOP_K]
    best  = results_sorted[0]

    scores   = [r["score"]                    for r in results]
    mean_err = [r["metrics"]["mean_err_deg"]  for r in results]
    lock_pct = [r["metrics"]["lock_pct"]      for r in results]
    kp_vals  = [r["params"]["KP"]             for r in results]
    kd_vals  = [r["params"]["KD"]             for r in results]
    qpos     = [r["params"]["Q_POS"]          for r in results]
    qvel     = [r["params"]["Q_VEL"]          for r in results]
    p95      = [r["metrics"]["p95_err_deg"]   for r in results]

    plt.style.use('dark_background')
    ACCENT = '#c9b8a0'
    CYAN   = '#52b8e0'
    GREEN  = '#7ecb8f'
    GOLD   = '#d4a84b'
    DIM    = '#444444'
    cmap   = LinearSegmentedColormap.from_list('turret', ['#e05252','#d4a84b','#7ecb8f'], N=256)

    fig = plt.figure(figsize=(18, 11), facecolor='#0a0a0a')
    fig.suptitle('TURRET OPTIMIZER — MONTE CARLO PARAMETER SWEEP',
                 color=ACCENT, fontsize=12, fontweight='bold', fontfamily='monospace', y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           left=0.06, right=0.97, top=0.93, bottom=0.07,
                           wspace=0.38, hspace=0.55)

    def style(ax, title, xlabel='', ylabel=''):
        ax.set_facecolor('#0d0d0d')
        ax.set_title(title, color=ACCENT, fontsize=8, fontfamily='monospace', pad=5)
        ax.set_xlabel(xlabel, color=DIM, fontsize=7)
        ax.set_ylabel(ylabel, color=DIM, fontsize=7)
        ax.tick_params(colors=DIM, labelsize=6)
        for spine in ax.spines.values(): spine.set_edgecolor('#222222')
        ax.grid(alpha=0.1, color='#333333')

    ax1 = fig.add_subplot(gs[0, 0])
    style(ax1, 'SCORE DISTRIBUTION', 'Composite Score', 'Count')
    ax1.hist(scores, bins=40, color=CYAN, alpha=0.7, edgecolor='none')
    ax1.axvline(best["score"], color=GREEN, lw=1.5, linestyle='--')

    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2, 'ERROR vs LOCK-ON %', 'Mean Angular Error (deg)', 'Lock-On %')
    sc2 = ax2.scatter(mean_err, lock_pct, c=scores, cmap=cmap, s=8, alpha=0.6, linewidths=0)
    ax2.scatter(best["metrics"]["mean_err_deg"], best["metrics"]["lock_pct"],
                c=GREEN, s=80, marker='*', zorder=5, label='Best')
    ax2.legend(fontsize=6, facecolor='#111111', edgecolor='#333333', labelcolor=ACCENT)
    plt.colorbar(sc2, ax=ax2, label='Score').ax.tick_params(labelsize=6, colors=DIM)

    ax3 = fig.add_subplot(gs[0, 2])
    style(ax3, 'KP vs KD SPACE', 'KP (Proportional)', 'KD (Derivative)')
    sc3 = ax3.scatter(kp_vals, kd_vals, c=mean_err, cmap=cmap, s=8, alpha=0.6, linewidths=0)
    ax3.scatter(best["params"]["KP"], best["params"]["KD"], c=GREEN, s=80, marker='*', zorder=5)
    plt.colorbar(sc3, ax=ax3, label='Mean Err (deg)').ax.tick_params(labelsize=6, colors=DIM)

    ax4 = fig.add_subplot(gs[0, 3])
    style(ax4, 'KALMAN NOISE SPACE', 'Q_POS (log)', 'Q_VEL (log)')
    sc4 = ax4.scatter(np.log10(qpos), np.log10(qvel), c=mean_err, cmap=cmap, s=8, alpha=0.6, linewidths=0)
    ax4.scatter(np.log10(best["params"]["Q_POS"]), np.log10(best["params"]["Q_VEL"]),
                c=GREEN, s=80, marker='*', zorder=5)
    plt.colorbar(sc4, ax=ax4, label='Mean Err (deg)').ax.tick_params(labelsize=6, colors=DIM)

    ax5 = fig.add_subplot(gs[1, 0])
    style(ax5, 'SETTLE TIME (top 20%)', 'Settle Time (s)', 'Count')
    top20 = [r["metrics"]["settle_time_s"] for r in results_sorted[:len(results)//5]]
    ax5.hist(top20, bins=30, color=GOLD, alpha=0.7, edgecolor='none')
    ax5.axvline(best["metrics"]["settle_time_s"], color=GREEN, lw=1.5, linestyle='--')

    ax6 = fig.add_subplot(gs[1, 1])
    style(ax6, 'MEAN vs P95 ERROR', 'Mean Error (deg)', 'P95 Error (deg)')
    ax6.scatter(mean_err, p95, c=scores, cmap=cmap, s=6, alpha=0.5, linewidths=0)
    ax6.scatter(best["metrics"]["mean_err_deg"], best["metrics"]["p95_err_deg"],
                c=GREEN, s=80, marker='*', zorder=5)

    ax7 = fig.add_subplot(gs[1, 2])
    style(ax7, 'KP SENSITIVITY', 'KP', 'Mean Error (deg)')
    ax7.scatter(kp_vals, mean_err, c=CYAN, s=5, alpha=0.4, linewidths=0)
    ax7.axvline(best["params"]["KP"], color=GREEN, lw=1.2, linestyle='--')

    ax8 = fig.add_subplot(gs[1, 3])
    style(ax8, 'KD SENSITIVITY', 'KD', 'Mean Error (deg)')
    ax8.scatter(kd_vals, mean_err, c=GOLD, s=5, alpha=0.4, linewidths=0)
    ax8.axvline(best["params"]["KD"], color=GREEN, lw=1.2, linestyle='--')

    ax9 = fig.add_subplot(gs[2, :])
    ax9.set_facecolor('#0d0d0d')
    ax9.axis('off')
    ax9.set_title('TOP 10 PARAMETER SETS', color=ACCENT, fontsize=8,
                  fontfamily='monospace', pad=5, loc='left')

    headers = ['Rank','Score','Mean Err','P95 Err','Lock%','Settle(s)','Q_POS','Q_VEL','KP','KI','KD']
    col_x   = np.linspace(0.01, 0.99, len(headers))
    row_h   = 0.082
    header_y = 0.92

    for x, h in zip(col_x, headers):
        ax9.text(x, header_y, h, color=ACCENT, fontsize=6.5, fontfamily='monospace',
                 fontweight='bold', ha='center', va='top', transform=ax9.transAxes)

    for rank, r in enumerate(top):
        y = header_y - (rank+1) * row_h
        row_color = GREEN if rank == 0 else ACCENT
        bg = '#1a1a2e' if rank == 0 else ('#111111' if rank%2==0 else '#0d0d0d')
        ax9.add_patch(plt.Rectangle((0, y-row_h*0.5), 1, row_h*0.9,
                                     transform=ax9.transAxes,
                                     facecolor=bg, edgecolor='none', zorder=0))
        m, p = r["metrics"], r["params"]
        vals = [f"#{rank+1}", f"{r['score']:.2f}", f"{m['mean_err_deg']:.2f}",
                f"{m['p95_err_deg']:.2f}", f"{m['lock_pct']:.1f}%",
                f"{m['settle_time_s']:.1f}s", f"{p['Q_POS']:.4f}",
                f"{p['Q_VEL']:.3f}", f"{p['KP']:.3f}", f"{p['KI']:.4f}", f"{p['KD']:.3f}"]
        for x, v in zip(col_x, vals):
            ax9.text(x, y, v, color=row_color, fontsize=6, fontfamily='monospace',
                     ha='center', va='center', transform=ax9.transAxes)

    plt.savefig('optimization_results.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a', edgecolor='none')
    print("\n  Saved: optimization_results.png")
    plt.show()


def save_best(results):
    results_sorted = sorted(results, key=lambda r: r["score"])
    top  = results_sorted[:TOP_K]
    best = top[0]

    with open("best_params.json", "w") as f:
        json.dump({"config": {"n_trials": N_TRIALS, "n_steps": N_STEPS,
                               "n_trajectories": N_TRAJECTORIES, "dt": DT},
                   "best_params": best["params"], "best_metrics": best["metrics"],
                   "best_score": best["score"], "top_k": top}, f, indent=2)
    print("  Saved: best_params.json")

    print(f"\n{'─'*60}")
    print(f"  BEST PARAMETERS FOUND")
    print(f"{'─'*60}")
    print(f"  Score:       {best['score']:.4f}")
    print(f"  Mean error:  {best['metrics']['mean_err_deg']:.2f} deg")
    print(f"  P95 error:   {best['metrics']['p95_err_deg']:.2f} deg")
    print(f"  Lock-on:     {best['metrics']['lock_pct']:.1f}%")
    print(f"  Settle time: {best['metrics']['settle_time_s']:.2f}s")
    print(f"{'─'*60}")
    print(f"  Q_POS = {best['params']['Q_POS']:.6f}")
    print(f"  Q_VEL = {best['params']['Q_VEL']:.4f}")
    print(f"  KP    = {best['params']['KP']:.4f}")
    print(f"  KI    = {best['params']['KI']:.6f}")
    print(f"  KD    = {best['params']['KD']:.4f}")
    print(f"{'─'*60}")
    print(f"\n  Paste these into turret_sim.py to use the optimized gains.")


def print_sim_patch():
    try:
        with open("best_params.json") as f:
            data = json.load(f)
        p = data["best_params"]
        print(f"\n  Paste into turret_sim.py:")
        print(f"  Q_POS   = {p['Q_POS']:.6f}")
        print(f"  Q_VEL   = {p['Q_VEL']:.4f}")
        print(f"  KP_PAN  = {p['KP']:.4f}")
        print(f"  KI_PAN  = {p['KI']:.6f}")
        print(f"  KD_PAN  = {p['KD']:.4f}")
        print(f"  KP_TILT = {p['KP']:.4f}")
        print(f"  KI_TILT = {p['KI']:.6f}")
        print(f"  KD_TILT = {p['KD']:.4f}")
    except FileNotFoundError:
        print("  Run the optimizer first.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--apply":
        print_sim_patch()
    else:
        results = random_search()
        save_best(results)
        plot_results(results)
