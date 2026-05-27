"""
=============================================================================
visualise.py  --  D4 Visualisation Dashboard (ATP-1 & ATP-3 only)
=============================================================================
Thesis  : Automation of Pyrometer Data Pre-processing for Metal Forming
          and Heat Treatment
Author  : Mallepalli Sravya Reddy
University West, 2026

Deliverable : D4 -- Visualisation tool for raw vs processed temperature
              with basic event markers

NOTE: Sensor calibration (ATP-2) is NOT part of this contribution.
      ATP-2 is in the parallel thesis by the parallel thesis contributor.
      This dashboard shows ATP-1 (denoising) and ATP-3 (compression) only.

WHAT THIS SHOWS:
    Plot 1 (top)        -- All stages overlaid with event markers
    Plot 2 (mid left)   -- Stage 0: Raw data
    Plot 3 (mid centre) -- Stage 1: After denoising (Kalman)
    Plot 4 (mid right)  -- Stage 2: After compression (Wavelet)
    Plot 5 (bot left)   -- Compression zoomed at peak
    Plot 6 (bot centre) -- Zoomed at peak temperature region
    Plot 7 (bot right)  -- Mean temperature bar chart per stage

EVENT MARKERS:
    Red shaded regions    = laser ON (temperature > 100 C)
    Dashed vertical line  = peak temperature point

HOW TO RUN:
    python visualise.py

    To use real NIST data: set DATA_PATH below and uncomment the
    data loading block.

OUTPUT:
    d4_dashboard.png
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import medfilt
from compress import wavelet_compress, wavelet_reconstruct

# =============================================================================
# CONFIGURATION
# =============================================================================
# DATA_PATH = r"C:\Users\sravy\OneDrive\Desktop\ThesisSravya\data (2)\data\Layer01.mat"

# =============================================================================
# DATA  -- replace with real NIST Layer01.mat path when running locally
# =============================================================================
# import scipy.io as sio
# mat   = sio.loadmat(DATA_PATH)
# L     = mat["Layer"][0, 0]
# raw3d = L["RadiantTemp"].astype(np.float32)
# sh_A  = float(L["SHvariable_A"].flat[0])
# sh_B  = float(L["SHvariable_B"].flat[0])
# frame_max = raw3d.max(axis=(0, 1))
# T_raw = np.clip(sh_A * frame_max + sh_B - 273.15, 0, 3000)
# T_raw = T_raw[T_raw > 10].astype(np.float32)

# Synthetic data fallback
np.random.seed(42)
n      = 2065
time_s = np.linspace(0, n * 0.002, n)
T_clean = (1200 + 800 * np.exp(-((time_s - 2.0)**2) / 0.8)
           + 400 * np.sin(2 * np.pi * 1.5 * time_s) * np.exp(-time_s / 3))
T_clean = np.clip(T_clean, 373, 2099).astype(np.float32)
rng = np.random.default_rng(42)
spikes = np.zeros(n, np.float32)
spikes[rng.choice(n, 40, replace=False)] = rng.uniform(300, 800, 40)
T_raw = np.clip(T_clean + rng.normal(0, 500, n).astype(np.float32)
                + spikes, 0, 3000).astype(np.float32)

print("=" * 65)
print("visualise.py  --  D4 Visualisation Dashboard")
print("ATP-1 Signal Denoising  &  ATP-3 Data Compression")
print("NOTE: ATP-2 Calibration is the parallel thesis -- not shown here")
print("=" * 65)
print("\n  Running pipeline ...")

# =============================================================================
# ATP-1: DENOISING  --  Kalman Filter (best classical result)
# =============================================================================
def kalman(sig, Q=1e-3, R=10.0):
    """
    Kalman filter denoiser.
    Q = 0.001 (process noise), R = 10.0 (measurement noise).
    Best ATP-1 result: 89.5% noise reduction on NIST Layer 01.
    Reference: Thesis Section 2.1.2 / Equation (2.3).
    """
    x, P, out = float(sig[0]), 1.0, np.zeros(len(sig), np.float32)
    for i in range(len(sig)):
        P += Q; K = P/(P+R); x = x+K*(sig[i]-x); P=(1-K)*P; out[i]=x
    return out

# Also compute median filter for spike count
T_med    = medfilt(T_raw.astype(np.float64), 7).astype(np.float32)
T_den    = kalman(T_raw)   # ATP-1 best result
spikes   = int((np.abs(T_raw - T_med) > 50).sum())

def noise_std(sig):
    base = np.convolve(sig, np.ones(20)/20, mode="same")
    return float(np.std(sig - base))

noise_raw = noise_std(T_raw)
noise_den = noise_std(T_den)

# =============================================================================
# ATP-3: COMPRESSION  --  Wavelet Haar 10% (best classical result)
# =============================================================================
cw      = wavelet_compress(T_den, keep_fraction=0.10)
T_recon = wavelet_reconstruct(cw).astype(np.float32)

HOT        = int(np.argmax(T_raw))
rmse_comp  = float(np.sqrt(np.mean((T_den - T_recon)**2)))
ratio      = T_den.nbytes / max(1, cw["nonzero"] * 8)

print(f"  Frames         : {n}")
print(f"  Peak temp      : {T_raw.max():.1f} °C at t={time_s[HOT]:.3f}s")
print(f"  Spikes removed : {spikes}")
print(f"  ATP-1 Kalman   : {noise_den:.1f} °C  ({(1-noise_den/noise_raw)*100:.1f}% reduction)")
print(f"  ATP-3 Wavelet  : CR={ratio:.1f}×  RMSE={rmse_comp:.1f} °C")

# Event markers (laser ON = temp > 100 C)
laser_regions = []
in_laser = False; start = 0
for i in range(n):
    if T_raw[i] > 100 and not in_laser:
        start = i; in_laser = True
    elif T_raw[i] <= 100 and in_laser:
        laser_regions.append((start, i)); in_laser = False
if in_laser:
    laser_regions.append((start, n - 1))

print(f"  Laser ON regions: {len(laser_regions)}")

# =============================================================================
# BUILD DASHBOARD  --  7-panel figure
# =============================================================================
print("\n  Building dashboard ...")

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#f8f9fa")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1 (top full width): All stages overlaid ──────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
for s, e in laser_regions:
    ax1.axvspan(time_s[s], time_s[e], alpha=0.08, color="red")
ax1.plot(time_s, T_raw,   color="#e74c3c", lw=0.7, alpha=0.5, label="Stage 0: Raw")
ax1.plot(time_s, T_den,   color="#3498db", lw=1.0,            label="Stage 1: Denoised (Kalman)")
ax1.plot(time_s, T_recon, color="#9b59b6", lw=1.0, ls="--",   label="Stage 2: Compressed (Wavelet 10%)")
ax1.axvline(x=time_s[HOT], color="black", lw=1.0, ls="--", alpha=0.5, label="Peak temp")
if laser_regions:
    s, _ = laser_regions[0]
    ax1.annotate("Laser ON", xy=(time_s[s], 200),
                 xytext=(time_s[s]+0.05, 500),
                 fontsize=8, color="red",
                 arrowprops=dict(arrowstyle="->", color="red"))
ax1.set_title("D4  Pipeline Dashboard — ATP-1 Denoising & ATP-3 Compression\n"
              "NIST Layer 01 IN625",
              fontsize=12, fontweight="bold")
ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Temperature (°C)")
ax1.legend(loc="upper left", fontsize=9, ncol=3)
ax1.set_facecolor("#ffffff"); ax1.grid(True, alpha=0.3)

# ── Plot 2 (mid left): Stage 0 raw ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time_s, T_raw, color="#e74c3c", lw=0.8)
for s, e in laser_regions:
    ax2.axvspan(time_s[s], time_s[e], alpha=0.06, color="red")
ax2.set_title("Stage 0  Raw Data", fontweight="bold", fontsize=10)
ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Temperature (°C)")
ax2.set_facecolor("#fff5f5"); ax2.grid(True, alpha=0.3)
ax2.text(0.02, 0.95, f"Max: {T_raw.max():.0f} °C\nNoise: {noise_raw:.1f} °C",
         transform=ax2.transAxes, fontsize=8, va="top", color="#e74c3c")

# ── Plot 3 (mid centre): Stage 1 denoised ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(time_s, T_raw, color="#e74c3c", lw=0.5, alpha=0.3, label="Raw (input)")
ax3.plot(time_s, T_den, color="#3498db", lw=1.0,            label="Kalman (output)")
ax3.set_title("Stage 1  ATP-1: Kalman Denoising", fontweight="bold", fontsize=10)
ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Temperature (°C)")
ax3.legend(fontsize=7); ax3.set_facecolor("#f0f8ff"); ax3.grid(True, alpha=0.3)
ax3.text(0.02, 0.95,
         f"Noise: {noise_raw:.1f} → {noise_den:.1f} °C\n"
         f"Reduction: {(1-noise_den/noise_raw)*100:.1f}%\n"
         f"Spikes removed: {spikes}",
         transform=ax3.transAxes, fontsize=8, va="top", color="#3498db")

# ── Plot 4 (mid right): ATP-3 compression overlay ────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(time_s, T_den,   color="#3498db", lw=0.7, alpha=0.5, label="Denoised (input)")
ax4.plot(time_s, T_recon, color="#9b59b6", lw=1.0, ls="--",   label="Wavelet (output)")
ax4.set_title("Stage 2  ATP-3: Wavelet Compression", fontweight="bold", fontsize=10)
ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Temperature (°C)")
ax4.legend(fontsize=7); ax4.set_facecolor("#fdf0ff"); ax4.grid(True, alpha=0.3)
ax4.text(0.02, 0.95,
         f"CR: {ratio:.1f}×\nRMSE: {rmse_comp:.1f} °C\n"
         f"ATP-3 target: {'MET ✓' if ratio >= 4 else 'below 4×'}",
         transform=ax4.transAxes, fontsize=8, va="top", color="#9b59b6")

# ── Plot 5 (bot left): After compression zoomed ──────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
Z1, Z2 = max(0, HOT-60), min(n, HOT+60)
ax5.plot(time_s[Z1:Z2], T_den[Z1:Z2],   color="#3498db", lw=0.9, label="Denoised")
ax5.plot(time_s[Z1:Z2], T_recon[Z1:Z2], color="#9b59b6", lw=1.1, ls="--", label="Compressed")
ax5.axvline(x=time_s[HOT], color="black", lw=1.0, ls="--", alpha=0.5)
ax5.set_title("ATP-3  Compression — Zoomed at Peak", fontweight="bold", fontsize=10)
ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Temperature (°C)")
ax5.legend(fontsize=7); ax5.set_facecolor("#fdf0ff"); ax5.grid(True, alpha=0.3)

# ── Plot 6 (bot centre): All stages zoomed ───────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(time_s[Z1:Z2], T_raw[Z1:Z2],   color="#e74c3c", lw=0.8, alpha=0.5, label="Raw")
ax6.plot(time_s[Z1:Z2], T_den[Z1:Z2],   color="#3498db", lw=1.0,            label="Denoised")
ax6.plot(time_s[Z1:Z2], T_recon[Z1:Z2], color="#9b59b6", lw=1.0, ls="--",   label="Compressed")
ax6.axvline(x=time_s[HOT], color="black", lw=1.0, ls="--", alpha=0.5)
ax6.set_title("Zoomed — Peak Temperature Region", fontweight="bold", fontsize=10)
ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Temperature (°C)")
ax6.legend(fontsize=7); ax6.set_facecolor("#fffef0"); ax6.grid(True, alpha=0.3)

# ── Plot 7 (bot right): Mean temperature bar chart ───────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
stages     = ["Raw", "Denoised\n(ATP-1)", "Compressed\n(ATP-3)"]
means      = [T_raw.mean(), T_den.mean(), T_recon.mean()]
stds       = [T_raw.std(),  T_den.std(),  T_recon.std()]
bar_colors = ["#e74c3c",    "#3498db",    "#9b59b6"]
bars = ax7.bar(stages, means, color=bar_colors, alpha=0.85,
               edgecolor="black", linewidth=0.5, width=0.5)
ax7.errorbar(stages, means, yerr=stds, fmt="none",
             color="black", capsize=5, lw=1.5)
for bar, val in zip(bars, means):
    ax7.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
             f"{val:.0f}°C", ha="center", va="bottom",
             fontsize=9, fontweight="bold")
ax7.set_title("Mean Temperature per Stage", fontweight="bold", fontsize=10)
ax7.set_ylabel("Mean Temperature (°C)")
ax7.set_facecolor("#f9f9f9"); ax7.grid(True, alpha=0.3, axis="y")

plt.suptitle(
    "D4  Visualisation Dashboard — ATP-1 & ATP-3 Pipeline\n"
    "Mallepalli Sravya Reddy — University West 2026",
    fontsize=13, fontweight="bold", y=1.01
)

plt.savefig("d4_dashboard.png", dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
print("  Dashboard saved → d4_dashboard.png")

print()
print("=" * 65)
print("D4 VISUALISATION COMPLETE")
print("=" * 65)
print("  7 panels generated:")
print("  1. All stages overlaid with laser ON/OFF event markers")
print("  2. Stage 0 — Raw data")
print("  3. Stage 1 — ATP-1: Kalman denoising")
print("  4. Stage 2 — ATP-3: Wavelet compression overlay")
print("  5. ATP-3 compression zoomed at peak")
print("  6. All stages zoomed at peak temperature")
print("  7. Mean temperature bar chart per stage")
print()
print("  NOTE: ATP-2 Calibration is the parallel thesis — see parallel thesis.")
print("=" * 65)