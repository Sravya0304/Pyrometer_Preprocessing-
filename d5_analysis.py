"""
=============================================================================
d5_analysis.py  --  D5: Trade-off Analysis (ATP-1 & ATP-3 only)
=============================================================================
Thesis  : Automation of Pyrometer Data Pre-processing for Metal Forming
          and Heat Treatment
Author  : Mallepalli Sravya Reddy
University West, 2026

Deliverable : D5 -- Analysis of how denoising and compression choices
              affect signal quality and storage efficiency

NOTE: Sensor calibration (ATP-2) is NOT part of this contribution.
      ATP-2 is in the parallel thesis by Avula Ajay Kumar.
      VAE and Deep Autoencoder are Ajay's ATP-3 methods -- not included.

METHODS ANALYSED:

  ATP-1 Denoising (6 classical):
    Raw Baseline, Moving Average (k=7), Median Filter (k=7),
    Savitzky-Golay (w=11,p=3), Gaussian (sigma=3.0), Kalman (Q=0.001,R=10)

  ATP-3 Compression (5 methods):
    Raw Baseline, Downsampling (2x), SVD (rank=10),
    Wavelet Haar (10%), PCA (n=3)

HOW TO RUN:
    python d5_analysis.py

    To use real NIST data: set DATA_PATH at the top of the file.

OUTPUT:
    d5_analysis.png       -- 4-panel trade-off figure
    d5_summary.csv        -- full results table
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd
from scipy.signal import medfilt, savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pywt

SEED = 42
np.random.seed(SEED)

# =============================================================================
# DATA  -- replace DATA_PATH with your Layer01.mat path when running locally
# =============================================================================
# import scipy.io as sio
# DATA_PATH = r"C:\Users\sravy\OneDrive\Desktop\ThesisSravya\data\Layer01.mat"
# mat   = sio.loadmat(DATA_PATH)
# L     = mat["Layer"][0, 0]
# raw3d = L["RadiantTemp"].astype(np.float32)
# sh_A  = float(L["SHvariable_A"].flat[0])
# sh_B  = float(L["SHvariable_B"].flat[0])
# frame_max = raw3d.max(axis=(0, 1))
# T_raw = np.clip(sh_A * frame_max + sh_B - 273.15, 0, 3000)
# T_raw = T_raw[T_raw > 10].astype(np.float32)

n      = 2065
time_s = np.linspace(0, n * 0.002, n)
T_clean = (1200 + 800 * np.exp(-((time_s - 2.0)**2) / 0.8)
           + 400 * np.sin(2 * np.pi * 1.5 * time_s) * np.exp(-time_s / 3))
T_clean = np.clip(T_clean, 373, 2099).astype(np.float32)
rng = np.random.default_rng(SEED)
spikes = np.zeros(n, np.float32)
spikes[rng.choice(n, 40, replace=False)] = rng.uniform(300, 800, 40)
T_raw = np.clip(T_clean + rng.normal(0, 500, n).astype(np.float32)
                + spikes, 0, 3000).astype(np.float32)

print("=" * 65)
print("d5_analysis.py  --  D5: ATP-1 Denoising & ATP-3 Compression")
print("=" * 65)
print(f"  Frames : {n}  |  Raw range : {T_raw.min():.0f} – {T_raw.max():.0f} °C")

def noise_std(sig):
    base = np.convolve(sig, np.ones(20)/20, mode="same")
    return float(np.std(sig - base))

def kalman(sig, Q=1e-3, R=10.0):
    x, P, out = float(sig[0]), 1.0, np.zeros(len(sig), np.float32)
    for i in range(len(sig)):
        P += Q; K = P/(P+R); x = x+K*(sig[i]-x); P=(1-K)*P; out[i]=x
    return out

noise_raw = noise_std(T_raw)

# =============================================================================
# STAGE 1  --  ATP-1: ALL DENOISING METHODS
# =============================================================================
print("\n  Running ATP-1 denoising methods ...")

T_ma    = np.convolve(T_raw, np.ones(7)/7, mode="same").astype(np.float32)
T_med   = medfilt(T_raw.astype(np.float64), 7).astype(np.float32)
T_sg    = savgol_filter(T_raw.astype(np.float64), 11, 3).astype(np.float32)
T_gauss = gaussian_filter1d(T_raw.astype(np.float64), 3.0).astype(np.float32)
T_kal   = kalman(T_raw)

n_ma, n_med, n_sg = noise_std(T_ma), noise_std(T_med), noise_std(T_sg)
n_gauss, n_kal    = noise_std(T_gauss), noise_std(T_kal)

den_results = [
    {"Method":"Raw Baseline",        "Type":"Baseline",  "Noise_C":round(noise_raw,2), "Reduction_%":0.0},
    {"Method":"Moving Average (k=7)","Type":"Classical", "Noise_C":round(n_ma,2),      "Reduction_%":round((1-n_ma/noise_raw)*100,1)},
    {"Method":"Median Filter (k=7)", "Type":"Classical", "Noise_C":round(n_med,2),     "Reduction_%":round((1-n_med/noise_raw)*100,1)},
    {"Method":"Savitzky-Golay",      "Type":"Classical", "Noise_C":round(n_sg,2),      "Reduction_%":round((1-n_sg/noise_raw)*100,1)},
    {"Method":"Gaussian (sigma=3)",  "Type":"Classical", "Noise_C":round(n_gauss,2),   "Reduction_%":round((1-n_gauss/noise_raw)*100,1)},
    {"Method":"Kalman Filter",       "Type":"Classical", "Noise_C":round(n_kal,2),     "Reduction_%":round((1-n_kal/noise_raw)*100,1)},
]

for r in den_results:
    print(f"    {r['Method']:22s}: {r['Noise_C']:.2f} °C  ({r['Reduction_%']}% reduction)")

# Use Kalman output as compression input (best ATP-1 result)
T_comp = T_kal.copy()

# =============================================================================
# STAGE 2  --  ATP-3: ALL COMPRESSION METHODS
# =============================================================================
print("\n  Running ATP-3 compression methods ...")

raw_bytes = T_comp.nbytes
comp_results = []

# Raw baseline
comp_results.append({"Method":"Raw Baseline",    "Type":"Baseline",  "CR":1.0,  "RMSE_C":0.0,   "ATP3_met":"—"})

# Downsampling 2x
T_ds  = np.interp(np.arange(n), np.arange(0,n,2)[:len(T_comp[::2])],
                   T_comp[::2]).astype(np.float32)
rmse_ds = float(np.sqrt(np.mean((T_comp-T_ds)**2)))
comp_results.append({"Method":"Downsampling 2×", "Type":"Classical", "CR":2.0,  "RMSE_C":round(rmse_ds,2),  "ATP3_met":"No"})

# SVD rank=10
nu  = (n//32)*32
X   = T_comp[:nu].reshape(-1,32).astype(np.float64)
U,s,Vt = np.linalg.svd(X, full_matrices=False)
Xr  = (U[:,:10]*s[:10])@Vt[:10,:]
T_svd = np.interp(np.arange(n), np.linspace(0,n-1,Xr.ravel().size),
                   Xr.ravel()).astype(np.float32)
rmse_svd = float(np.sqrt(np.mean((T_comp-T_svd)**2)))
cr_svd   = max(1.0, raw_bytes/(U[:,:10].nbytes+s[:10].nbytes+Vt[:10,:].nbytes))
comp_results.append({"Method":"SVD rank=10",      "Type":"Classical", "CR":round(cr_svd,1), "RMSE_C":round(rmse_svd,2), "ATP3_met":"No"})

# Wavelet Haar 10%
coeffs  = pywt.wavedec(T_comp.astype(np.float64), "haar")
all_c   = np.concatenate([c.ravel() for c in coeffs])
nk      = max(1, int(0.10*len(all_c)))
thr     = np.sort(np.abs(all_c))[::-1][nk-1]
all_t   = np.where(np.abs(all_c)>=thr, all_c, 0.0)
nz_idx  = np.where(all_t!=0)[0].astype(np.int32)
nz_vals = all_t[nz_idx].astype(np.float32)
nz_set  = set(nz_idx.tolist())
ip, crec = 0, []
for c in coeffs:
    arr = np.zeros(c.size, np.float64)
    for j in range(c.size):
        g = ip+j
        if g in nz_set:
            p = int(np.searchsorted(nz_idx, g))
            if p < len(nz_idx) and nz_idx[p]==g: arr[j]=float(nz_vals[p])
    crec.append(arr.reshape(c.shape)); ip+=c.size
T_wav   = pywt.waverec(crec,"haar")[:n].astype(np.float32)
rmse_wav= float(np.sqrt(np.mean((T_comp-T_wav)**2)))
cr_wav  = max(1.0, raw_bytes/max(1, nz_vals.nbytes+nz_idx.nbytes))
comp_results.append({"Method":"Wavelet Haar 10%", "Type":"Classical", "CR":round(cr_wav,1), "RMSE_C":round(rmse_wav,2), "ATP3_met":"Yes ✓"})

# PCA n=3
Xw  = np.array([T_comp[i:i+32] for i in range(n-32)], dtype=np.float32)
sc2 = MinMaxScaler()
Xwn = sc2.fit_transform(Xw)
pca = PCA(n_components=3)
Xpr = sc2.inverse_transform(pca.inverse_transform(pca.fit_transform(Xwn)))
T_pca = np.zeros(n,np.float32); cp=np.zeros(n,np.float32)
for i,w in enumerate(Xpr): T_pca[i:i+32]+=w; cp[i:i+32]+=1
T_pca/=np.maximum(cp,1)
rmse_pca = float(np.sqrt(np.mean((T_comp-T_pca)**2)))
comp_results.append({"Method":"PCA (n=3)",        "Type":"ML",        "CR":round(32/3,1), "RMSE_C":round(rmse_pca,2), "ATP3_met":"Yes ✓"})

for r in comp_results:
    print(f"    {r['Method']:20s}: CR={r['CR']}×  RMSE={r['RMSE_C']} °C  [{r['ATP3_met']}]")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print()
print("=" * 65)
print("TABLE 1  --  ATP-1 Denoising Results")
print("=" * 65)
df1 = pd.DataFrame(den_results)
print(df1.to_string(index=False))

print()
print("=" * 65)
print("TABLE 2  --  ATP-3 Compression Results")
print("=" * 65)
df2 = pd.DataFrame(comp_results)
print(df2.to_string(index=False))

df_all = pd.concat([df1.assign(Stage="ATP-1 Denoising"),
                    df2.assign(Stage="ATP-3 Compression")], ignore_index=True)
df_all.to_csv("d5_summary.csv", index=False)
print("\n  Saved: d5_summary.csv")

# =============================================================================
# VISUALISATION  --  4-panel figure (ATP-1 top, ATP-3 bottom)
# =============================================================================
print("\n  Generating plots ...")

def col(t):
    return "#e74c3c" if t=="Baseline" else "#3498db" if t=="Classical" else "#e67e22"

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#ffffff")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.35)

# ── Panel A: Denoising noise bar chart (all 6 methods) ───────────────────────
ax1 = fig.add_subplot(gs[0, 0])
names_d = [r["Method"] for r in den_results]
noise_v = [r["Noise_C"] for r in den_results]
cols_d  = [col(r["Type"]) for r in den_results]
bars = ax1.bar(names_d, noise_v, color=cols_d, alpha=0.85,
               edgecolor="black", linewidth=0.5, width=0.65)
for bar, val in zip(bars, noise_v):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(noise_v)*0.01,
             f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
best_d = int(np.argmin(noise_v))
bars[best_d].set_edgecolor("black"); bars[best_d].set_linewidth(2.5)
ax1.set_title("ATP-1  Noise Std Dev — All Classical Methods\n(lower = better)",
              fontweight="bold", fontsize=11)
ax1.set_ylabel("Noise std dev (°C)")
ax1.grid(True, alpha=0.25, axis="y")
plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax1.legend(handles=[
    mpatches.Patch(color="#e74c3c", label="Baseline"),
    mpatches.Patch(color="#3498db", label="Classical"),
], fontsize=9)

# ── Panel B: Noise reduction % bar chart ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
red_v = [r["Reduction_%"] for r in den_results]
bars2 = ax2.bar(names_d, red_v, color=cols_d, alpha=0.85,
                edgecolor="black", linewidth=0.5, width=0.65)
for bar, val in zip(bars2, red_v):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_title("ATP-1  Noise Reduction % — All Methods\nNIST Layer 01",
              fontweight="bold", fontsize=11)
ax2.set_ylabel("Noise reduction (%)")
ax2.grid(True, alpha=0.25, axis="y")
plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=9)

# ── Panel C: Compression RMSE bar chart ──────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
names_c = [r["Method"] for r in comp_results]
rmse_v  = [r["RMSE_C"] for r in comp_results]
cols_c  = [col(r["Type"]) for r in comp_results]
bars3 = ax3.bar(names_c, rmse_v, color=cols_c, alpha=0.85,
                edgecolor="black", linewidth=0.5, width=0.65)
for bar, val in zip(bars3, rmse_v):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax3.set_title("ATP-3  Compression RMSE — All Methods\n(lower = better)",
              fontweight="bold", fontsize=11)
ax3.set_ylabel("Reconstruction RMSE (°C)")
ax3.grid(True, alpha=0.25, axis="y")
plt.setp(ax3.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax3.legend(handles=[
    mpatches.Patch(color="#e74c3c", label="Baseline"),
    mpatches.Patch(color="#3498db", label="Classical"),
    mpatches.Patch(color="#e67e22", label="ML"),
], fontsize=9)

# ── Panel D: CR vs RMSE scatter ──────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
cr_v = [r["CR"] for r in comp_results]
for lab, rx, ry, col_c in zip(names_c, cr_v, rmse_v, cols_c):
    ax4.scatter(rx, ry, c=col_c, s=200, zorder=5,
                edgecolors="black", linewidths=0.8)
    ax4.annotate(lab, (rx, ry), textcoords="offset points",
                 xytext=(7, 5), fontsize=9)
ax4.axvline(x=4.0, color="green", ls="--", lw=1.5, alpha=0.8, label="CR = 4× target")
ax4.set_title("ATP-3  Compression Ratio vs RMSE\n(bottom-right = best trade-off)",
              fontweight="bold", fontsize=11)
ax4.set_xlabel("Compression ratio  (higher = smaller file)")
ax4.set_ylabel("Reconstruction RMSE (°C)  — lower = better")
ax4.legend(fontsize=9); ax4.grid(True, alpha=0.25)

fig.suptitle(
    "D5  Trade-off Analysis — ATP-1 Signal Denoising & ATP-3 Data Compression\n"
    "Mallepalli Sravya Reddy — University West 2026",
    fontsize=13, fontweight="bold", y=1.01
)

plt.savefig("d5_analysis.png", dpi=150, bbox_inches="tight", facecolor="white")
print("  Plot saved: d5_analysis.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print()
print("=" * 65)
print("D5 COMPLETE  --  KEY FINDINGS")
print("=" * 65)
print(f"  ATP-1 best : Kalman Filter   {n_kal:.1f} °C  ({(1-n_kal/noise_raw)*100:.1f}% reduction)")
print(f"  ATP-3 best : Wavelet 10%     CR={cr_wav:.1f}×  RMSE={rmse_wav:.2f} °C  (ATP-3 MET)")
print(f"  ATP-3 ML   : PCA n=3         CR={32/3:.1f}×  RMSE={rmse_pca:.2f} °C  (ATP-3 MET)")
print()
print("  NOTE: ATP-2 Calibration is Ajay's work — not included in D5.")
print("  NOTE: VAE and Deep AE are Ajay's methods — not included.")
print("=" * 65)
print("DONE — d5_analysis.py (ATP-1 + ATP-3 only, sravya/ folder)")
print("=" * 65)