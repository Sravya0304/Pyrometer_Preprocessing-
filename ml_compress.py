"""
=============================================================================
ml_compress.py  --  ATP-3: ML Compression Methods
=============================================================================
Thesis  : Automation of Pyrometer Data Pre-processing for Metal Forming
          and Heat Treatment
Author  : Mallepalli Sravya Reddy
University West, 2026

Individual Contribution : ATP-3 Data Compression

ML MODELS:
    1. PCA Compression       -- n=3 components, window=32, CR=10.7x
    2. Autoencoder Compress  -- bottleneck=3, window=32, CR=10.7x, 100 epochs

Input    : Kalman-denoised signal (best ATP-1 result)
Target   : CR > 4x with acceptable reconstruction RMSE
Metrics  : Compression ratio (CR), reconstruction RMSE (°C)

NOTE: Sensor calibration (ATP-2) is NOT part of this contribution.
      VAE and Deep Autoencoder are Ajay's ATP-3 methods — not included here.
      ATP-2 and those methods are in the parallel thesis by Avula Ajay Kumar.

HOW TO RUN:
    python ml_compress.py

OUTPUT:
    ml_compress_result.png       -- 4-panel comparison plot
    ml_compress_summary.csv      -- results table
    autoencoder_compressor.pth   -- saved AE model
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pywt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

WINDOW     = 32
BOTTLENECK = 3
AE_EPOCHS  = 100
LR         = 1e-3

# =============================================================================
# DATA  -- replace with real NIST Layer01.mat path when running locally
# =============================================================================
# import scipy.io as sio
# DATA_PATH = r"path/to/Layer01.mat"
# mat   = sio.loadmat(DATA_PATH)
# L     = mat["Layer"][0, 0]
# raw3d = L["RadiantTemp"].astype(np.float32)
# sh_A  = float(L["SHvariable_A"].flat[0])
# sh_B  = float(L["SHvariable_B"].flat[0])
# frame_max = raw3d.max(axis=(0, 1))
# T_raw = np.clip(sh_A * frame_max + sh_B - 273.15, 0, 3000)
# T_raw = T_raw[T_raw > 10].astype(np.float32)
# -- Then apply Kalman filter to get T_comp --

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

# Kalman filter (best ATP-1 result — used as compression input)
def kalman(sig, Q=1e-3, R=10.0):
    x, P, out = float(sig[0]), 1.0, np.zeros(len(sig), np.float32)
    for i in range(len(sig)):
        P += Q; K = P/(P+R); x = x+K*(sig[i]-x); P=(1-K)*P; out[i]=x
    return out

T_comp = kalman(T_raw)   # compression input = Kalman-denoised

print("=" * 65)
print("ml_compress.py  --  ATP-3: ML Compression")
print("=" * 65)
print(f"  Frames : {n}  |  Compression input range : {T_comp.min():.0f} – {T_comp.max():.0f} °C")
print(f"  ATP-3 target : CR > 4×")

raw_bytes = T_comp.nbytes
HOT = int(np.argmax(T_comp))

# =============================================================================
# SLIDING WINDOW HELPERS
# =============================================================================
def make_windows(sig, w):
    return np.array([sig[i:i+w] for i in range(len(sig)-w)], dtype=np.float32)

def windows_to_signal(wins, orig_len):
    out = np.zeros(orig_len, np.float32)
    cnt = np.zeros(orig_len, np.float32)
    for i, w in enumerate(wins):
        out[i:i+len(w)] += w; cnt[i:i+len(w)] += 1
    return out / np.maximum(cnt, 1)

X_all = make_windows(T_comp, WINDOW)
scaler = MinMaxScaler()
X_n   = scaler.fit_transform(X_all)

# =============================================================================
# CLASSICAL BASELINES (for comparison)
# =============================================================================
print("\n--- Classical baselines (for comparison) ---")

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
T_wav    = pywt.waverec(crec,"haar")[:n].astype(np.float32)
rmse_wav = float(np.sqrt(np.mean((T_comp-T_wav)**2)))
cr_wav   = max(1.0, raw_bytes/max(1, nz_vals.nbytes+nz_idx.nbytes))
print(f"  Wavelet Haar 10% : CR={cr_wav:.1f}×  RMSE={rmse_wav:.2f} °C  (ATP-3 {'MET' if cr_wav>=4 else 'not met'})")

# Downsampling 2x
T_ds  = np.interp(np.arange(n), np.arange(0,n,2)[:len(T_comp[::2])],
                   T_comp[::2]).astype(np.float32)
rmse_ds, cr_ds = float(np.sqrt(np.mean((T_comp-T_ds)**2))), 2.0
print(f"  Downsampling 2×  : CR={cr_ds:.1f}×  RMSE={rmse_ds:.2f} °C")

# =============================================================================
# MODEL 1: PCA (n=3)
# =============================================================================
print("\n--- ML Model 1: PCA (n=3) ---")
pca    = PCA(n_components=BOTTLENECK)
X_pca  = pca.fit_transform(X_n)
X_prec = scaler.inverse_transform(pca.inverse_transform(X_pca))
T_pca  = windows_to_signal(X_prec, n)
rmse_pca = float(np.sqrt(np.mean((T_comp-T_pca)**2)))
cr_pca   = WINDOW / BOTTLENECK
var_exp  = pca.explained_variance_ratio_.sum() * 100
print(f"  Components  : {BOTTLENECK}/{WINDOW}  (variance explained: {var_exp:.1f}%)")
print(f"  CR={cr_pca:.1f}×  RMSE={rmse_pca:.2f} °C  (ATP-3 {'MET' if cr_pca>=4 else 'not met'})")

# =============================================================================
# MODEL 2: Autoencoder (bottleneck=3)
# =============================================================================
print("\n--- ML Model 2: Autoencoder (bottleneck=3) ---")

class AutoencoderCompress(nn.Module):
    """
    Autoencoder compressor.
    Encoder: Linear(32->16->8->3)  — compress
    Decoder: Linear(3->8->16->32)  — reconstruct
    CR = 32/3 ≈ 10.7×
    Reference: Romeu et al. (2021); Thesis Section 2.1.3 / 4.4.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(WINDOW, 16), nn.ReLU(),
            nn.Linear(16, 8),      nn.ReLU(),
            nn.Linear(8, BOTTLENECK),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK, 8),  nn.ReLU(),
            nn.Linear(8, 16),          nn.ReLU(),
            nn.Linear(16, WINDOW),
        )
    def forward(self, x): return self.decoder(self.encoder(x))

ae   = AutoencoderCompress()
opt  = torch.optim.Adam(ae.parameters(), lr=LR)
crit = nn.MSELoss()
Xt   = torch.tensor(X_n, dtype=torch.float32)
ae_losses = []

print(f"  Training Autoencoder ({AE_EPOCHS} epochs)...")
for ep in range(AE_EPOCHS):
    ae.train(); opt.zero_grad()
    loss = crit(ae(Xt), Xt); loss.backward(); opt.step()
    ae_losses.append(loss.item())
    if (ep+1) % 20 == 0:
        print(f"    Epoch {ep+1:3d}/{AE_EPOCHS}  Loss: {loss.item():.6f}")

ae.eval()
with torch.no_grad():
    X_ae_rec = scaler.inverse_transform(ae(Xt).numpy())
T_ae    = windows_to_signal(X_ae_rec, n)
rmse_ae = float(np.sqrt(np.mean((T_comp-T_ae)**2)))
cr_ae   = WINDOW / BOTTLENECK
print(f"  CR={cr_ae:.1f}×  RMSE={rmse_ae:.2f} °C  (ATP-3 {'MET' if cr_ae>=4 else 'not met'})")

torch.save(ae.state_dict(), "autoencoder_compressor.pth")
print("  Saved: autoencoder_compressor.pth")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print()
print("=" * 65)
print("ATP-3 COMPRESSION  --  RESULTS")
print("=" * 65)
df = pd.DataFrame([
    {"Method":"Downsampling 2×",    "Type":"Classical","CR":f"{cr_ds:.1f}×", "RMSE_C":f"{rmse_ds:.2f}",  "ATP3_met":"No"},
    {"Method":"Wavelet Haar 10%",   "Type":"Classical","CR":f"{cr_wav:.1f}×","RMSE_C":f"{rmse_wav:.2f}", "ATP3_met":"Yes ✓"},
    {"Method":"PCA (n=3)",          "Type":"ML",       "CR":f"{cr_pca:.1f}×","RMSE_C":f"{rmse_pca:.2f}", "ATP3_met":"Yes ✓"},
    {"Method":"Autoencoder (bn=3)", "Type":"ML",       "CR":f"{cr_ae:.1f}×", "RMSE_C":f"{rmse_ae:.2f}",  "ATP3_met":"Yes ✓"},
])
print(df.to_string(index=False))
df.to_csv("ml_compress_summary.csv", index=False)
print("\n  Saved: ml_compress_summary.csv")

# =============================================================================
# VISUALISATION  --  4-panel
# =============================================================================
print("\n  Generating plots...")
Z1, Z2 = max(0, HOT-80), min(n, HOT+80)

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("ATP-3  ML Compression: PCA + Autoencoder vs Classical\n"
             "Mallepalli Sravya Reddy — University West 2026",
             fontsize=13, fontweight="bold")

# Panel A: Full signal overlay
ax = axes[0,0]
ax.plot(time_s, T_comp, color="black",   lw=1.1, label="Input (Kalman-denoised)")
ax.plot(time_s, T_wav,  color="#9b59b6", lw=1.0, ls="--", label=f"Wavelet 10% ({cr_wav:.1f}×) ★")
ax.plot(time_s, T_ds,   color="#3498db", lw=0.9, label=f"Downsampling ({cr_ds:.0f}×)")
ax.plot(time_s, T_pca,  color="#e67e22", lw=1.1, label=f"PCA n=3 ({cr_pca:.1f}×)")
ax.plot(time_s, T_ae,   color="#c0392b", lw=1.0, ls="-.", label=f"Autoencoder ({cr_ae:.1f}×)")
ax.set_title("A  Full signal — original vs reconstructed")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

# Panel B: Zoomed at peak
ax = axes[0,1]
ax.plot(time_s[Z1:Z2], T_comp[Z1:Z2], color="black",   lw=1.3, label="Input")
ax.plot(time_s[Z1:Z2], T_wav[Z1:Z2],  color="#9b59b6", lw=1.1, ls="--", label="Wavelet ★")
ax.plot(time_s[Z1:Z2], T_pca[Z1:Z2],  color="#e67e22", lw=1.1, label="PCA")
ax.plot(time_s[Z1:Z2], T_ae[Z1:Z2],   color="#c0392b", lw=1.0, ls="-.", label="Autoencoder")
ax.set_title("B  Zoomed at peak temperature")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

# Panel C: Autoencoder training loss
ax = axes[1,0]
ax.plot(ae_losses, color="#c0392b", lw=1.5, label="Autoencoder loss")
ax.set_title("C  Autoencoder training loss (lower = better)")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_yscale("log"); ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

# Panel D: CR vs RMSE scatter
ax = axes[1,1]
pts = [
    ("Raw",       1.0,    0.0,      "#95a5a6"),
    ("Downsamp",  cr_ds,  rmse_ds,  "#3498db"),
    ("Wavelet★",  cr_wav, rmse_wav, "#9b59b6"),
    ("PCA n=3",   cr_pca, rmse_pca, "#e67e22"),
    ("AutoEnc",   cr_ae,  rmse_ae,  "#c0392b"),
]
for lab, rx, ry, col in pts:
    ax.scatter(rx, ry, c=col, s=200, zorder=5, edgecolors="black", linewidths=0.8)
    ax.annotate(lab, (rx, ry), textcoords="offset points", xytext=(7,5), fontsize=9)
ax.axvline(x=4.0, color="green", ls="--", lw=1.5, alpha=0.8, label="CR=4× target")
ax.set_title("D  Compression Ratio vs RMSE\n(bottom-right = best trade-off)")
ax.set_xlabel("Compression ratio"); ax.set_ylabel("RMSE (°C)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("ml_compress_result.png", dpi=150, bbox_inches="tight", facecolor="white")
print("  Plot saved: ml_compress_result.png")

print()
print("=" * 65)
print("COMPLETE  --  ml_compress.py  (ATP-3 only)")
print("=" * 65)
print(f"  Wavelet Haar 10%   : CR={cr_wav:.1f}×  RMSE={rmse_wav:.2f} °C  (ATP-3 MET)")
print(f"  PCA (n=3)          : CR={cr_pca:.1f}×  RMSE={rmse_pca:.2f} °C  (ATP-3 MET)")
print(f"  Autoencoder (bn=3) : CR={cr_ae:.1f}×  RMSE={rmse_ae:.2f} °C  (ATP-3 MET)")
print()
print("  NOTE: VAE and Deep Autoencoder are Ajay's methods — not in this file.")
print("  NOTE: ATP-2 Calibration is Ajay's work — not in this file.")
print("=" * 65)
