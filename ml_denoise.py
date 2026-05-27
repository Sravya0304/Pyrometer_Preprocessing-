"""
=============================================================================
ml_denoise.py  --  ATP-1: ML Denoising Methods
=============================================================================
Thesis  : Automation of Pyrometer Data Pre-processing for Metal Forming
          and Heat Treatment
Author  : Mallepalli Sravya Reddy
University West, 2026

Individual Contribution : ATP-1 Signal Denoising

ML MODELS:
    1. CNN Denoiser      -- 4 Conv1d layers, window=32, 50 epochs
    2. LSTM Denoiser     -- 2 layers, hidden=64, 50 epochs
    3. Autoencoder Den.  -- bottleneck=8, 50 epochs
    4. BiLSTM Denoiser   -- 2 layers, hidden=64, bidirectional, 50 epochs

Training target : median-filtered signal (clean reference)
Evaluation      : noise std dev (C), noise reduction (%)

NOTE: ATP-2 Calibration is NOT part of this contribution.
      ATP-2 is in the parallel thesis by Avula Ajay Kumar.

HOW TO RUN:
    python ml_denoise.py

OUTPUT:
    ml_denoise_result.png         -- 4-panel comparison plot
    ml_denoise_summary.csv        -- results table
    cnn_denoiser.pth              -- saved CNN model
    lstm_denoiser.pth             -- saved LSTM model
    autoencoder_denoiser.pth      -- saved AE model
    bilstm_denoiser.pth           -- saved BiLSTM model
=============================================================================
"""

import numpy as np
import scipy.signal as sig_mod
from scipy.signal import medfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

WINDOW   = 32
EPOCHS   = 50
BATCH    = 32
LR       = 1e-3

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
print("ml_denoise.py  --  ATP-1: ML Denoising")
print("=" * 65)
print(f"  Frames : {n}  |  Range : {T_raw.min():.0f} – {T_raw.max():.0f} °C")

def noise_std(sig):
    base = np.convolve(sig, np.ones(20)/20, mode="same")
    return float(np.std(sig - base))

noise_raw = noise_std(T_raw)
HOT = int(np.argmax(T_raw))

# Training target = median-filtered signal
T_med = medfilt(T_raw.astype(np.float64), 7).astype(np.float32)

# Normalise
scaler = MinMaxScaler()
T_rn   = scaler.fit_transform(T_raw.reshape(-1,1)).ravel().astype(np.float32)
T_mn   = scaler.transform(T_med.reshape(-1,1)).ravel().astype(np.float32)
split  = int(0.8 * n)

# =============================================================================
# DATASET
# =============================================================================
class WindowDS(Dataset):
    def __init__(self, x, y, w): self.x, self.y, self.w = x, y, w
    def __len__(self): return len(self.x) - self.w
    def __getitem__(self, i):
        return (torch.tensor(self.x[i:i+self.w]).unsqueeze(0),
                torch.tensor(self.y[i:i+self.w]).unsqueeze(0))

dl = DataLoader(WindowDS(T_rn[:split], T_mn[:split], WINDOW),
                batch_size=BATCH, shuffle=True)

# =============================================================================
# MODELS
# =============================================================================
class CNNDenoiser(nn.Module):
    """
    1D CNN denoiser — 4 Conv1d layers.
    Learns local noise patterns via convolution.
    Reference: LeCun et al. (1989); Thesis Section 2.1.2 / 4.3.2.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 16, 5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 1,  7, padding=3),
        )
    def forward(self, x): return self.net(x)

class LSTMDenoiser(nn.Module):
    """
    2-layer LSTM denoiser, hidden=64.
    Gating mechanism retains long-term trend, suppresses noise.
    Reference: Hochreiter & Schmidhuber (1997); Thesis Section 2.1.2.
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True)
        self.fc   = nn.Linear(64, 1)
    def forward(self, x):
        x = x.squeeze(1).unsqueeze(2)
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(2).unsqueeze(1)

class AEDenoiser(nn.Module):
    """
    Autoencoder denoiser — bottleneck=8.
    Encoder: Linear(32->16->8). Decoder: Linear(8->16->32).
    Bottleneck forces noise removal (noise has high entropy).
    Reference: Romeu et al. (2021); Thesis Section 2.1.2.
    """
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(WINDOW,16),nn.ReLU(),nn.Linear(16,8))
        self.dec = nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,WINDOW))
    def forward(self, x):
        f = x.squeeze(1)
        return self.dec(self.enc(f)).unsqueeze(1)

class BiLSTMDenoiser(nn.Module):
    """
    Bidirectional LSTM denoiser — 2 layers, hidden=64, bidirectional=True.
    Uses both past AND future context → better offline denoising.
    Output: Linear(128->1) (128 = forward + backward concatenated).
    Reference: Schmidhuber (2015); Thesis Section 2.1.2.
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.fc   = nn.Linear(128, 1)
    def forward(self, x):
        x = x.squeeze(1).unsqueeze(2)
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(2).unsqueeze(1)

# =============================================================================
# TRAIN & PREDICT
# =============================================================================
def train(model, dl, epochs, name):
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    losses = []
    print(f"\n  Training {name} ({epochs} epochs)...")
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad(); loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            ep_loss += loss.item()
        avg = ep_loss / len(dl)
        losses.append(avg)
        if (ep+1) % 10 == 0:
            print(f"    Epoch {ep+1:3d}/{epochs}  Loss: {avg:.6f}")
    return losses

def predict(model, sig_n):
    model.eval()
    pred = np.zeros(len(sig_n), np.float32)
    cnt  = np.zeros(len(sig_n), np.float32)
    with torch.no_grad():
        for i in range(0, len(sig_n)-WINDOW, WINDOW//2):
            out = model(torch.tensor(sig_n[i:i+WINDOW]).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
            pred[i:i+WINDOW] += out
            cnt[i:i+WINDOW]  += 1
    cnt = np.maximum(cnt, 1)
    return scaler.inverse_transform((pred/cnt).reshape(-1,1)).ravel().astype(np.float32)

# =============================================================================
# RUN ALL 4 MODELS
# =============================================================================
cnn   = CNNDenoiser();   l_cnn   = train(cnn,   dl, EPOCHS, "CNN")
lstm  = LSTMDenoiser();  l_lstm  = train(lstm,  dl, EPOCHS, "LSTM")
aed   = AEDenoiser();    l_aed   = train(aed,   dl, EPOCHS, "Autoencoder")
bilst = BiLSTMDenoiser();l_bilst = train(bilst, dl, EPOCHS, "BiLSTM")

T_cnn  = predict(cnn,   T_rn)
T_lstm = predict(lstm,  T_rn)
T_aed  = predict(aed,   T_rn)
T_bi   = predict(bilst, T_rn)

n_cnn, n_lstm, n_aed, n_bi = [noise_std(t) for t in [T_cnn,T_lstm,T_aed,T_bi]]

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print()
print("=" * 65)
print("ATP-1 ML DENOISING  --  RESULTS")
print("=" * 65)
print(f"  Raw baseline noise : {noise_raw:.2f} °C")
print(f"  Median target noise: {noise_std(T_med):.2f} °C")
print()

df = pd.DataFrame([
    {"Method":"CNN Denoiser",      "Type":"ML","Noise_C":round(n_cnn,2),  "Reduction_%":f"{(1-n_cnn/noise_raw)*100:.1f}"},
    {"Method":"LSTM Denoiser",     "Type":"ML","Noise_C":round(n_lstm,2), "Reduction_%":f"{(1-n_lstm/noise_raw)*100:.1f}"},
    {"Method":"Autoencoder Den.",  "Type":"ML","Noise_C":round(n_aed,2),  "Reduction_%":f"{(1-n_aed/noise_raw)*100:.1f}"},
    {"Method":"BiLSTM Denoiser",   "Type":"ML","Noise_C":round(n_bi,2),   "Reduction_%":f"{(1-n_bi/noise_raw)*100:.1f}"},
])
print(df.to_string(index=False))
df.to_csv("ml_denoise_summary.csv", index=False)
print("\n  Saved: ml_denoise_summary.csv")

# Save models
torch.save(cnn.state_dict(),   "cnn_denoiser.pth")
torch.save(lstm.state_dict(),  "lstm_denoiser.pth")
torch.save(aed.state_dict(),   "autoencoder_denoiser.pth")
torch.save(bilst.state_dict(), "bilstm_denoiser.pth")
print("  Saved: cnn_denoiser.pth, lstm_denoiser.pth, autoencoder_denoiser.pth, bilstm_denoiser.pth")

# =============================================================================
# VISUALISATION  --  4-panel
# =============================================================================
print("\n  Generating plots...")
Z1, Z2 = max(0, HOT-80), min(n, HOT+80)

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("ATP-1  ML Denoising: CNN + LSTM + Autoencoder + BiLSTM\n"
             "Mallepalli Sravya Reddy — University West 2026",
             fontsize=13, fontweight="bold")

# Panel A: Full signal
ax = axes[0,0]
ax.plot(time_s, T_raw,  color="#e74c3c", lw=0.5, alpha=0.35, label="Raw")
ax.plot(time_s, T_med,  color="#3498db", lw=1.2, label="Median (training target)")
ax.plot(time_s, T_cnn,  color="#e67e22", lw=0.9, label="CNN")
ax.plot(time_s, T_lstm, color="#c0392b", lw=0.9, ls="--", label="LSTM")
ax.plot(time_s, T_aed,  color="#8e44ad", lw=0.9, ls="-.", label="Autoencoder")
ax.plot(time_s, T_bi,   color="#27ae60", lw=0.9, ls=":",  label="BiLSTM")
ax.set_title("A  Full signal — all ML methods"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

# Panel B: Zoomed at peak
ax = axes[0,1]
ax.plot(time_s[Z1:Z2], T_raw[Z1:Z2],  color="#e74c3c", lw=0.7, alpha=0.4, label="Raw")
ax.plot(time_s[Z1:Z2], T_med[Z1:Z2],  color="#3498db", lw=1.3, label="Median target")
ax.plot(time_s[Z1:Z2], T_cnn[Z1:Z2],  color="#e67e22", lw=1.1, label="CNN")
ax.plot(time_s[Z1:Z2], T_lstm[Z1:Z2], color="#c0392b", lw=1.0, ls="--", label="LSTM")
ax.plot(time_s[Z1:Z2], T_aed[Z1:Z2],  color="#8e44ad", lw=1.0, ls="-.", label="Autoencoder")
ax.plot(time_s[Z1:Z2], T_bi[Z1:Z2],   color="#27ae60", lw=1.0, ls=":",  label="BiLSTM")
ax.set_title("B  Zoomed at peak temperature"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

# Panel C: Training loss curves
ax = axes[1,0]
ax.plot(l_cnn,   color="#e67e22", lw=1.4, label="CNN")
ax.plot(l_lstm,  color="#c0392b", lw=1.4, ls="--", label="LSTM")
ax.plot(l_aed,   color="#8e44ad", lw=1.4, ls="-.", label="Autoencoder")
ax.plot(l_bilst, color="#27ae60", lw=1.4, ls=":",  label="BiLSTM")
ax.set_title("C  Training loss curves (lower = better)")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_yscale("log"); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

# Panel D: Noise bar chart
ax = axes[1,1]
names = ["CNN", "LSTM", "Autoencoder", "BiLSTM"]
vals  = [n_cnn, n_lstm, n_aed, n_bi]
cols  = ["#e67e22","#c0392b","#8e44ad","#27ae60"]
bars  = ax.bar(names, vals, color=cols, alpha=0.85, edgecolor="black", linewidth=0.5, width=0.6)
ax.axhline(y=noise_raw, color="#e74c3c", ls="--", lw=1.2, label=f"Raw ({noise_raw:.0f} °C)")
ax.axhline(y=noise_std(T_med), color="#3498db", ls="--", lw=1.2, label=f"Median target ({noise_std(T_med):.0f} °C)")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("D  Noise Std Dev — ML Methods (lower = better)")
ax.set_ylabel("Noise std dev (°C)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.25, axis="y")

plt.tight_layout()
plt.savefig("ml_denoise_result.png", dpi=150, bbox_inches="tight", facecolor="white")
print("  Plot saved: ml_denoise_result.png")

print()
print("=" * 65)
print("COMPLETE  --  ml_denoise.py  (ATP-1 only)")
print("=" * 65)
print(f"  CNN Denoiser     : {n_cnn:.1f} °C  ({(1-n_cnn/noise_raw)*100:.1f}% reduction)")
print(f"  LSTM Denoiser    : {n_lstm:.1f} °C  ({(1-n_lstm/noise_raw)*100:.1f}% reduction)")
print(f"  Autoencoder Den. : {n_aed:.1f} °C  ({(1-n_aed/noise_raw)*100:.1f}% reduction)")
print(f"  BiLSTM Denoiser  : {n_bi:.1f} °C  ({(1-n_bi/noise_raw)*100:.1f}% reduction)")
print()
print("  NOTE: ATP-2 Calibration is Ajay's work — not in this file.")
print("=" * 65)
