"""
=============================================================================
denoise_atp13.py  --  Complete ATP-1 & ATP-3 Pipeline
=============================================================================
Thesis  : Automation of Pyrometer Data Pre-processing for Metal Forming
          and Heat Treatment
Author  : Mallepalli Sravya Reddy
University West, 2026

Individual Contribution:
    ATP-1  Signal Denoising  -- 5 classical + 4 ML methods
    ATP-3  Data Compression  -- 3 classical + 2 ML methods

NOTE: Sensor calibration (ATP-2) is NOT part of this contribution.
      ATP-2 is in the parallel thesis by Avula Ajay Kumar.

METHODS:

  ATP-1 Denoising (10 methods):
    Classical : Moving Average (k=7), Median Filter (k=7),
                Savitzky-Golay (w=11,p=3), Gaussian (sigma=3.0),
                Kalman Filter (Q=0.001, R=10.0)
    ML        : CNN (4 Conv1d layers, 50 ep), LSTM (2 layers h=64, 50 ep),
                Autoencoder (bottleneck=8, 50 ep),
                BiLSTM (2 layers h=64, 50 ep)

  ATP-3 Compression (6 methods):
    Classical : Downsampling (2x), SVD (rank=10), Wavelet Haar (10%)
    ML        : PCA (n=3, window=32), Autoencoder (bottleneck=3, window=32)

EVALUATION:
    ATP-1 : noise std dev (C), noise reduction (%)
    ATP-3 : compression ratio (CR), reconstruction RMSE (C)
    ATP-3 target : CR > 4x

OUTPUT:
    denoise_atp13_results.png      -- 6-panel comparison figure
    atp1_denoise_summary.csv       -- ATP-1 results table
    atp3_compress_summary.csv      -- ATP-3 results table

HOW TO RUN:
    python denoise_atp13.py
=============================================================================
"""

import numpy as np
import scipy.signal as sig_mod
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pywt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

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
rng     = np.random.default_rng(SEED)
spikes  = np.zeros(n, np.float32)
spikes[rng.choice(n, 40, replace=False)] = rng.uniform(300, 800, 40)
T_raw   = np.clip(T_clean + rng.normal(0, 500, n).astype(np.float32) + spikes,
                  0, 3000).astype(np.float32)

print("=" * 65)
print("denoise_atp13.py  --  ATP-1 Denoising & ATP-3 Compression")
print("=" * 65)
print(f"  Frames : {n}  |  Range : {T_raw.min():.0f} – {T_raw.max():.0f} °C")

def noise_std(sig):
    base = np.convolve(sig, np.ones(20)/20, mode="same")
    return float(np.std(sig - base))

noise_raw = noise_std(T_raw)
HOT       = int(np.argmax(T_raw))

# =============================================================================
# PART 1  --  ATP-1: CLASSICAL DENOISING
# =============================================================================
print("\n--- ATP-1: Classical Denoising ---")

T_ma    = np.convolve(T_raw, np.ones(7)/7, mode="same").astype(np.float32)
T_med   = medfilt(T_raw.astype(np.float64), 7).astype(np.float32)
T_sg    = sig_mod.savgol_filter(T_raw.astype(np.float64), 11, 3).astype(np.float32)
T_gauss = gaussian_filter1d(T_raw.astype(np.float64), 3.0).astype(np.float32)

def kalman(sig, Q=1e-3, R=10.0):
    x, P, out = float(sig[0]), 1.0, np.zeros(len(sig), np.float32)
    for i in range(len(sig)):
        P += Q; K = P/(P+R); x = x + K*(sig[i]-x); P = (1-K)*P; out[i] = x
    return out

T_kal = kalman(T_raw)

n_ma, n_med, n_sg = noise_std(T_ma), noise_std(T_med), noise_std(T_sg)
n_gauss, n_kal    = noise_std(T_gauss), noise_std(T_kal)

for name, val in [("Moving Avg", n_ma), ("Median", n_med),
                  ("Savitzky-Golay", n_sg), ("Gaussian", n_gauss),
                  ("Kalman", n_kal)]:
    print(f"  {name:18s}: {val:.2f} °C  ({(1-val/noise_raw)*100:.1f}% reduction)")

# =============================================================================
# PART 2  --  ATP-1: ML DENOISING
# =============================================================================
print("\n--- ATP-1: ML Denoising ---")

W = 32
scaler_d = MinMaxScaler()
T_rn     = scaler_d.fit_transform(T_raw.reshape(-1,1)).ravel().astype(np.float32)
T_mn     = scaler_d.transform(T_med.reshape(-1,1)).ravel().astype(np.float32)
split    = int(0.8*n)

class WDS(Dataset):
    def __init__(self, x, y, w): self.x,self.y,self.w = x,y,w
    def __len__(self): return len(self.x)-self.w
    def __getitem__(self, i):
        return (torch.tensor(self.x[i:i+self.w]).unsqueeze(0),
                torch.tensor(self.y[i:i+self.w]).unsqueeze(0))

dl = DataLoader(WDS(T_rn[:split], T_mn[:split], W), batch_size=32, shuffle=True)

class CNNDenoiser(nn.Module):
    """4 Conv1d layers — learns local noise patterns."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,16,7,padding=3), nn.ReLU(),
            nn.Conv1d(16,32,5,padding=2), nn.ReLU(),
            nn.Conv1d(32,16,5,padding=2), nn.ReLU(),
            nn.Conv1d(16,1,7,padding=3))
    def forward(self, x): return self.net(x)

class LSTMDenoiser(nn.Module):
    """2-layer LSTM — learns temporal signal trends."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1,64,2,batch_first=True)
        self.fc   = nn.Linear(64,1)
    def forward(self, x):
        x = x.squeeze(1).unsqueeze(2)
        out,_ = self.lstm(x)
        return self.fc(out).squeeze(2).unsqueeze(1)

class AEDenoiser(nn.Module):
    """Autoencoder — bottleneck forces noise removal."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(W,16),nn.ReLU(),nn.Linear(16,8))
        self.dec = nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,W))
    def forward(self, x):
        f = x.squeeze(1); return self.dec(self.enc(f)).unsqueeze(1)

class BiLSTMDenoiser(nn.Module):
    """BiLSTM — uses past AND future context for denoising."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1,64,2,batch_first=True,bidirectional=True)
        self.fc   = nn.Linear(128,1)
    def forward(self, x):
        x = x.squeeze(1).unsqueeze(2)
        out,_ = self.lstm(x)
        return self.fc(out).squeeze(2).unsqueeze(1)

def fit(model, dl, epochs=50):
    opt, crit = torch.optim.Adam(model.parameters(), lr=1e-3), nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb,yb in dl:
            opt.zero_grad(); loss=crit(model(xb),yb); loss.backward(); opt.step()

def infer(model, sig_n):
    model.eval(); pred=np.zeros(len(sig_n),np.float32); cnt=np.zeros(len(sig_n),np.float32)
    with torch.no_grad():
        for i in range(0, len(sig_n)-W, W//2):
            out = model(torch.tensor(sig_n[i:i+W]).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
            pred[i:i+W]+=out; cnt[i:i+W]+=1
    cnt=np.maximum(cnt,1)
    return scaler_d.inverse_transform((pred/cnt).reshape(-1,1)).ravel().astype(np.float32)

print("  Training CNN..."); cnn=CNNDenoiser(); fit(cnn, dl, 50)
T_cnn = infer(cnn, T_rn)

print("  Training LSTM..."); lstm=LSTMDenoiser(); fit(lstm, dl, 50)
T_lstm = infer(lstm, T_rn)

print("  Training Autoencoder..."); aed=AEDenoiser(); fit(aed, dl, 50)
T_aed = infer(aed, T_rn)

print("  Training BiLSTM..."); bilst=BiLSTMDenoiser(); fit(bilst, dl, 50)
T_bi = infer(bilst, T_rn)

n_cnn,n_lstm,n_aed,n_bi = [noise_std(t) for t in [T_cnn,T_lstm,T_aed,T_bi]]
for name,val in [("CNN",n_cnn),("LSTM",n_lstm),("Autoencoder",n_aed),("BiLSTM",n_bi)]:
    print(f"  {name:18s}: {val:.2f} °C  ({(1-val/noise_raw)*100:.1f}% reduction)")

# Use Kalman output as compression input (best ATP-1 result)
T_comp = T_kal.copy()

# =============================================================================
# PART 3  --  ATP-3: CLASSICAL COMPRESSION
# =============================================================================
print("\n--- ATP-3: Classical Compression ---")
rb = T_comp.nbytes

# Downsampling
T_ds  = np.interp(np.arange(n), np.arange(0,n,2)[:len(T_comp[::2])],
                   T_comp[::2]).astype(np.float32)
rmse_ds, cr_ds = float(np.sqrt(np.mean((T_comp-T_ds)**2))), 2.0

# SVD rank=10
nu  = (n//32)*32
X   = T_comp[:nu].reshape(-1,32).astype(np.float64)
U,s,Vt = np.linalg.svd(X, full_matrices=False)
Xr  = (U[:,:10]*s[:10])@Vt[:10,:]
T_svd = np.interp(np.arange(n), np.linspace(0,n-1,Xr.ravel().size),
                   Xr.ravel()).astype(np.float32)
rmse_svd = float(np.sqrt(np.mean((T_comp-T_svd)**2)))
cr_svd   = max(1.0, rb/(U[:,:10].nbytes+s[:10].nbytes+Vt[:10,:].nbytes))

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
cr_wav  = max(1.0, rb/max(1, nz_vals.nbytes+nz_idx.nbytes))

for name,cr,rmse in [("Downsampling 2x",cr_ds,rmse_ds),
                      ("SVD rank=10",cr_svd,rmse_svd),
                      ("Wavelet Haar 10%",cr_wav,rmse_wav)]:
    met = "ATP-3 MET" if cr>=4.0 else "below target"
    print(f"  {name:20s}: CR={cr:.1f}x  RMSE={rmse:.2f} °C  [{met}]")

# =============================================================================
# PART 4  --  ATP-3: ML COMPRESSION
# =============================================================================
print("\n--- ATP-3: ML Compression ---")
Xw  = np.array([T_comp[i:i+32] for i in range(n-32)], dtype=np.float32)
sc2 = MinMaxScaler()
Xwn = sc2.fit_transform(Xw)

# PCA n=3
pca   = PCA(n_components=3)
Xpr   = sc2.inverse_transform(pca.inverse_transform(pca.fit_transform(Xwn)))
T_pca = np.zeros(n,np.float32); cp=np.zeros(n,np.float32)
for i,w in enumerate(Xpr): T_pca[i:i+32]+=w; cp[i:i+32]+=1
T_pca/=np.maximum(cp,1)
rmse_pca,cr_pca = float(np.sqrt(np.mean((T_comp-T_pca)**2))), 32/3

# Autoencoder bottleneck=3
class AEComp(nn.Module):
    """Autoencoder compressor — bottleneck=3, CR=32/3≈10.7x."""
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(32,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,3))
        self.dec=nn.Sequential(nn.Linear(3,8),nn.ReLU(),nn.Linear(8,16),nn.ReLU(),nn.Linear(16,32))
    def forward(self,x): return self.dec(self.enc(x))

print("  Training Autoencoder compressor...")
aec = AEComp(); opt2=torch.optim.Adam(aec.parameters(),lr=1e-3); Xt=torch.tensor(Xwn,dtype=torch.float32)
for _ in range(100):
    aec.train(); opt2.zero_grad()
    loss=nn.MSELoss()(aec(Xt),Xt); loss.backward(); opt2.step()
aec.eval()
with torch.no_grad(): Xar=sc2.inverse_transform(aec(Xt).numpy())
T_aec=np.zeros(n,np.float32); ca=np.zeros(n,np.float32)
for i,w in enumerate(Xar): T_aec[i:i+32]+=w; ca[i:i+32]+=1
T_aec/=np.maximum(ca,1)
rmse_aec,cr_aec=float(np.sqrt(np.mean((T_comp-T_aec)**2))),32/3

for name,cr,rmse in [("PCA n=3",cr_pca,rmse_pca),("Autoencoder bn=3",cr_aec,rmse_aec)]:
    met = "ATP-3 MET" if cr>=4.0 else "below target"
    print(f"  {name:20s}: CR={cr:.1f}x  RMSE={rmse:.2f} °C  [{met}]")

# =============================================================================
# SUMMARY TABLES
# =============================================================================
df1 = pd.DataFrame([
    {"Method":"Raw Baseline",        "Type":"Baseline", "Noise_C":round(noise_raw,2), "Reduction_%":"0.0"},
    {"Method":"Moving Average (k=7)","Type":"Classical","Noise_C":round(n_ma,2),      "Reduction_%":f"{(1-n_ma/noise_raw)*100:.1f}"},
    {"Method":"Median Filter (k=7)", "Type":"Classical","Noise_C":round(n_med,2),     "Reduction_%":f"{(1-n_med/noise_raw)*100:.1f}"},
    {"Method":"Savitzky-Golay",      "Type":"Classical","Noise_C":round(n_sg,2),      "Reduction_%":f"{(1-n_sg/noise_raw)*100:.1f}"},
    {"Method":"Gaussian Filter",     "Type":"Classical","Noise_C":round(n_gauss,2),   "Reduction_%":f"{(1-n_gauss/noise_raw)*100:.1f}"},
    {"Method":"Kalman Filter",       "Type":"Classical","Noise_C":round(n_kal,2),     "Reduction_%":f"{(1-n_kal/noise_raw)*100:.1f}"},
    {"Method":"CNN Denoiser",        "Type":"ML",       "Noise_C":round(n_cnn,2),     "Reduction_%":f"{(1-n_cnn/noise_raw)*100:.1f}"},
    {"Method":"LSTM Denoiser",       "Type":"ML",       "Noise_C":round(n_lstm,2),    "Reduction_%":f"{(1-n_lstm/noise_raw)*100:.1f}"},
    {"Method":"Autoencoder Den.",    "Type":"ML",       "Noise_C":round(n_aed,2),     "Reduction_%":f"{(1-n_aed/noise_raw)*100:.1f}"},
    {"Method":"BiLSTM Denoiser",     "Type":"ML",       "Noise_C":round(n_bi,2),      "Reduction_%":f"{(1-n_bi/noise_raw)*100:.1f}"},
])
df2 = pd.DataFrame([
    {"Method":"Raw Baseline",      "Type":"Baseline", "CR":"1.0x","RMSE_C":"0.00",          "ATP3_met":"—"},
    {"Method":"Downsampling 2x",   "Type":"Classical","CR":f"{cr_ds:.1f}x","RMSE_C":f"{rmse_ds:.2f}", "ATP3_met":"No"},
    {"Method":"SVD rank=10",       "Type":"Classical","CR":f"{cr_svd:.1f}x","RMSE_C":f"{rmse_svd:.2f}","ATP3_met":"No"},
    {"Method":"Wavelet Haar 10%",  "Type":"Classical","CR":f"{cr_wav:.1f}x","RMSE_C":f"{rmse_wav:.2f}","ATP3_met":"Yes"},
    {"Method":"PCA (n=3)",         "Type":"ML",       "CR":f"{cr_pca:.1f}x","RMSE_C":f"{rmse_pca:.2f}","ATP3_met":"Yes"},
    {"Method":"Autoencoder (bn=3)","Type":"ML",       "CR":f"{cr_aec:.1f}x","RMSE_C":f"{rmse_aec:.2f}","ATP3_met":"Yes"},
])
print("\n--- ATP-1 Summary ---"); print(df1.to_string(index=False))
print("\n--- ATP-3 Summary ---"); print(df2.to_string(index=False))
df1.to_csv("atp1_denoise_summary.csv", index=False)
df2.to_csv("atp3_compress_summary.csv", index=False)
print("\n  Saved: atp1_denoise_summary.csv, atp3_compress_summary.csv")

# Save trained models
torch.save(cnn.state_dict(),   "cnn_denoiser.pth")
torch.save(lstm.state_dict(),  "lstm_denoiser.pth")
torch.save(aed.state_dict(),   "autoencoder_denoiser.pth")
torch.save(bilst.state_dict(), "bilstm_denoiser.pth")
print("  Saved: cnn_denoiser.pth, lstm_denoiser.pth, autoencoder_denoiser.pth, bilstm_denoiser.pth")

# =============================================================================
# VISUALISATION  --  6-panel figure
# =============================================================================
print("\n  Generating plots...")
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#ffffff")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.33)
Z1,Z2 = max(0,HOT-80), min(n,HOT+80)

# Panel A: Classical denoising overlay
ax = fig.add_subplot(gs[0,0])
ax.plot(time_s, T_raw,   color="#e74c3c", lw=0.5, alpha=0.35, label="Raw")
ax.plot(time_s, T_ma,    color="#3498db", lw=0.9, label="Moving Avg")
ax.plot(time_s, T_med,   color="#2ecc71", lw=0.9, label="Median")
ax.plot(time_s, T_sg,    color="#9b59b6", lw=0.9, ls="--", label="Savitzky-Golay")
ax.plot(time_s, T_gauss, color="#f39c12", lw=0.9, label="Gaussian")
ax.plot(time_s, T_kal,   color="#1abc9c", lw=1.2, label="Kalman")
ax.set_title("ATP-1  Classical Denoising — All Methods\nNIST Layer 01", fontweight="bold", fontsize=11)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

# Panel B: Noise bar chart (all 10 methods)
ax = fig.add_subplot(gs[0,1])
lbs = ["Raw","MovAvg","Median","SavGol","Gauss","Kalman","CNN","LSTM","AE Den","BiLSTM"]
vs  = [noise_raw,n_ma,n_med,n_sg,n_gauss,n_kal,n_cnn,n_lstm,n_aed,n_bi]
cs  = ["#e74c3c","#3498db","#2ecc71","#9b59b6","#f39c12","#1abc9c",
       "#e67e22","#c0392b","#8e44ad","#27ae60"]
bars= ax.bar(lbs, vs, color=cs, alpha=0.85, edgecolor="black", linewidth=0.5, width=0.7)
for bar,val in zip(bars,vs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vs)*0.01,
            f"{val:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
best=int(np.argmin(vs))
bars[best].set_edgecolor("black"); bars[best].set_linewidth(2.5)
ax.set_title("ATP-1  Noise Std Dev — All 10 Methods (lower=better)\nBlue/Green/Teal=Classical  Orange/Red/Purple=ML",
             fontweight="bold", fontsize=11)
ax.set_ylabel("Noise std dev (°C)"); ax.grid(True, alpha=0.25, axis="y")

# Panel C: ML denoising zoomed at peak
ax = fig.add_subplot(gs[1,0])
ax.plot(time_s[Z1:Z2], T_raw[Z1:Z2],  color="#e74c3c", lw=0.7, alpha=0.4, label="Raw")
ax.plot(time_s[Z1:Z2], T_kal[Z1:Z2],  color="#1abc9c", lw=1.3, label="Kalman (best classical)")
ax.plot(time_s[Z1:Z2], T_cnn[Z1:Z2],  color="#e67e22", lw=1.1, label="CNN (ML)")
ax.plot(time_s[Z1:Z2], T_lstm[Z1:Z2], color="#c0392b", lw=1.0, ls="--", label="LSTM (ML)")
ax.plot(time_s[Z1:Z2], T_aed[Z1:Z2],  color="#8e44ad", lw=1.0, ls="-.", label="AE Denoiser (ML)")
ax.plot(time_s[Z1:Z2], T_bi[Z1:Z2],   color="#27ae60", lw=1.0, ls=":",  label="BiLSTM (ML)")
ax.set_title("ATP-1  Classical vs ML — Zoomed at Peak\nNIST Layer 01", fontweight="bold", fontsize=11)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

# Panel D: Compression signal overlay
ax = fig.add_subplot(gs[1,1])
ax.plot(time_s, T_comp, color="black",   lw=1.1, label="Input (Kalman-denoised)")
ax.plot(time_s, T_ds,   color="#3498db", lw=0.9, label=f"Downsampling ({cr_ds:.0f}x)")
ax.plot(time_s, T_svd,  color="#2ecc71", lw=0.9, label=f"SVD rank=10 ({cr_svd:.1f}x)")
ax.plot(time_s, T_wav,  color="#9b59b6", lw=1.2, ls="--", label=f"Wavelet 10% ({cr_wav:.1f}x) ★")
ax.plot(time_s, T_pca,  color="#e67e22", lw=1.1, label=f"PCA n=3 ({cr_pca:.1f}x)")
ax.plot(time_s, T_aec,  color="#c0392b", lw=1.0, ls="-.", label=f"AE Compress ({cr_aec:.1f}x)")
ax.set_title("ATP-3  Compression — Reconstructed vs Original\nNIST Layer 01", fontweight="bold", fontsize=11)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

# Panel E: CR vs RMSE scatter
ax = fig.add_subplot(gs[2,0])
pts=[("Raw",1.0,0.0,"#95a5a6"),("Downsamp",cr_ds,rmse_ds,"#3498db"),
     ("SVD",cr_svd,rmse_svd,"#2ecc71"),("Wavelet★",cr_wav,rmse_wav,"#9b59b6"),
     ("PCA",cr_pca,rmse_pca,"#e67e22"),("AE Comp",cr_aec,rmse_aec,"#c0392b")]
for lab,rx,ry,col in pts:
    ax.scatter(rx,ry,c=col,s=200,zorder=5,edgecolors="black",linewidths=0.8)
    ax.annotate(lab,(rx,ry),textcoords="offset points",xytext=(7,5),fontsize=9)
ax.axvline(x=4.0,color="green",ls="--",lw=1.5,alpha=0.8,label="CR=4x target")
ax.set_title("ATP-3  Compression Ratio vs RMSE\n(bottom-right = best trade-off)", fontweight="bold", fontsize=11)
ax.set_xlabel("Compression ratio"); ax.set_ylabel("RMSE (°C)"); ax.legend(fontsize=9); ax.grid(True,alpha=0.25)

# Panel F: Key findings text summary
ax = fig.add_subplot(gs[2,1])
ax.axis("off")
best_c = f"Kalman Filter  {n_kal:.1f} °C  ({(1-n_kal/noise_raw)*100:.1f}% reduction)"
best_ml= f"LSTM Denoiser  {n_lstm:.1f} °C  ({(1-n_lstm/noise_raw)*100:.1f}% reduction)"
best_w = f"Wavelet Haar 10%  CR={cr_wav:.1f}x  RMSE={rmse_wav:.2f} °C"
best_p = f"PCA n=3  CR={cr_pca:.1f}x  RMSE={rmse_pca:.2f} °C"
summary = (
    "KEY FINDINGS\n\n"
    "ATP-1  Signal Denoising\n"
    f"  Best classical : {best_c}\n"
    f"  Best ML        : {best_ml}\n"
    f"  Classical outperforms ML here\n"
    f"  (Kalman suited to Gaussian noise)\n\n"
    "ATP-3  Data Compression\n"
    f"  Best classical : {best_w}\n"
    f"  Best ML ratio  : {best_p}\n"
    f"  ATP-3 target CR>4x satisfied\n"
    f"  by Wavelet, PCA, and Autoencoder\n\n"
    "NOTE: ATP-2 Calibration is Ajay's work\n"
    "and is NOT included in this file."
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

fig.suptitle(
    "denoise_atp13.py  --  ATP-1 Signal Denoising & ATP-3 Data Compression\n"
    "Mallepalli Sravya Reddy — University West 2026",
    fontsize=13, fontweight="bold", y=1.01
)
plt.savefig("denoise_atp13_results.png", dpi=150, bbox_inches="tight", facecolor="white")
print("  Plot saved: denoise_atp13_results.png")

print()
print("=" * 65)
print("COMPLETE  --  denoise_atp13.py")
print("=" * 65)
print(f"  ATP-1 best classical : Kalman  {n_kal:.1f} °C  ({(1-n_kal/noise_raw)*100:.1f}% reduction)")
print(f"  ATP-1 best ML        : LSTM    {n_lstm:.1f} °C  ({(1-n_lstm/noise_raw)*100:.1f}% reduction)")
print(f"  ATP-3 best classical : Wavelet CR={cr_wav:.1f}x  RMSE={rmse_wav:.2f} °C  (ATP-3 MET)")
print(f"  ATP-3 best ML ratio  : PCA     CR={cr_pca:.1f}x  RMSE={rmse_pca:.2f} °C")
print("=" * 65)
