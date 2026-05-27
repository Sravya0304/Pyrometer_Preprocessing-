# Pyrometer Data Pre-processing — Signal Denoising & Data Compression

**Master's Thesis — University West, 2026**
**Author: Mallepalli Sravya Reddy**
**Programme: Master of Engineering Science with Specialization in AI and Automation**
**Industry Collaborator: AP&T (Advanced Pressure Technology)**
**Supervisors: Amit Kumar Mishra, Rashid Ali**
**Examiner: Fredrik Sikström**

---

## Individual Contribution

This repository contains the individual thesis contribution of **Mallepalli Sravya Reddy**, focusing on two stages of the pyrometer data pre-processing pipeline:

| Stage | Acceptance Test | Description |
|-------|----------------|-------------|
| **ATP-1** | Signal Denoising | Reduce noise in raw pyrometer signals while preserving genuine temperature events |
| **ATP-3** | Data Compression | Reduce storage footprint of processed time-series while preserving temperature accuracy |

> **Note:** Sensor calibration (ATP-2) is addressed in a separate parallel thesis contribution.

---

## Datasets

### NIST AMBench Dataset
- Source: National Institute of Standards and Technology (open-source)
- Process: Laser powder bed fusion (IN625, 195W, 800mm/s)
- Files: Layer01.mat to Layer10.mat (10 layers, 2065 frames per layer)
- Used for: Training and evaluating all ATP-1 and ATP-3 methods

### PODFAM Dataset (AP&T Industrial Data)
- Source: Aconity MIDI laser powder bed fusion machine (real two-colour pyrometer)
- Files: 10 point cloud scan files (000.280.pcd to 010.010.pcd)
- Used for: Validating classical methods on real industrial pyrometer data

---

## Methods Implemented

### ATP-1 Signal Denoising (10 methods)

| Method | Type | Key Parameter |
|--------|------|--------------|
| Raw Baseline | Baseline | — |
| Moving Average | Classical | k = 7 |
| Median Filter | Classical | k = 7 |
| Savitzky-Golay | Classical | window = 11, degree = 3 |
| Gaussian Filter | Classical | sigma = 3.0 |
| Kalman Filter | Classical | Q = 0.001, R = 10.0 |
| CNN Denoiser | ML | 4 Conv1d layers, 50 epochs |
| LSTM Denoiser | ML | 2 layers, hidden = 64, 50 epochs |
| Autoencoder Denoiser | ML | bottleneck = 8, 50 epochs |
| BiLSTM Denoiser | ML | 2 layers, hidden = 64, 50 epochs |

**Best result:** Kalman Filter — 89.5% noise reduction on NIST, 88.4% on PODFAM

### ATP-3 Data Compression (6 methods)

| Method | Type | Compression Ratio | RMSE (°C) | ATP-3 Target Met |
|--------|------|-------------------|-----------|-----------------|
| Raw Baseline | Baseline | 1.0× | 0.00 | — |
| Downsampling 2× | Classical | 2.0× | 21.97 | No |
| SVD rank=10 | Classical | 2.1× | 131.6 | No |
| Wavelet Haar 10% | Classical | 5.0× | 36.55 | **Yes ✓** |
| PCA (n=3) | ML | 10.7× | 58.10 | Yes ✓ |
| Autoencoder (bn=3) | ML | 10.7× | 170.80 | Yes ✓ |

**Best result:** Wavelet Haar 10% — CR = 5.0×, RMSE = 36.55°C (ATP-3 target: CR > 4×)

---

## Repository Structure

```
pyrometer-denoising-compression/
│
├── D1 — Data Loading & Pipeline Runner
│   └── layer01_pipeline.py
│
├── D2 — Modular Pipeline Modules
│   ├── denoise.py              # ATP-1 denoising module
│   ├── compress.py             # ATP-3 compression module
│   └── denoise_atp13.py        # Combined ATP-1 + ATP-3 standalone script
│
├── D3 — Classical vs ML Comparison
│   ├── classical_methods.py    # All 5 classical denoising + 4 compression methods
│   ├── ml_denoise.py           # CNN, LSTM, Autoencoder, BiLSTM denoisers
│   ├── ml_compress.py          # PCA, Autoencoder compressors
│   └── d3_comparison.py        # Full classical vs ML comparison (4-panel figure)
│
├── D4 — Visualisation Dashboard
│   └── visualise.py            # Multi-panel dashboard with event markers
│
├── D5 — Trade-off Analysis
│   └── d5_analysis.py          # Cross-method trade-off analysis
│
├── PODFAM — Industrial Dataset Validation
│   ├── load_podfam.py          # Single PODFAM file pipeline
│   ├── podfam_all_files.py     # All 10 PODFAM files evaluation
│   └── podfam_summary.py       # PODFAM summary and statistics
│
├── Results (CSV)
│   ├── atp1_denoise_summary.csv
│   ├── atp3_compress_summary.csv
│   ├── d3_denoise_summary.csv
│   ├── d3_compress_summary.csv
│   ├── d5_summary.csv
│   ├── podfam_all_results.csv
│   └── podfam_summary.csv
│
├── Results (PNG)
│   ├── classical_results.png
│   ├── d3_comparison.png
│   ├── d4_dashboard.png
│   ├── d5_analysis.png
│   ├── denoise_atp13_results.png
│   ├── ml_denoise_result.png
│   ├── ml_compress_result.png
│   ├── podfam_result.png
│   ├── podfam_all_results.png
│   └── 000.280_result.png ... 010.010_result.png (10 PODFAM files)
│
└── Trained Models
    ├── cnn_denoiser.pth
    ├── lstm_denoiser.pth
    ├── autoencoder_denoiser.pth
    ├── bilstm_denoiser.pth
    └── autoencoder_compressor.pth
```

---

## How to Run

### Requirements
```bash
pip install numpy scipy matplotlib pandas pywavelets scikit-learn torch
```

### Run order (recommended)

**Step 1 — Load NIST data and run full pipeline:**
```bash
python layer01_pipeline.py
```

**Step 2 — Run all denoising and compression methods:**
```bash
python denoise_atp13.py
```

**Step 3 — Classical vs ML comparison (D3):**
```bash
python classical_methods.py
python ml_denoise.py
python ml_compress.py
python d3_comparison.py
```

**Step 4 — Visualisation dashboard (D4):**
```bash
python visualise.py
```

**Step 5 — Trade-off analysis (D5):**
```bash
python d5_analysis.py
```

**Step 6 — PODFAM industrial dataset:**
```bash
python load_podfam.py
python podfam_all_files.py
python podfam_summary.py
```

> To use real NIST data: set `DATA_PATH` in each script to your local `Layer01.mat` path.
> All scripts include a synthetic data fallback so they run without the `.mat` file.

---

## Key Results

### ATP-1 Signal Denoising (NIST Layer 01)

| Best Classical | Best ML |
|----------------|---------|
| Kalman Filter: **89.5% noise reduction** | LSTM: 31.8% noise reduction |

### ATP-3 Data Compression (NIST Layer 01)

| Best Classical | Best ML Ratio |
|----------------|--------------|
| Wavelet 10%: **CR = 5.0×, RMSE = 36.55°C** | PCA n=3: CR = 10.7×, RMSE = 58.10°C |

### PODFAM Industrial Validation (10 files)

| Stage | Best Method | Result |
|-------|-------------|--------|
| ATP-1 | Kalman Filter | 88.4% noise reduction (every file) |
| ATP-3 | Wavelet 10% | CR = 3.1×, RMSE < 2°C (every file) |

---

## Evaluation Metrics

- **ATP-1:** Noise standard deviation (°C), Noise reduction (%)
- **ATP-3:** Compression ratio (CR), Reconstruction RMSE (°C)
- **ATP-3 target:** CR > 4× — satisfied by Wavelet, PCA, and Autoencoder

---

## References

- NIST AMBench Dataset: Yeung et al., NIST TN 2121, 2020
- Kalman Filter: Alfaouri & Daqrouq, American Journal of Applied Sciences, 2008
- CNN Architecture: LeCun et al., Neural Computation, 1989
- LSTM: Hochreiter & Schmidhuber, Neural Computation, 1997
- Wavelet Compression: Michau & Fink, Structural Health Monitoring, 2022
- Autoencoder Compression: Romeu et al., IEEE Sensors Journal, 2021
- Savitzky-Golay: Savitzky & Golay, Analytical Chemistry, 1964
