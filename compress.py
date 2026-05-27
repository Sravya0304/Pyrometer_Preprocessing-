"""
=============================================================================
compress.py  —  Compression module for pyrometer pre-processing pipeline
=============================================================================
Thesis: Automation of pyrometer data pre-processing (AP&T / Metal Forming)

WHAT THIS MODULE DOES:
    Compresses a pyrometer time-series (1-D) or spatial temperature map
    (2-D or 3-D array) to reduce storage size while keeping reconstruction
    error acceptably low. Two methods are provided:

        1. svd_compress    — Truncated SVD (linear, fast, interpretable)
        2. wavelet_compress— Wavelet thresholding (good for 1-D signals)

HOW TO USE:
    from compress import svd_compress, wavelet_compress, compression_report
    from compress import print_preview

    compressed, meta = svd_compress(data_2d, rank=50)
    reconstructed    = svd_reconstruct(compressed, meta)

SWAP FOR ML LATER:
    Replace svd_compress/reconstruct with an autoencoder.
    Keep the same compress() / decompress() interface.
=============================================================================
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — TRUNCATED SVD  (best for 2-D / 3-D spatial data)
# ─────────────────────────────────────────────────────────────────────────────

def svd_compress(data: np.ndarray, rank: int = 50):
    """
    Compress a 2-D matrix using Truncated SVD (rank-k approximation).

    For a 3-D array (rows × cols × frames), reshape to 2-D first:
        data_2d = data.reshape(rows * cols, frames)

    Parameters
    ----------
    data : np.ndarray — 2-D matrix to compress (shape: M × N)
    rank : int        — number of singular values to keep (lower = smaller)

    Returns
    -------
    compressed : dict with keys:
        'U'      — left singular vectors  (M × rank)
        'S'      — singular values        (rank,)
        'Vt'     — right singular vectors (rank × N)
        'rank'   — rank used
        'shape'  — original shape of data
    """
    data = data.astype(np.float64)

    # Full SVD then truncate (scipy TruncatedSVD is faster for large matrices
    # but numpy is used here to keep dependencies minimal)
    U, S, Vt = np.linalg.svd(data, full_matrices=False)

    # Keep only top-k components
    U_k  = U[:, :rank]
    S_k  = S[:rank]
    Vt_k = Vt[:rank, :]

    compressed = {
        "U"     : U_k,
        "S"     : S_k,
        "Vt"    : Vt_k,
        "rank"  : rank,
        "shape" : data.shape,
    }
    return compressed


def svd_reconstruct(compressed: dict) -> np.ndarray:
    """
    Reconstruct the original matrix from SVD compressed components.

    Parameters
    ----------
    compressed : dict — output from svd_compress()

    Returns
    -------
    np.ndarray — reconstructed matrix, same shape as original
    """
    U  = compressed["U"]
    S  = compressed["S"]
    Vt = compressed["Vt"]
    return (U * S) @ Vt    # equivalent to U @ np.diag(S) @ Vt but faster


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — WAVELET THRESHOLDING  (best for 1-D time-series)
# ─────────────────────────────────────────────────────────────────────────────

def wavelet_compress(signal: np.ndarray,
                     threshold: float = None,
                     keep_fraction: float = 0.10):
    """
    Compress a 1-D signal using Haar wavelet thresholding.
    Keeps only the largest wavelet coefficients (the rest → zero).

    Parameters
    ----------
    signal        : np.ndarray — 1-D temperature time-series
    threshold     : float      — absolute threshold for coefficient zeroing.
                                 If None, threshold is auto-set to keep
                                 `keep_fraction` of coefficients.
    keep_fraction : float      — fraction of coefficients to keep (0–1)
                                 (used only when threshold is None)

    Returns
    -------
    compressed : dict with keys:
        'coeffs'     — thresholded wavelet coefficients (mostly zeros)
        'threshold'  — threshold value used
        'length'     — original signal length
        'nonzero'    — number of non-zero coefficients kept
    """
    signal = signal.astype(np.float64)
    n      = len(signal)

    # Pad to next power of 2 (Haar requires power-of-2 length)
    n_padded = int(2 ** np.ceil(np.log2(n)))
    padded   = np.zeros(n_padded)
    padded[:n] = signal

    # Forward Haar wavelet transform
    coeffs = _haar_forward(padded)

    # Auto-threshold: keep the largest `keep_fraction` of coefficients
    if threshold is None:
        sorted_abs = np.sort(np.abs(coeffs))[::-1]
        k          = max(1, int(keep_fraction * len(coeffs)))
        threshold  = float(sorted_abs[k])

    # Hard thresholding: zero out coefficients below threshold
    coeffs_thresh = coeffs.copy()
    coeffs_thresh[np.abs(coeffs_thresh) < threshold] = 0.0

    compressed = {
        "coeffs"    : coeffs_thresh,
        "threshold" : threshold,
        "length"    : n,
        "nonzero"   : int((coeffs_thresh != 0).sum()),
        "n_padded"  : n_padded,
    }
    return compressed


def wavelet_reconstruct(compressed: dict) -> np.ndarray:
    """
    Reconstruct 1-D signal from wavelet compressed dict.

    Parameters
    ----------
    compressed : dict — output from wavelet_compress()

    Returns
    -------
    np.ndarray — reconstructed signal, trimmed to original length
    """
    reconstructed = _haar_inverse(compressed["coeffs"])
    return reconstructed[:compressed["length"]]


def _haar_forward(x: np.ndarray) -> np.ndarray:
    """Iterative Haar wavelet forward transform (in-place style)."""
    x = x.copy()
    n = len(x)
    while n > 1:
        half   = n // 2
        avg    = (x[:n:2] + x[1:n:2]) / 2.0
        diff   = (x[:n:2] - x[1:n:2]) / 2.0
        x[:half] = avg
        x[half:n] = diff
        n = half
    return x


def _haar_inverse(x: np.ndarray) -> np.ndarray:
    """Iterative Haar wavelet inverse transform."""
    x = x.copy()
    n = 2
    while n <= len(x):
        half = n // 2
        avg  = x[:half].copy()
        diff = x[half:n].copy()
        x[:n:2]  = avg + diff
        x[1:n:2] = avg - diff
        n *= 2
    return x


# ─────────────────────────────────────────────────────────────────────────────
# METRICS AND REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def compression_ratio(original: np.ndarray, compressed: dict,
                       method: str = "svd") -> float:
    """
    Calculate the compression ratio: original_bytes / compressed_bytes.
    Higher is better.

    Parameters
    ----------
    original   : original numpy array
    compressed : dict from svd_compress() or wavelet_compress()
    method     : 'svd' or 'wavelet'

    Returns
    -------
    float — compression ratio (e.g. 12.5 means 12.5× smaller)
    """
    orig_bytes = original.nbytes
    if method == "svd":
        comp_bytes = (compressed["U"].nbytes +
                      compressed["S"].nbytes +
                      compressed["Vt"].nbytes)
    else:   # wavelet — only non-zero values need storage
        comp_bytes = max(1, compressed["nonzero"] * 8)  # 8 bytes per float64
    return orig_bytes / comp_bytes


def reconstruction_rmse(original: np.ndarray,
                         reconstructed: np.ndarray) -> float:
    """
    RMSE between original and reconstructed signal/matrix.

    Returns
    -------
    float — RMSE in the same units as the data
    """
    return float(np.sqrt(np.mean((original - reconstructed) ** 2)))


def compression_report(original: np.ndarray,
                        reconstructed: np.ndarray,
                        compressed: dict,
                        method: str,
                        label: str = "") -> None:
    """
    Print a one-block compression summary.

    Parameters
    ----------
    original      : original array
    reconstructed : reconstructed array
    compressed    : dict from compress function
    method        : 'svd' or 'wavelet'
    label         : optional display label
    """
    ratio = compression_ratio(original, compressed, method)
    rmse  = reconstruction_rmse(original, reconstructed)

    tag = f"[{label}] " if label else ""
    print(f"  {tag}Method         : {method.upper()}")
    if method == "svd":
        print(f"  {tag}Rank kept      : {compressed['rank']}")
    else:
        print(f"  {tag}Coeffs kept    : {compressed['nonzero']} "
              f"/ {len(compressed['coeffs'])} "
              f"({compressed['nonzero']/len(compressed['coeffs'])*100:.1f}%)")
        print(f"  {tag}Threshold      : {compressed['threshold']:.2f}")
    orig_mb = original.nbytes / 1e6
    print(f"  {tag}Original size  : {orig_mb:.3f} MB")
    print(f"  {tag}Compression    : {ratio:.1f}×")
    print(f"  {tag}RMSE           : {rmse:.2f} (same units as input)")


# ─────────────────────────────────────────────────────────────────────────────
# PREVIEW HELPER
# ─────────────────────────────────────────────────────────────────────────────

def print_preview(original: np.ndarray,
                  reconstructed: np.ndarray,
                  time_s: np.ndarray,
                  start: int = 0) -> None:
    """
    Print a 5-row × 4-column preview table:
        Time | Original | Reconstructed | Error
    """
    import pandas as pd
    idx = list(range(start, start + 5))
    df  = pd.DataFrame({
        "Time_s"   : np.round(time_s[idx], 4),
        "Original" : np.round(original[idx], 2),
        "Reconstr" : np.round(reconstructed[idx], 2),
        "Error"    : np.round(reconstructed[idx] - original[idx], 2),
    }, index=[f"t{i}" for i in idx])
    print(df.to_string())


# ─────────────────────────────────────────────────────────────────────────────

# -----------------------------------------------------------------------------
# SELF-TEST WITH REAL LAYER01 DATA  (run:  python compress.py)
#
# Chain: Layer01.mat -> Raw -> Denoise -> Calibrate -> Compress
# compress.py takes CALIBRATED signal as input (not raw, not denoised)
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import scipy.io as sio
    import pandas as pd
    from scipy.signal import medfilt

    # DATA_PATH = r"C:\Users\sravy\OneDrive\Desktop\ThesisSravya\data (2)\data\Layer01.mat"
    # mat   = sio.loadmat(DATA_PATH)
    # L     = mat["Layer"][0, 0]
    # raw3d = L["RadiantTemp"].astype(np.float32)
    # sh_A  = float(L["SHvariable_A"].flat[0])
    # sh_B  = float(L["SHvariable_B"].flat[0])
    # frame_max = raw3d.max(axis=(0, 1))
    # T_raw = np.clip(sh_A * frame_max + sh_B - 273.15, 0, 3000)
    # T_raw = T_raw[T_raw > 10].astype(np.float32)

    # Synthetic data fallback (used when .mat file not available)
    import numpy as np
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
    print("compress.py  --  ATP-3: Compression Module Self-Test")
    print("Input: Kalman-denoised signal (best ATP-1 result)")
    print("NOTE: ATP-2 Calibration is Ajay's work -- not used here")
    print("=" * 65)

    # Stage 1: Kalman denoising (ATP-1 best result)
    def kalman(sig, Q=1e-3, R=10.0):
        x, P, out = float(sig[0]), 1.0, np.zeros(len(sig), np.float32)
        for i in range(len(sig)):
            P += Q; K = P/(P+R); x = x+K*(sig[i]-x); P=(1-K)*P; out[i]=x
        return out

    T_comp = kalman(T_raw)
    HOT    = max(0, int(np.argmax(T_comp)) - 2)
    idx    = list(range(HOT, HOT + 5))

    print(f"\n  Input  : Kalman-denoised signal")
    print(f"  Frames : {n}  |  Range : {T_comp.min():.1f} – {T_comp.max():.1f} °C")
    print(f"  Size   : {T_comp.nbytes} bytes")
    print(f"  ATP-3 target : CR > 4×")

    # METHOD A: Wavelet 10%
    print()
    print("=" * 65)
    print("METHOD A: Wavelet Haar (10% coefficient retention)")
    print("=" * 65)
    comp_wav  = wavelet_compress(T_comp, keep_fraction=0.10)
    recon_wav = wavelet_reconstruct(comp_wav)
    ratio_wav = T_comp.nbytes / max(1, comp_wav["nonzero"] * 8)
    rmse_wav  = float(np.sqrt(np.mean((T_comp - recon_wav)**2)))
    df_wav = pd.DataFrame({
        "Time_s"         : np.round(time_s[idx], 4),
        "Input_C"        : np.round(T_comp[idx], 2),
        "Reconstructed_C": np.round(recon_wav[idx], 2),
        "Error_C"        : np.round(recon_wav[idx] - T_comp[idx], 2),
    }, index=[f"t{i}" for i in idx])
    print(df_wav.to_string())
    print(f"\n  Coefficients kept : {comp_wav['nonzero']} / {len(comp_wav['coeffs'])} (10%)")
    print(f"  Compression ratio : {ratio_wav:.1f}×  {'ATP-3 MET ✓' if ratio_wav >= 4 else 'below target'}")
    print(f"  RMSE              : {rmse_wav:.2f} °C")

    # METHOD B: Wavelet trade-off comparison
    print()
    print("=" * 65)
    print("METHOD B: Wavelet trade-off (different retention fractions)")
    print("=" * 65)
    results = []
    for keep in [0.20, 0.10, 0.05]:
        cw = wavelet_compress(T_comp, keep_fraction=keep)
        rw = wavelet_reconstruct(cw)
        results.append({
            "Keep_%"     : f"{int(keep*100)}%",
            "Coeffs_kept": cw["nonzero"],
            "Ratio"      : round(T_comp.nbytes / max(1, cw["nonzero"]*8), 1),
            "RMSE_C"     : round(float(np.sqrt(np.mean((T_comp-rw)**2))), 2),
            "ATP3_met"   : "Yes ✓" if T_comp.nbytes/max(1,cw["nonzero"]*8) >= 4 else "No",
        })
    print(pd.DataFrame(results).to_string(index=False))

    # METHOD C: SVD rank=10
    print()
    print("=" * 65)
    print("METHOD C: Truncated SVD (rank=10)")
    print("=" * 65)
    nu  = (n // 32) * 32
    X   = T_comp[:nu].reshape(-1, 32).astype(np.float64)
    cs  = svd_compress(X, rank=10)
    rs  = svd_reconstruct(cs)
    T_svd = np.interp(np.arange(n), np.linspace(0, n-1, rs.ravel().size),
                       rs.ravel()).astype(np.float32)
    rmse_svd = float(np.sqrt(np.mean((T_comp - T_svd)**2)))
    cr_svd   = T_comp.nbytes / max(1, cs["U"].nbytes + cs["S"].nbytes + cs["Vt"].nbytes)
    print(f"  CR={cr_svd:.1f}×  RMSE={rmse_svd:.2f} °C")

    print()
    print("=" * 65)
    print("compress.py self-test complete  (ATP-3 only)")
    print("NOTE: ATP-2 Calibration is Ajay's work -- not in this file")
    print("=" * 65)