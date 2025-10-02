#!/usr/bin/env python3
import os, sys, glob
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
print(os.listdir("data/raw"))

# ========== HELPER FUNCTIONS ==========

def window_signal(signal, fs, win_size=60, step=30):
    """Window a 1D signal into overlapping segments"""
    step_size = step * fs
    win_len = win_size * fs
    for i in range(0, len(signal) - win_len, step_size):
        yield signal[i:i+win_len]

def extract_stats(window):
    """Basic statistics for any signal window"""
    return {
        "mean": np.mean(window),
        "std": np.std(window),
        "min": np.min(window),
        "max": np.max(window),
        "median": np.median(window),
        "iqr": np.percentile(window, 75) - np.percentile(window, 25),
    }

def extract_freq(window, fs):
    """Frequency features using Welch"""
    f, pxx = welch(window, fs=fs)
    power = np.sum(pxx)
    entropy_val = entropy(pxx / np.sum(pxx))
    return {"power": power, "spec_entropy": entropy_val}

# ========== MODALITY-SPECIFIC FEATURES ==========

def process_eda(path, fs=4, win_size=60, step=30):
    raw = pd.read_csv(path, header=None).squeeze()
    signal = raw.iloc[2:].astype(float).to_numpy()
    feats = []
    for win in window_signal(signal, fs, win_size, step):
        f = {}
        f.update({f"eda_{k}": v for k, v in extract_stats(win).items()})
        f.update({f"eda_{k}": v for k, v in extract_freq(win, fs).items()})
        feats.append(f)
    return pd.DataFrame(feats)

def process_hr(path, fs=1, win_size=60, step=30):
    raw = pd.read_csv(path, header=None).squeeze().to_numpy()
    signal = raw.astype(float)
    feats = []
    for win in window_signal(signal, fs, win_size, step):
        f = {f"hr_{k}": v for k, v in extract_stats(win).items()}
        feats.append(f)
    return pd.DataFrame(feats)

def process_temp(path, fs=4, win_size=60, step=30):
    raw = pd.read_csv(path, header=None).squeeze()
    signal = raw.iloc[2:].astype(float).to_numpy()
    feats = []
    for win in window_signal(signal, fs, win_size, step):
        f = {f"temp_{k}": v for k, v in extract_stats(win).items()}
        feats.append(f)
    return pd.DataFrame(feats)

def process_acc(path, fs=32, win_size=60, step=30):
    raw = pd.read_csv(path, header=None).to_numpy()[2:]
    signal = raw.astype(float)
    x, y, z = signal[:,0], signal[:,1], signal[:,2]
    feats = []
    for wx, wy, wz in zip(window_signal(x, fs, win_size, step),
        window_signal(y, fs, win_size, step),
        window_signal(z, fs, win_size, step)):
        mag = np.sqrt(wx**2 + wy**2 + wz**2)
        f = {}
        f.update({f"accx_{k}": v for k, v in extract_stats(wx).items()})
        f.update({f"accy_{k}": v for k, v in extract_stats(wy).items()})
        f.update({f"accz_{k}": v for k, v in extract_stats(wz).items()})
        f.update({f"acc_mag_{k}": v for k, v in extract_stats(mag).items()})
        feats.append(f)
    return pd.DataFrame(feats)

def process_ibi(path, win_size=60, step=30):
    raw = pd.read_csv(path, header=None).to_numpy()
    ibi_vals = raw[:,0].astype(float)
    feats = []
    for win in window_signal(ibi_vals, fs=1, win_size=win_size, step=step):
        mean_rr = np.mean(win)
        sdnn = np.std(win)
        rmssd = np.sqrt(np.mean(np.diff(win)**2))
        f = {"ibi_mean": mean_rr, "ibi_sdnn": sdnn, "ibi_rmssd": rmssd}
        feats.append(f)
    return pd.DataFrame(feats)

# ========== MAIN PIPELINE ==========

def preprocess_subject(subj_dir, out_dir="processed", subj_id="S02"):
    print(f"[INFO] Processing {subj_id}")
    os.makedirs(out_dir, exist_ok=True)

    dfs = []
    dfs.append(process_eda(os.path.join(subj_dir, "EDA.csv")))
    dfs.append(process_hr(os.path.join(subj_dir, "HR.csv")))
    dfs.append(process_temp(os.path.join(subj_dir, "TEMP.csv")))
    dfs.append(process_acc(os.path.join(subj_dir, "ACC.csv")))
    dfs.append(process_ibi(os.path.join(subj_dir, "IBI.csv")))

    # Align by index
    df = pd.concat(dfs, axis=1).dropna().reset_index(drop=True)
    STEP=30
    df.insert(0, "window_id", range(len(df)))
    df.insert(1, "time_sec", df["window_id"] * STEP)

    # Example phase mapping (seconds from start, replace with tags.csv later)
    phases = {
        0: (0, 300),    # baseline
        1: (301, 900),  # stress
        2: (901, 1200)  # amusement
    }

    labels = []
    for idx in range(len(df)):
        t = df.loc[idx, "time_sec"]
        if phases[0][0] <= t <= phases[0][1]:
            labels.append(0)
        elif phases[1][0] <= t <= phases[1][1]:
            labels.append(1)
        elif phases[2][0] <= t <= phases[2][1]:
            labels.append(2)
        else:
            labels.append(-1)  # unknown/outside
    df["label"] = labels 

    out_path = os.path.join(out_dir, f"{subj_id}.csv")
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved {out_path} with shape {df.shape}")
    return df

if __name__ == "__main__":
    subj_dir = "data/raw/S02"   # change to your subject folder
    preprocess_subject(subj_dir, out_dir="data/processed/WESAD_wrist", subj_id="S02")
