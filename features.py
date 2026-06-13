"""Signal-processing feature extraction (numpy/scipy only -- no MNE needed).

Band-power features: band-pass each trial, then take log-variance per channel.
Log-variance of a band-passed signal is (up to a constant) the log band-power,
the classic feature for sensorimotor rhythm decoding.
"""
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass(X, sfreq, l_freq, h_freq, order=4):
    """Zero-phase Butterworth band-pass along the last (time) axis."""
    nyq = sfreq / 2.0
    b, a = butter(order, [l_freq / nyq, h_freq / nyq], btype="band")
    return filtfilt(b, a, X, axis=-1)


def band_power_features(X, sfreq, bands=((8, 13), (13, 30)), order=4):
    """X (n_trials, n_channels, n_times) -> (n_trials, n_channels * n_bands)."""
    feats = []
    for (lo, hi) in bands:
        Xf = bandpass(X, sfreq, lo, hi, order=order)
        log_var = np.log(np.var(Xf, axis=-1) + 1e-12)     # (n_trials, n_channels)
        feats.append(log_var)
    return np.concatenate(feats, axis=1)
