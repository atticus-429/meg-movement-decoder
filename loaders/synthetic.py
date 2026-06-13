"""Synthetic MEG-Motor generator.

Produces data with the SAME array contract as the real HCP loader so the whole
pipeline (features -> decode -> evaluate) can be smoke-tested end-to-end on a
laptop with no data and no GPU.

The signal is physiologically motivated: each "movement" induces a beta-band
event-related desynchronization (ERD, a power DROP) over the sensors
CONTRALATERAL to the moving hand. A real laterality decoder should therefore
recover well-above-chance accuracy here -- which is exactly what verifies the
decoder is doing something real, not just shuffling shapes.
"""
import numpy as np


def make_synthetic_motor(n_trials_per_class=80, n_channels=64, sfreq=250.0,
                         tmin=-0.5, tmax=1.5, classes=("LH", "RH"),
                         noise=3.0, lapse_rate=0.12, seed=0):
    rng = np.random.default_rng(seed)
    n_times = int(round((tmax - tmin) * sfreq))
    times = tmin + np.arange(n_times) / sfreq

    # Split sensors into two hemispheres (first half = left hemi, second = right).
    half = n_channels // 2
    left_hemi = np.arange(0, half)
    right_hemi = np.arange(half, n_channels)

    beta_f = 20.0                       # Hz, sensorimotor beta

    X, y = [], []
    for ci, cname in enumerate(classes):
        # Contralateral hemisphere shows the ERD (power drop) during movement.
        erd_hemi = right_hemi if cname in ("LH", "LF") else left_hemi
        for _ in range(n_trials_per_class):
            sig = rng.standard_normal((n_channels, n_times)) * noise
            phase = rng.uniform(0, 2 * np.pi, n_channels)[:, None]
            beta = np.sin(2 * np.pi * beta_f * times[None, :] + phase)

            # trial-to-trial variability: jittered onset/duration and variable
            # ERD depth over the (full) contralateral sensorimotor hemisphere.
            onset = rng.normal(-0.1, 0.08)
            dur = rng.uniform(1.1, 1.5)
            mv_idx = np.where((times >= onset) & (times <= onset + dur))[0]
            depth = rng.uniform(0.35, 0.55)            # residual beta amplitude

            amp = np.ones((n_channels, n_times))
            evoked = np.zeros((n_channels, n_times))
            # "lapse" trials carry no movement signal (subject didn't move
            # clearly / mislabel) -> caps achievable accuracy below 100%, as in
            # real behavioral data.
            has_move = len(mv_idx) and rng.random() > lapse_rate
            if has_move:
                # (1) induced beta ERD: contralateral power DROP (what CSP /
                #     band-power decoders use)
                amp[np.ix_(erd_hemi, mv_idx)] = depth
                # (2) movement-evoked field: a brief PHASE-LOCKED contralateral
                #     deflection at onset (what a time-domain CNN like EEGNet uses)
                bump = np.exp(-0.5 * ((times - onset) / 0.05) ** 2)
                evoked[erd_hemi, :] = 2.0 * bump[None, :]
            trial = sig + 2.0 * amp * beta + evoked
            X.append(trial.astype(np.float32))
            y.append(ci)

    X = np.asarray(X)
    y = np.asarray(y, dtype=int)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], float(sfreq), list(classes)
