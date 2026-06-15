"""Loader for the Yeom et al. 2023 MEG 3D-reaching dataset.

  Yeom, Kim, Chung (2023), Scientific Data 10:552.
  figshare DOI 10.6084/m9.figshare.c.6431021 (CC-BY) -- no login, no agreement.

This is the PRIMARY Rung-1 dataset: cued, event-timed movement execution with 4
reach directions, directly downloadable (unlike HCP), loadable with plain scipy.

Dataset structure (the real files are MATLAB v7.3 -> read via mat73):
  - top-level `epoched_data` = a 4-cell array (one per reach direction); each cell
    is `channels x time x trials`. Plus an `info` dict = the MNE-Python measurement
    info (ch_names, sfreq, ...).
  - 319 channels: 306 MEG (102 magnetometers + 204 planar gradiometers),
    plus triggers, EOG and accelerometer.
  - ~30 trials/direction/session; window -1..+2 s from cue onset; sfreq 600.615 Hz.

Returns the standard contract:
  X (n_trials, n_channels, n_times) float32, y (n_trials,) int, sfreq, class_names

CHANNEL SELECTION: magnetometers/gradiometers are identified from the file's MNE
`info` ch_names (MEGIN convention: last digit 1 = magnetometer, 2/3 = planar
gradiometer), so sensor_type "all"/"mag"/"grad" are EXACT on the real Yeom files
(306/102/204), and sfreq is taken from `info`. Files lacking an info dict (e.g.
synthetic mocks) fall back to a positional guess (mags at 2::3); pass an explicit
`mag_idx` to force the indices. (Note: classical CSP/band-power decode reach
*direction* poorly -- it is not a beta-ERD effect -- so the conv->transformer is
the decoder that works here.)
"""
import os
import glob
import numpy as np
import scipy.io
from scipy.signal import resample_poly, butter, sosfiltfilt
from fractions import Fraction

ALL_DIRS = ["dir0", "dir1", "dir2", "dir3"]   # 4 reach directions (semantics TBD)


def _bandfilter(x, sfreq, band, order=4):
    """Zero-phase Butterworth filter along the TIME axis (axis=1) of an
    (channels, time, trials) array, applied to the FULL epoch (before windowing,
    so a low-frequency filter has no short-window edge artifacts).
      band = h (scalar)   -> low-pass < h Hz
      band = (l, h)       -> band-pass l..h Hz (l=None -> low-pass < h)
    Returns the same dtype. Motivated by the pre-movement literature: the
    directional signal lives in slow (<~7 Hz, esp. 0.1-3 Hz SCP) time-domain
    potentials; band power gives ~chance (Waldert 2008)."""
    if band is None:
        return x
    if np.isscalar(band):
        sos = butter(order, float(band), btype="low", fs=sfreq, output="sos")
    else:
        lo, hi = band
        if lo is None:
            sos = butter(order, float(hi), btype="low", fs=sfreq, output="sos")
        else:
            sos = butter(order, [float(lo), float(hi)], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=1).astype(x.dtype, copy=False)


def _load_mat(path):
    """scipy.io.loadmat, falling back to mat73 for v7.3/HDF5 .mat files."""
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as e:                 # v7.3 -> HDF5
        try:
            import mat73
        except ImportError:
            raise NotImplementedError(
                f"{path} looks like a MATLAB v7.3 (HDF5) file. "
                f"Install the optional reader:  pip install mat73") from e
        return mat73.loadmat(path)


def _find_direction_cells(mat):
    """Locate the variable holding the 4 direction cells; return (key, list-of-4)."""
    for key, val in mat.items():
        if key.startswith("__"):
            continue
        # numpy object array (scipy) or python list (mat73) of length 4
        cells = None
        if isinstance(val, np.ndarray) and val.dtype == object:
            flat = val.ravel()
            if flat.size == 4:
                cells = list(flat)
        elif isinstance(val, (list, tuple)) and len(val) == 4:
            cells = list(val)
        if cells is not None and all(np.asarray(c).ndim == 3 for c in cells):
            return key, [np.asarray(c) for c in cells]
    raise RuntimeError(
        "Could not find a 4-cell (channels x time x trials) variable in the .mat. "
        f"Top-level keys: {[k for k in mat if not k.startswith('__')]}. "
        "Inspect the file and adjust _find_direction_cells if the layout differs.")


def _sensor_indices(sensor_type, n_meg, mag_idx):
    if sensor_type == "all":
        return np.arange(n_meg)
    mags = np.asarray(mag_idx) if mag_idx is not None else np.arange(2, n_meg, 3)
    if sensor_type == "mag":
        print(f"[yeom] WARNING: sensor_type='mag' uses assumed magnetometer indices "
              f"{mags[:3].tolist()}...; verify against channel names or pass mag_idx.")
        return mags
    if sensor_type == "grad":
        return np.array([i for i in range(n_meg) if i not in set(mags.tolist())])
    raise ValueError(f"sensor_type must be all/mag/grad, got {sensor_type!r}")


def _info_dict(mat):
    """The nested MNE info dict, if present (mat73 dict or scipy mat_struct)."""
    info = mat.get("info") if isinstance(mat, dict) else None
    if info is None:
        return None
    if isinstance(info, dict):
        return info.get("info", info)      # mat73: real info is nested under 'info'
    return getattr(info, "info", info)     # scipy mat_struct


def _meg_channel_index(mat):
    """Classify MEG channels from info['ch_names'] using the MEGIN naming convention
    (last digit 1 = magnetometer, 2/3 = planar gradiometer). Returns
    {'all','mag','grad': index arrays into the channel axis} or None if unavailable."""
    info = _info_dict(mat)
    ch_names = info.get("ch_names") if isinstance(info, dict) else getattr(info, "ch_names", None)
    if ch_names is None or len(ch_names) == 0:
        return None
    meg, mag, grad = [], [], []
    for i, name in enumerate(ch_names):
        c = str(name).replace(" ", "")
        if c.upper().startswith("MEG") and c[-1:] in ("1", "2", "3"):
            meg.append(i)
            (mag if c[-1] == "1" else grad).append(i)
    if not meg:
        return None
    return {"all": np.array(meg), "mag": np.array(mag), "grad": np.array(grad)}


def _info_sfreq(mat):
    info = _info_dict(mat)
    if info is None:
        return None
    v = info.get("sfreq") if isinstance(info, dict) else getattr(info, "sfreq", None)
    try:
        return float(np.ravel(v)[0]) if v is not None else None
    except Exception:
        return None


def _accel_indices(mat):
    """Indices of the 3 accelerometer channels (by name), or None."""
    info = _info_dict(mat)
    ch_names = info.get("ch_names") if isinstance(info, dict) else getattr(info, "ch_names", None)
    if not ch_names:
        return None
    idx = [i for i, n in enumerate(ch_names) if "accel" in str(n).lower()]
    return np.array(idx) if idx else None


def _eog_indices(mat):
    """Indices of the EOG (eye) channels (by name), or None. Used as a gaze/ocular
    confound control: decoding direction from EOG alone reveals whether a 'neural'
    pre-movement direction signal is actually eye movement toward the target."""
    info = _info_dict(mat)
    ch_names = info.get("ch_names") if isinstance(info, dict) else getattr(info, "ch_names", None)
    if not ch_names:
        return None
    idx = [i for i, n in enumerate(ch_names) if "eog" in str(n).lower()]
    return np.array(idx) if idx else None


def _detect_movement_onset(accel, sfreq, tmin, k=4.0, min_dur=0.03, search_start=0.0):
    """Per-trial movement onset (sample index) from a (n_axes, T) accelerometer trace.
    Detrend each axis by the pre-cue baseline (removes the gravity offset), take the
    Euclidean magnitude, and return the first sustained crossing of
    baseline_mean + k*baseline_std at/after `search_start` s. None if no crossing."""
    T = accel.shape[1]
    t = tmin + np.arange(T) / sfreq
    base = t < 0.0
    if base.sum() < 5:
        base = t < (t.min() + 0.2)
    detr = accel - accel[:, base].mean(axis=1, keepdims=True)
    mag = np.sqrt((detr ** 2).sum(axis=0))
    thr = mag[base].mean() + k * (mag[base].std() + 1e-12)
    cand = np.where((mag > thr) & (t >= search_start))[0]
    if len(cand) == 0:
        return None
    md = max(int(min_dur * sfreq), 1)
    for i in cand:
        if i + md <= T and (mag[i:i + md] > thr).all():
            return int(i)
    return int(cand[0])


def load_yeom(data_path, subject=None, session=None, sensor_type="all",
              classes=None, tmin=-1.0, tmax=2.0, crop=None, resample=None,
              n_meg=306, mag_idx=None, sfreq=600.615, seed=0,
              align="cue", onset_k=4.0, band=None):
    # --- resolve the .mat file -------------------------------------------------
    if os.path.isdir(data_path):
        # recursive so zip-extracted nested folders (e.g. on Colab/Drive) are found
        cands = sorted(glob.glob(os.path.join(data_path, "**", "*.mat"), recursive=True))
        for tok in (subject, session):
            if tok is not None:
                cands = [f for f in cands if str(tok) in os.path.basename(f)]
        if not cands:
            allmat = glob.glob(os.path.join(data_path, "**", "*.mat"), recursive=True)
            raise FileNotFoundError(
                f"No matching .mat under {data_path} for subject={subject} "
                f"session={session}. Available: {[os.path.basename(f) for f in allmat]}")
        path = cands[0]
    else:
        path = data_path
    print(f"[yeom] loading {os.path.basename(path)}")

    mat = _load_mat(path)
    key, cells = _find_direction_cells(mat)
    print(f"[yeom] using variable '{key}' with 4 direction cells")

    # --- which directions to keep ---------------------------------------------
    if classes is None:
        keep = list(range(4)); class_names = list(ALL_DIRS)
    else:
        keep = [ALL_DIRS.index(c) for c in classes]; class_names = list(classes)

    # consistency check across cells (channels, time)
    shapes = [c.shape for c in cells]
    if len({s[0] for s in shapes}) != 1 or len({s[1] for s in shapes}) != 1:
        raise RuntimeError(f"direction cells disagree on channels/time: {shapes}")

    # channel selection: prefer the MNE info ch_names (authoritative), else positional
    if sensor_type == "eog":                         # gaze/ocular confound control
        eog = _eog_indices(mat)
        if eog is None:
            raise RuntimeError("sensor_type='eog' but no EOG channels in info ch_names")
        sel, sel_src = eog, "eog"
    else:
        chan = None if mag_idx is not None else _meg_channel_index(mat)
        if chan is not None:
            if sensor_type not in chan:
                raise ValueError(f"sensor_type must be all/mag/grad/eog, got {sensor_type!r}")
            sel, sel_src = chan[sensor_type], "info"
        else:
            sel, sel_src = _sensor_indices(sensor_type, n_meg, mag_idx), "positional"

    # sampling rate: prefer the value stored in the file's info
    isf = _info_sfreq(mat)
    if isf is not None:
        sfreq = isf

    if align == "movement":
        # --- accelerometer-gated: per-trial window relative to MOVEMENT onset ---
        acc_idx = _accel_indices(mat)
        if acc_idx is None:
            raise RuntimeError("align='movement' needs accelerometer channels, "
                               "none found in info ch_names")
        win = crop if crop is not None else (-0.5, 0.0)   # seconds, relative to onset
        w0, w1 = int(round(win[0] * sfreq)), int(round(win[1] * sfreq))
        X_list, y_list, onset_s, n_drop = [], [], [], 0
        for new_label, d in enumerate(keep):
            cell = cells[d]                                # (319, T, n_trials)
            # filter MEG on the FULL epoch before per-trial windowing; accel stays raw
            meg, acc = _bandfilter(cell[sel], sfreq, band), cell[acc_idx]
            for tr in range(cell.shape[2]):
                onset = _detect_movement_onset(acc[:, :, tr], sfreq, tmin, k=onset_k)
                if onset is None or onset + w0 < 0 or onset + w1 > cell.shape[1]:
                    n_drop += 1
                    continue
                X_list.append(meg[:, onset + w0:onset + w1, tr])   # (n_meg, win)
                y_list.append(new_label)
                onset_s.append(tmin + onset / sfreq)
        if not X_list:
            raise RuntimeError("no trials survived accel-gating; relax onset_k or the window")
        X = np.stack(X_list).astype(np.float32)            # (n_kept, n_meg, win)
        y = np.array(y_list, dtype=int)
        onset_s = np.array(onset_s)
        print(f"[yeom] accel-gated movement-onset window: kept {len(y)}/{len(y)+n_drop} trials | "
              f"onset median {1000*np.median(onset_s):.0f} ms post-cue "
              f"(IQR {1000*np.percentile(onset_s,25):.0f}-{1000*np.percentile(onset_s,75):.0f}) | "
              f"window {win[0]:+.2f}..{win[1]:+.2f}s rel. onset")
    else:
        # --- cue-locked: single fixed window for every trial --------------------
        X_list, y_list = [], []
        for new_label, d in enumerate(keep):
            arr = _bandfilter(cells[d][sel], sfreq, band)   # MEG, filtered on full epoch
            trials = np.transpose(arr, (2, 0, 1))        # -> (trials, channels, time)
            X_list.append(trials)
            y_list.append(np.full(trials.shape[0], new_label, dtype=int))
        X = np.concatenate(X_list, axis=0).astype(np.float32)
        y = np.concatenate(y_list, axis=0)
        if crop is not None:                             # seconds, relative to tmin (cue)
            lo = int(round((crop[0] - tmin) * sfreq))
            hi = int(round((crop[1] - tmin) * sfreq))
            lo, hi = max(lo, 0), min(hi, X.shape[2])
            X = X[:, :, lo:hi]

    # --- optional anti-aliased downsample -------------------------------------
    if resample is not None and abs(resample - sfreq) > 1e-6:
        frac = Fraction(resample / sfreq).limit_denominator(1000)
        X = resample_poly(X, frac.numerator, frac.denominator, axis=2).astype(np.float32)
        sfreq = float(resample)

    # shuffle so folds aren't direction-blocked (match synthetic loader)
    perm = np.random.default_rng(seed).permutation(len(y))
    X, y = X[perm], y[perm]

    print(f"[yeom] {X.shape[0]} trials | {X.shape[1]} channels ({sensor_type}, {sel_src}) | "
          f"{X.shape[2]} samples | sfreq={sfreq:.1f} Hz | band={band} | classes {class_names}")
    return X, y, float(sfreq), class_names


def load_full_epochs(data_path, subject=None, session=None, sensor_type="all",
                     classes=None, tmin=-1.0, n_meg=306, mag_idx=None,
                     sfreq=600.615, onset_k=4.0, seed=0):
    """Causal-replay front-end for the pseudo-online harness (pseudo_online.py).

    Returns the FULL broadband epoch for every trial -- NO band-filtering,
    windowing, or resampling -- plus the per-trial accelerometer movement-onset
    sample index and the cue sample index, so the harness can apply all three of
    those operations CAUSALLY itself. (load_yeom does them ACAUSALLY: sosfiltfilt
    over the full epoch + resample_poly + onset-relative windowing.)

    Contract:
      X (n_trials, n_channels, n_times) float32, y (n_trials,) int, sfreq (float),
      onsets (n_trials,) int  -- accel movement-onset sample index into the time
                                 axis, or -1 if no onset was detected,
      cue_idx (int)           -- the cue (t=0) sample index = round(-tmin*sfreq),
      class_names (list).
    """
    # --- resolve the .mat (same matching as load_yeom) ---
    if os.path.isdir(data_path):
        cands = sorted(glob.glob(os.path.join(data_path, "**", "*.mat"), recursive=True))
        for tok in (subject, session):
            if tok is not None:
                cands = [f for f in cands if str(tok) in os.path.basename(f)]
        if not cands:
            raise FileNotFoundError(f"No matching .mat under {data_path} for "
                                    f"subject={subject} session={session}")
        path = cands[0]
    else:
        path = data_path
    print(f"[yeom] (full-epoch) loading {os.path.basename(path)}")

    mat = _load_mat(path)
    key, cells = _find_direction_cells(mat)

    if classes is None:
        keep = list(range(4)); class_names = list(ALL_DIRS)
    else:
        keep = [ALL_DIRS.index(c) for c in classes]; class_names = list(classes)

    # channel selection: prefer the MNE info ch_names (authoritative), else positional
    chan = None if mag_idx is not None else _meg_channel_index(mat)
    if chan is not None:
        if sensor_type not in chan:
            raise ValueError(f"sensor_type must be all/mag/grad, got {sensor_type!r}")
        sel = chan[sensor_type]
    else:
        sel = _sensor_indices(sensor_type, n_meg, mag_idx)

    isf = _info_sfreq(mat)
    if isf is not None:
        sfreq = isf
    cue_idx = int(round(-tmin * sfreq))

    acc_idx = _accel_indices(mat)
    if acc_idx is None:
        print("[yeom] WARNING: no accelerometer channels found; onsets set to -1 "
              "(onset-anchored conditions will then have no trials).")

    X_list, y_list, onsets = [], [], []
    for new_label, d in enumerate(keep):
        cell = np.asarray(cells[d])                  # (n_all_ch, time, trials)
        meg = cell[sel]                              # broadband, unfiltered
        acc = cell[acc_idx] if acc_idx is not None else None
        for tr in range(cell.shape[2]):
            X_list.append(meg[:, :, tr])
            y_list.append(new_label)
            if acc is not None:
                on = _detect_movement_onset(acc[:, :, tr], sfreq, tmin, k=onset_k)
                onsets.append(int(on) if on is not None else -1)
            else:
                onsets.append(-1)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=int)
    onsets = np.array(onsets, dtype=int)
    perm = np.random.default_rng(seed).permutation(len(y))
    X, y, onsets = X[perm], y[perm], onsets[perm]

    n_det = int((onsets >= 0).sum())
    print(f"[yeom] full epochs: {X.shape[0]} trials | {X.shape[1]} ch ({sensor_type}) | "
          f"{X.shape[2]} samp | sfreq={sfreq:.1f} Hz | cue@{cue_idx} | "
          f"onset detected {n_det}/{len(onsets)} | classes {class_names}")
    return X, y, float(sfreq), onsets, cue_idx, class_names
