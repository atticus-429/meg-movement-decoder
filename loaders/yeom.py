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
from scipy.signal import resample_poly
from fractions import Fraction

ALL_DIRS = ["dir0", "dir1", "dir2", "dir3"]   # 4 reach directions (semantics TBD)


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


def load_yeom(data_path, subject=None, session=None, sensor_type="all",
              classes=None, tmin=-1.0, tmax=2.0, crop=None, resample=None,
              n_meg=306, mag_idx=None, sfreq=600.615, seed=0):
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
    chan = None if mag_idx is not None else _meg_channel_index(mat)
    if chan is not None:
        if sensor_type not in chan:
            raise ValueError(f"sensor_type must be all/mag/grad, got {sensor_type!r}")
        sel, sel_src = chan[sensor_type], "info"
    else:
        sel, sel_src = _sensor_indices(sensor_type, n_meg, mag_idx), "positional"

    # sampling rate: prefer the value stored in the file's info
    isf = _info_sfreq(mat)
    if isf is not None:
        sfreq = isf

    X_list, y_list = [], []
    for new_label, d in enumerate(keep):
        arr = cells[d][sel]                          # select MEG channels (absolute idx)
        trials = np.transpose(arr, (2, 0, 1))        # -> (trials, channels, time)
        X_list.append(trials)
        y_list.append(np.full(trials.shape[0], new_label, dtype=int))
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0)

    # --- optional crop (seconds, relative to tmin) ----------------------------
    if crop is not None:
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
          f"{X.shape[2]} samples | sfreq={sfreq:.1f} Hz | classes {class_names}")
    return X, y, float(sfreq), class_names
