"""Real HCP MEG Motor loader (via the `mne-hcp` package).

  pip install mne-hcp      # provides the importable module `hcp`

This is the REAL-DATA branch. It is not exercised by the synthetic smoke test,
because it needs (a) mne-hcp installed and (b) a downloaded HCP subject on disk
(see ../download_hcp.md). HCP field conventions vary slightly across releases,
so on first run inspect the printed trigger-code table and, if needed, adjust
DEFAULT_CODE_MAP below to match what you see.

Returns the standard contract: X (n_trials, n_channels, n_times), y, sfreq,
class_names -- identical to loaders/synthetic.make_synthetic_motor.
"""
import numpy as np

# HCP Motor task has two runs.
MOTOR_RUNS = (0, 1)

# Best-effort map from HCP Motor movement trigger code -> our class label.
# VERIFY against the printed code table on first run and edit if necessary.
DEFAULT_CODE_MAP = {
    1: "LH",   # left hand
    2: "LF",   # left foot
    4: "RH",   # right hand
    5: "RF",   # right foot
}


def _read_run_epochs(subject, hcp_path, run_index, tmin, tmax, code_map):
    import hcp
    import mne

    raw = hcp.read_raw(subject=subject, hcp_path=hcp_path,
                       data_type="task_motor", run_index=run_index)
    raw.load_data()
    # HCP MEG is magnetometer-based (4D/BTi); keep MEG channels only.
    raw.pick_types(meg=True, ref_meg=False)

    trial_info = hcp.read_trial_info(subject=subject, hcp_path=hcp_path,
                                     data_type="task_motor", run_index=run_index)
    stim = trial_info["stim"]
    comments = list(stim["comments"]) if "comments" in stim else []
    codes = np.asarray(stim["codes"])

    # Heuristic: the onset-sample column and the condition column. The last
    # column of HCP trial_info codes is the event onset (in samples); a column
    # whose values match our code_map keys is the condition.
    onset_col = codes.shape[1] - 1
    cond_col = None
    for c in range(codes.shape[1] - 1):
        present = set(np.unique(codes[:, c]).tolist())
        if present & set(code_map.keys()):
            cond_col = c
            break
    if cond_col is None:
        raise RuntimeError(
            "Could not locate the movement-condition column in HCP trial_info.\n"
            f"  trial_info['stim']['comments'] = {comments}\n"
            f"  unique codes per column = "
            f"{[np.unique(codes[:, c]).tolist() for c in range(codes.shape[1])]}\n"
            "Inspect these and set DEFAULT_CODE_MAP / cond_col accordingly.")

    onsets = codes[:, onset_col].astype(int)
    conds = codes[:, cond_col].astype(int)

    events = np.column_stack([onsets, np.zeros_like(onsets), conds])
    # keep only events whose code we know how to label
    keep = np.isin(conds, list(code_map.keys()))
    events = events[keep]
    event_id = {code_map[c]: int(c) for c in np.unique(events[:, 2])}

    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, on_missing="warn")
    return epochs


def load_hcp_motor(subject, hcp_path, classes=("LH", "RH"),
                   tmin=-0.5, tmax=1.5, code_map=None, resample=250.0):
    try:
        import hcp  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The real HCP loader needs the `mne-hcp` package:\n"
            "    pip install mne-hcp\n"
            "and a downloaded HCP subject (see download_hcp.md).") from e

    code_map = code_map or DEFAULT_CODE_MAP

    all_epochs = []
    for run in MOTOR_RUNS:
        try:
            all_epochs.append(_read_run_epochs(subject, hcp_path, run,
                                               tmin, tmax, code_map))
        except Exception as e:                       # noqa: BLE001
            print(f"[hcp] run {run} skipped: {e}")
    if not all_epochs:
        raise RuntimeError(f"No HCP Motor data loaded for subject {subject} "
                           f"under {hcp_path}. Check the download path/layout.")

    import mne
    epochs = mne.concatenate_epochs(all_epochs)
    if resample:
        epochs.resample(resample)

    # restrict to the requested classes and build a compact 0..k-1 label vector
    classes = [c for c in classes if c in epochs.event_id]
    if len(classes) < 2:
        raise RuntimeError(f"Fewer than 2 requested classes present. "
                           f"Available: {list(epochs.event_id)}")
    epochs = epochs[classes]

    X = epochs.get_data().astype(np.float32)         # (n_trials, n_channels, n_times)
    inv = {v: k for k, v in epochs.event_id.items()}
    labels = [inv[code] for code in epochs.events[:, 2]]
    name_to_idx = {name: i for i, name in enumerate(classes)}
    y = np.array([name_to_idx[name] for name in labels], dtype=int)

    print(f"[hcp] subject {subject}: {X.shape[0]} trials across "
          f"{len(classes)} classes {classes}")
    return X, y, float(epochs.info["sfreq"]), list(classes)
