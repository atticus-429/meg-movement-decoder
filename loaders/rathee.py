"""Loader for the Rathee et al. 2021 MEG motor/cognitive imagery BIDS dataset.

  Rathee, Raza, Roy, Prasad (2021), Scientific Data 8:120.
  figshare collection DOI 10.6084/m9.figshare.c.5101544 (MEG_BIDS.zip, ~52 GB).

This is the IMAGERY test (vs Yeom's overt execution): real motor/cognitive
imagery, the regime closest to the project's attempted/imagined-movement goal.

Task `bcimici`, one raw .fif per subject/session, 4 classes by STIM-channel code:
  hand=4, feet=8, wordgen=16, subtraction=32  (50 trials/class/session).
Trial: 2 s rest -> 5 s imagery -> ITI; the class trigger marks imagery onset, and
the paper's decoding window is 0.5-3.5 s post-onset (the default tmin/tmax here).
306 MEG channels (102 mag + 204 grad), 1000 Hz.

Reference accuracies (the SOTA bar): intra-session within-subject binary ~96-99%
(filter-bank CSP); inter-session ~56-69% (covariate shift).

Returns the standard contract: X (n_trials, n_channels, n_times), y, sfreq, class_names.
For the pure motor-imagery 2-class contrast use classes=('hand','feet').

NOTE: untested on the real 52 GB data here -- on first run check the printed event
codes/counts (expect 4/8/16/32 with ~50 each) and adjust if the STIM layout differs.
"""
import os
import glob
import numpy as np

TRIG = {"hand": 4, "feet": 8, "wordgen": 16, "subtraction": 32}   # STIM codes (paper)
ALL_CLASSES = ("hand", "feet", "wordgen", "subtraction")


def _read_raw(bids_root, subject, session):
    """Read the subject/session raw .fif (mne-bids if available, else glob)."""
    import mne
    try:
        from mne_bids import BIDSPath, read_raw_bids
        bp = BIDSPath(subject=str(subject), session=str(session), task="bcimici",
                      datatype="meg", root=bids_root)
        return read_raw_bids(bp, verbose="ERROR")
    except Exception as e:                                   # noqa: BLE001
        pat = os.path.join(bids_root, f"sub-{subject}", f"ses-{session}", "meg",
                           f"sub-{subject}_ses-{session}_task-bcimici*meg.fif")
        cands = [c for c in sorted(glob.glob(pat))
                 if not any(s in c for s in ("split-02", "split-03", "split-04"))]
        if not cands:
            raise FileNotFoundError(f"no Rathee .fif at {pat} ({e})")
        return mne.io.read_raw_fif(cands[0], verbose="ERROR")


def load_rathee(bids_root, subject, session="1", sensor_type="all", classes=None,
                tmin=0.5, tmax=3.5, resample=None, stim_channel="STI101", seed=0):
    import mne
    mne.set_log_level("ERROR")
    classes = list(classes) if classes else list(ALL_CLASSES)

    raw = _read_raw(bids_root, subject, session)
    raw.load_data()

    # events from the STIM channel (documented codes 4/8/16/32)
    sc = stim_channel if stim_channel in raw.ch_names else None
    events = mne.find_events(raw, stim_channel=sc, shortest_event=1,
                             consecutive=True, verbose="ERROR")
    codes, counts = np.unique(events[:, 2], return_counts=True)
    print(f"[rathee] STIM events: {dict(zip(codes.tolist(), counts.tolist()))}")

    event_id = {c: TRIG[c] for c in classes}
    ev = events[np.isin(events[:, 2], list(event_id.values()))]
    if len(ev) == 0:
        raise RuntimeError(f"no events match class codes {list(event_id.values())}; "
                           f"found {dict(zip(codes.tolist(), counts.tolist()))} -- "
                           "check stim_channel / TRIG mapping")

    meg = {"all": True, "mag": "mag", "grad": "grad"}[sensor_type]
    picks = mne.pick_types(raw.info, meg=meg, eeg=False, stim=False, eog=False, ecg=False)

    epochs = mne.Epochs(raw, ev, event_id=event_id, tmin=tmin, tmax=tmax, picks=picks,
                        baseline=None, preload=True, on_missing="warn", verbose="ERROR")
    if resample:
        epochs.resample(resample, verbose="ERROR")

    X = epochs.get_data(copy=True).astype(np.float32)        # (trials, ch, time)
    code_to_name = {TRIG[c]: c for c in classes}
    name_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([name_to_idx[code_to_name[c]] for c in epochs.events[:, 2]], dtype=int)

    perm = np.random.default_rng(seed).permutation(len(y))
    X, y = X[perm], y[perm]
    print(f"[rathee] sub-{subject} ses-{session}: {X.shape[0]} trials | {X.shape[1]} ch "
          f"({sensor_type}) | {X.shape[2]} samp | sfreq {epochs.info['sfreq']:.0f} Hz | "
          f"classes {classes} | counts {np.bincount(y).tolist()}")
    return X, y, float(epochs.info["sfreq"]), classes
