"""Tier-2 cross-subject transfer: pretrain on N-1 subjects, calibrate on the
target subject's small data, test on held-out target data.

Logic: the MEG reach-direction response = a SHARED part (same MEGIN 306 array,
similar motor somatotopy) + a SUBJECT-SPECIFIC part (head position, SNR, tuning).
Pretraining learns the shared structure from ~2000 trials; calibration fixes the
subject-specific part from the target's ~K trials. Per calibration budget K we
report:
  - zeroshot (K=0)     : pretrained model, no calibration
  - full     (flavor A): fine-tune ALL weights on K target trials
  - frozen   (flavor C): freeze trunk, adapt input adapter + head on K trials
  - scratch            : a fresh model trained on the SAME K trials -- the
                         baseline that decides whether pretraining ACTUALLY
                         transfers (full/frozen >> scratch) or not (~equal).

Protocol = cross-session: calibrate on the target's session 1, test on session 2.
The held-out subject is absent from pretraining; test is disjoint from calibration.

Importable `pretrain_then_calibrate`; run as a script for a synthetic
multi-subject smoke test (no real data needed):
    python hcp_motor_decoder/transfer.py
"""
import os
import sys
import copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import bootstrap_error_ci


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _acc_ci(y_true, y_pred, seed=0):
    err, lo, hi = bootstrap_error_ci(y_true, y_pred, seed=seed)
    return 100.0 * (1.0 - err), 100.0 * (1.0 - hi), 100.0 * (1.0 - lo)


def _stratified_sample(y, k_per_class, seed=0):
    """Indices for up to k_per_class trials of EACH class (stratified)."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        idx.extend(ci[:k_per_class].tolist())
    return np.array(sorted(idx))


def _resolve_device(model):
    import torch
    dev = model.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return dev


def _save_pretrained(model, path, n_channels):
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": model.net_.state_dict(),
                "mean": model.mean_, "std": model.std_,
                "classes": model.classes_,
                "nch": int(n_channels), "ncl": int(len(model.classes_))}, path)


def _load_pretrained(model, path):
    """Reconstruct a fitted model from a checkpoint (net_ is lazily built)."""
    import torch
    blob = torch.load(path, map_location="cpu")
    model.classes_ = blob["classes"]
    model.mean_, model.std_ = blob["mean"], blob["std"]
    model.device_ = _resolve_device(model)
    model.net_ = model._build(blob["nch"], blob["ncl"]).to(model.device_)
    model.net_.load_state_dict(blob["state"])
    return model


# --------------------------------------------------------------------------- #
# the experiment
# --------------------------------------------------------------------------- #
def pretrain_then_calibrate(subjects, build_fn, K_per_class_list=(0, 5, 10, 20, 30),
                            seed=0, finetune_lr=1e-4, finetune_epochs=30,
                            save_dir=None, verbose=True):
    """subjects: dict name -> {'ses1': (X, y), 'ses2': (X, y)} (same class set).
    build_fn: () -> a fresh TorchConvTransformer (carries hyperparams).
    Returns a list of result-row dicts (held, K, n_cal, variant, acc, ci_lo, ci_hi).
    """
    names = sorted(subjects)
    rows = []
    for held in names:
        others = [s for s in names if s != held]
        Xtr = np.concatenate([subjects[s][ses][0] for s in others
                              for ses in subjects[s]])
        ytr = np.concatenate([subjects[s][ses][1] for s in others
                              for ses in subjects[s]])

        # ---- pretrain on the other subjects (cache for resume) ----
        ckpt = os.path.join(save_dir, f"pretrain_wo_{held}.pt") if save_dir else None
        model = build_fn()
        if ckpt and os.path.exists(ckpt):
            _load_pretrained(model, ckpt)
            if verbose:
                print(f"[{held}] loaded pretrained checkpoint", flush=True)
        else:
            if verbose:
                print(f"[{held}] pretraining on {len(others)} subjects "
                      f"({len(ytr)} trials)...", flush=True)
            model.fit(Xtr, ytr)
            if ckpt:
                _save_pretrained(model, ckpt, Xtr.shape[1])
        pre_state = copy.deepcopy(model.net_.state_dict())

        Xcal, ycal = subjects[held]["ses1"]      # calibration source
        Xte, yte = subjects[held]["ses2"]        # held-out test (never calibrated on)

        # ---- K=0 : zero-shot ----
        a, l, h = _acc_ci(yte, model.predict(Xte), seed=seed)
        rows.append(dict(held=held, K=0, n_cal=0, variant="zeroshot",
                         acc=a, ci_lo=l, ci_hi=h))
        if verbose:
            print(f"[{held}] K=0  zeroshot {a:.1f}%", flush=True)

        # ---- K>0 : full (A) / frozen (C) / scratch ----
        for K in [k for k in K_per_class_list if k > 0]:
            cal_idx = _stratified_sample(ycal, K, seed=seed)
            Xc, yc = Xcal[cal_idx], ycal[cal_idx]

            model.net_.load_state_dict(pre_state)
            model.finetune(Xc, yc, lr=finetune_lr, n_epochs=finetune_epochs,
                           freeze_trunk=False)
            accA = _acc_ci(yte, model.predict(Xte), seed=seed)

            model.net_.load_state_dict(pre_state)
            model.finetune(Xc, yc, lr=finetune_lr, n_epochs=finetune_epochs,
                           freeze_trunk=True)
            accC = _acc_ci(yte, model.predict(Xte), seed=seed)

            scratch = build_fn()
            scratch.fit(Xc, yc)
            accS = _acc_ci(yte, scratch.predict(Xte), seed=seed)

            for variant, (a, l, h) in [("full", accA), ("frozen", accC),
                                       ("scratch", accS)]:
                rows.append(dict(held=held, K=K, n_cal=int(len(yc)),
                                 variant=variant, acc=a, ci_lo=l, ci_hi=h))
            if verbose:
                print(f"[{held}] K={K} ({len(yc)} cal):  full {accA[0]:.1f}  "
                      f"frozen {accC[0]:.1f}  scratch {accS[0]:.1f}", flush=True)
    return rows


def summarize(rows):
    """Mean accuracy per (K, variant) across held-out subjects -> dict."""
    out = {}
    Ks = sorted({r["K"] for r in rows})
    variants = ["zeroshot", "full", "frozen", "scratch"]
    for K in Ks:
        for v in variants:
            accs = [r["acc"] for r in rows if r["K"] == K and r["variant"] == v]
            if accs:
                out[(K, v)] = (float(np.mean(accs)), float(np.std(accs)), len(accs))
    return out


# --------------------------------------------------------------------------- #
# synthetic multi-subject smoke test (no real data)
# --------------------------------------------------------------------------- #
def _make_synthetic_subjects(n_subjects=4, n_channels=32, sfreq=125.0, seed=0):
    """Per-subject channel MIX + gain (consistent across the subject's 2 sessions,
    different across subjects) mimics head-position / SNR differences -> zero-shot
    transfer is imperfect, calibration should recover."""
    from loaders.synthetic import make_synthetic_motor
    rng = np.random.default_rng(seed)
    subjects = {}
    sf = sfreq
    for s in range(n_subjects):
        M = np.eye(n_channels) + 0.3 * rng.standard_normal((n_channels, n_channels)) / np.sqrt(n_channels)
        gain = np.exp(0.2 * rng.standard_normal((n_channels, 1)))
        d = {}
        for ses, sd in (("ses1", 1000 * s + 1), ("ses2", 1000 * s + 2)):
            X, y, sf, _ = make_synthetic_motor(
                n_trials_per_class=20, n_channels=n_channels, sfreq=sfreq,
                tmin=-0.2, tmax=0.6, classes=("LH", "RH", "LF", "RF"), seed=sd)
            Xmix = (np.einsum("ij,njt->nit", M, X) * gain[None]).astype(np.float32)
            d[ses] = (Xmix, y)
        subjects[f"S{s}"] = d
    return subjects, sf


def _smoke():
    from decode import TorchConvTransformer
    subjects, sf = _make_synthetic_subjects(n_subjects=4, n_channels=32, seed=0)
    shapes = {s: {k: v[0].shape for k, v in d.items()} for s, d in subjects.items()}
    print("subjects:", list(subjects), "| example shapes:", shapes["S0"])

    def build_fn():
        return TorchConvTransformer(
            sfreq=sf, n_spatial=32, conv_dim=32, d_model=32, n_heads=2,
            n_layers=1, n_conv_blocks=2, n_epochs=15, batch_size=16, seed=0)

    rows = pretrain_then_calibrate(subjects, build_fn, K_per_class_list=(0, 10, 20),
                                   finetune_lr=1e-3, finetune_epochs=15, seed=0)
    summ = summarize(rows)
    print("\n--- mean over subjects (acc%) ---")
    print(f"{'K':>3}  {'zeroshot':>9} {'full':>7} {'frozen':>7} {'scratch':>8}")
    for K in sorted({k for k, _ in summ}):
        def g(v):
            return f"{summ[(K, v)][0]:.1f}" if (K, v) in summ else "  -"
        print(f"{K:>3}  {g('zeroshot'):>9} {g('full'):>7} {g('frozen'):>7} {g('scratch'):>8}")

    # plumbing assertions (the point of the smoke test)
    variants_present = {v for (_, v) in summ}
    assert {"zeroshot", "full", "frozen", "scratch"} <= variants_present, variants_present
    Kmax = max(k for k, _ in summ)
    zs = summ[(0, "zeroshot")][0]
    best_cal = max(summ[(Kmax, "full")][0], summ[(Kmax, "frozen")][0])
    print(f"\nzeroshot {zs:.1f}%  ->  best calibrated @K={Kmax} {best_cal:.1f}%")
    # science sanity (lenient): calibration should not be WORSE than zero-shot
    assert best_cal >= zs - 5.0, f"calibration ({best_cal:.1f}) << zeroshot ({zs:.1f})"
    print("SMOKE OK")


if __name__ == "__main__":
    _smoke()
