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
def _calibrate_sweep(model, pre_state, pre_mean, pre_std, Xpool, ypool, Xte, yte,
                     build_fn, K_per_class_list, finetune_lr, finetune_epochs,
                     seed, held, unit, verbose):
    """Pretrained `model` cached in pre_state/pre_mean/pre_std. Sweep calibration
    budget K over (Xpool,ypool), test on the fixed (Xte,yte). Rows: zeroshot (K=0)
    + full/frozen/scratch per K>0. Restores the pristine pretrained state first so
    this is reusable across units of the same held subject."""
    rows = []
    # K=0 zero-shot: pristine pretrained model (weights + PRETRAIN standardization)
    model.net_.load_state_dict(pre_state)
    model.mean_, model.std_ = pre_mean, pre_std
    a, l, h = _acc_ci(yte, model.predict(Xte), seed=seed)
    rows.append(dict(held=held, unit=unit, K=0, n_cal=0, variant="zeroshot",
                     acc=a, ci_lo=l, ci_hi=h))
    if verbose:
        print(f"[{held}/{unit}] K=0  zeroshot {a:.1f}%", flush=True)
    for K in [k for k in K_per_class_list if k > 0]:
        idx = _stratified_sample(ypool, K, seed=seed)
        Xc, yc = Xpool[idx], ypool[idx]
        model.net_.load_state_dict(pre_state)
        model.finetune(Xc, yc, lr=finetune_lr, n_epochs=finetune_epochs, freeze_trunk=False)
        accA = _acc_ci(yte, model.predict(Xte), seed=seed)
        model.net_.load_state_dict(pre_state)
        model.finetune(Xc, yc, lr=finetune_lr, n_epochs=finetune_epochs, freeze_trunk=True)
        accC = _acc_ci(yte, model.predict(Xte), seed=seed)
        scratch = build_fn(); scratch.fit(Xc, yc)
        accS = _acc_ci(yte, scratch.predict(Xte), seed=seed)
        for variant, (a, l, h) in [("full", accA), ("frozen", accC), ("scratch", accS)]:
            rows.append(dict(held=held, unit=unit, K=K, n_cal=int(len(yc)),
                             variant=variant, acc=a, ci_lo=l, ci_hi=h))
        if verbose:
            print(f"[{held}/{unit}] K={K} ({len(yc)} cal):  full {accA[0]:.1f}  "
                  f"frozen {accC[0]:.1f}  scratch {accS[0]:.1f}", flush=True)
    return rows


def pretrain_then_calibrate(subjects, build_fn, K_per_class_list=(0, 5, 10, 20),
                            protocol="within_session", test_per_class=10,
                            seed=0, finetune_lr=1e-4, finetune_epochs=30,
                            save_dir=None, verbose=True):
    """subjects: dict name -> {'ses1': (X, y), 'ses2': (X, y)} (same class set).
    build_fn: () -> a fresh TorchConvTransformer.
    protocol:
      'within_session' (default; the realistic BCI case -- recalibrate each
        session): per target session, hold out test_per_class/class as a FIXED
        test set, calibrate on K/class from the rest, test within the SAME session.
      'cross_session' (stretch goal -- calibrate once, use another day):
        calibrate on the target's session 1, test on session 2.
    The held-out subject is always absent from pretraining; test is disjoint from
    calibration. Returns a list of result-row dicts.
    """
    names = sorted(subjects)
    rows = []
    for held in names:
        others = [s for s in names if s != held]
        Xtr = np.concatenate([subjects[s][ses][0] for s in others for ses in subjects[s]])
        ytr = np.concatenate([subjects[s][ses][1] for s in others for ses in subjects[s]])

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
        pre_mean, pre_std = model.mean_.copy(), model.std_.copy()

        if protocol == "cross_session":
            Xpool, ypool = subjects[held]["ses1"]
            Xte, yte = subjects[held]["ses2"]
            rows += _calibrate_sweep(model, pre_state, pre_mean, pre_std, Xpool, ypool,
                                     Xte, yte, build_fn, K_per_class_list, finetune_lr,
                                     finetune_epochs, seed, held, "xses", verbose)
        else:  # within_session: per session, fixed held-out test
            for ses in sorted(subjects[held]):
                X, y = subjects[held][ses]
                te_idx = _stratified_sample(y, test_per_class, seed=seed + 1)
                mask = np.ones(len(y), bool); mask[te_idx] = False
                Xte, yte = X[te_idx], y[te_idx]
                Xpool, ypool = X[mask], y[mask]
                rows += _calibrate_sweep(model, pre_state, pre_mean, pre_std, Xpool, ypool,
                                         Xte, yte, build_fn, K_per_class_list, finetune_lr,
                                         finetune_epochs, seed, held, ses, verbose)
    return rows


def cross_window_transfer(subjects, build_fn, split_sample=None,
                          K_per_class_list=(0, 5, 10, 20), test_per_class=10,
                          include_others_strong=False, seed=0,
                          finetune_lr=1e-4, finetune_epochs=30,
                          save_dir=None, verbose=True):
    """Cross-WINDOW transfer to lift the weak PRE window using the strong DURING
    window. subjects: name -> {ses: (X_full, y)} where X_full spans the COMBINED
    movement-onset window [-0.5,+0.5]s -- the early half is PRE (planning), the late
    half is DURING (execution); split at split_sample (default T//2). Both halves
    have the same shape so one model fits both.

    Per subject/session: split trials into CAL and TEST (stratified, TEST =
    test_per_class/class); PRETRAIN the feature-extractor on the DURING half of CAL
    trials (+ all OTHER subjects' DURING if include_others_strong); then calibrate on
    the PRE half of CAL trials (K/class) and test on the PRE half of TEST trials.
    Reuses _calibrate_sweep -> rows carry zeroshot/full/frozen/scratch per K, where
    scratch = PRE-only baseline and zeroshot = DURING-pretrained, no PRE calibration.
    No leakage: TEST trials' slices (either window) never enter pretrain or calibrate.
    """
    names = sorted(subjects)
    rows = []
    for held in names:
        others_X = others_y = None
        if include_others_strong:
            ox, oy = [], []
            for s in names:
                if s == held:
                    continue
                for ses in subjects[s]:
                    Xf, yy = subjects[s][ses]
                    sp = (Xf.shape[2] // 2) if split_sample is None else split_sample
                    ox.append(Xf[:, :, sp:]); oy.append(yy)            # DURING half
            others_X, others_y = np.concatenate(ox), np.concatenate(oy)

        for ses in sorted(subjects[held]):
            Xf, y = subjects[held][ses]
            sp = (Xf.shape[2] // 2) if split_sample is None else split_sample
            pre, during = Xf[:, :, :sp], Xf[:, :, sp:]
            te_idx = _stratified_sample(y, test_per_class, seed=seed + 1)
            mask = np.ones(len(y), bool); mask[te_idx] = False
            pre_te, y_te = pre[te_idx], y[te_idx]
            pre_pool, y_pool = pre[mask], y[mask]
            dur_pool = during[mask]                                    # CAL trials' DURING

            if include_others_strong:
                Xs = np.concatenate([others_X, dur_pool])
                ys = np.concatenate([others_y, y_pool])
            else:
                Xs, ys = dur_pool, y_pool

            ckpt = os.path.join(save_dir, f"xwin_{held}_{ses}.pt") if save_dir else None
            model = build_fn()
            if ckpt and os.path.exists(ckpt):
                _load_pretrained(model, ckpt)
                if verbose:
                    print(f"[{held}/{ses}] loaded DURING-pretrain checkpoint", flush=True)
            else:
                if verbose:
                    print(f"[{held}/{ses}] pretrain on DURING ({len(ys)} trials"
                          f"{', +others' if include_others_strong else ''})...", flush=True)
                model.fit(Xs, ys)
                if ckpt:
                    _save_pretrained(model, ckpt, Xs.shape[1])
            pre_state = copy.deepcopy(model.net_.state_dict())
            pre_mean, pre_std = model.mean_.copy(), model.std_.copy()

            rows += _calibrate_sweep(model, pre_state, pre_mean, pre_std, pre_pool, y_pool,
                                     pre_te, y_te, build_fn, K_per_class_list, finetune_lr,
                                     finetune_epochs, seed, held, ses, verbose)
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

    rows = pretrain_then_calibrate(subjects, build_fn, K_per_class_list=(0, 5, 10),
                                   protocol="within_session", test_per_class=5,
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


def _make_xwin_subjects(n_subjects=4, n_channels=32, sfreq=150.0, seed=0):
    """Combined-window synthetic subjects for the cross-window smoke: one
    [-0.5,+0.5]s window per (subject, session), sliced 50/50 into PRE | DURING."""
    from loaders.synthetic import make_synthetic_motor
    rng = np.random.default_rng(seed)
    subjects, sf = {}, sfreq
    for s in range(n_subjects):
        M = np.eye(n_channels) + 0.3 * rng.standard_normal((n_channels, n_channels)) / np.sqrt(n_channels)
        gain = np.exp(0.2 * rng.standard_normal((n_channels, 1)))
        d = {}
        for ses, sd in (("ses1", 10 * s + 1), ("ses2", 10 * s + 2)):
            X, y, sf, _ = make_synthetic_motor(
                n_trials_per_class=20, n_channels=n_channels, sfreq=sfreq,
                tmin=-0.5, tmax=0.5, classes=("LH", "RH", "LF", "RF"), seed=sd)
            X = (np.einsum("ij,njt->nit", M, X) * gain[None]).astype(np.float32)
            d[ses] = (X, y)
        subjects[f"S{s}"] = d
    return subjects, sf


def _smoke_xwin():
    from decode import TorchConvTransformer
    subjects, sf = _make_xwin_subjects(n_subjects=4, n_channels=32, seed=0)
    print("\n[cross-window smoke] subjects:", list(subjects),
          "| X_full:", subjects["S0"]["ses1"][0].shape)

    def build_fn():
        return TorchConvTransformer(
            sfreq=sf, n_spatial=32, conv_dim=32, d_model=32, n_heads=2,
            n_layers=1, n_conv_blocks=2, n_epochs=15, batch_size=16, seed=0)

    rows = cross_window_transfer(subjects, build_fn, K_per_class_list=(0, 5, 10),
                                 test_per_class=5, include_others_strong=False,
                                 finetune_lr=1e-3, finetune_epochs=15, seed=0)
    summ = summarize(rows)
    print("--- cross-window mean over subjects (acc%) ---")
    print(f"{'K':>3}  {'zeroshot':>9} {'full':>7} {'frozen':>7} {'scratch':>8}")
    for K in sorted({k for k, _ in summ}):
        def g(v):
            return f"{summ[(K, v)][0]:.1f}" if (K, v) in summ else "  -"
        print(f"{K:>3}  {g('zeroshot'):>9} {g('full'):>7} {g('frozen'):>7} {g('scratch'):>8}")
    variants = {v for (_, v) in summ}
    assert {"zeroshot", "full", "frozen", "scratch"} <= variants, variants
    print("XWIN SMOKE OK")


if __name__ == "__main__":
    _smoke()
    _smoke_xwin()
