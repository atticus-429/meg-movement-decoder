#!/usr/bin/env python
"""Pseudo-online / causal replay for the `tdlinear` pre-movement decoder.

Question this answers: how much accuracy does `tdlinear` LOSE when every
operation is forced CAUSAL (no future samples), vs. the offline acausal pipeline
that produced the §12 numbers? That delta is the gate for whether a true online
build is worth pursuing -- the linear classifier is trivially real-time; the
offline pipeline's *acausality* is the real obstacle.

The offline `load_yeom` cheats on the future in three ways; each gets a causal
counterpart here, and we vary them one at a time so the penalty is attributable:

  1. FILTER     `sosfiltfilt` (zero-phase, reads backward)  ->  `sosfilt` (causal)
  2. WINDOW     onset-anchored (accel onset is future info) ->  cue-anchored
                (cue is known in real time; jitter vs onset costs accuracy)
  3. RESAMPLE   `resample_poly` (acausal polyphase)         ->  causal decimation
                (causal anti-alias low-pass + integer subsample)
  +  NORMALISE  per-(channel x timepoint) z-score           ->  per-channel
                (a sliding online window has no stable per-timepoint template)

Conditions (each = one setting of those four toggles), per band x window cell:
  offline       acausal filter | onset | feature-scale | acausal resample  (= §12)
  c:filter      CAUSAL  filter | onset | feature-scale | acausal resample  (filter only)
  c:scale       acausal filter | onset | CHANNEL-scale | acausal resample  (scale only)
  c:resamp      acausal filter | onset | feature-scale | CAUSAL  resample  (resample only)
  causal-onset  CAUSAL  filter | onset | CHANNEL-scale | CAUSAL  resample  (all causal, perfect onset)
  causal-cue    CAUSAL  filter | CUE   | CHANNEL-scale | CAUSAL  resample  (all causal, realistic)

Read: offline -> causal-onset = total causal penalty assuming a perfect onset
detector; causal-onset -> causal-cue = the extra cost of not knowing onset.
The c:* rows attribute the offline->causal-onset gap to filter / scale / resample.

Scope: a MEASUREMENT harness. NOT a deployment -- no real onset detector, no
LSL/streaming, no hardware. Those follow only if the penalty is acceptable.

Run:
    python hcp_motor_decoder/pseudo_online.py                 # synthetic smoke + grid
    python hcp_motor_decoder/pseudo_online.py --source yeom \
        --yeom-path ./yeom_data --subject Sub_1
"""
import os
import sys
import argparse
from collections import Counter
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, resample_poly
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from decode import Flatten
from evaluate import bootstrap_error_ci

# Grid mirrors §12 (Low-frequency / late-window PRE test).
BANDS = [None, 3.0, 7.0, (0.1, 3.0)]                  # None=broadband; scalar=low-pass; tuple=band-pass
ONSET_WINDOWS = [(-0.5, 0.0), (-0.3, 0.0), (-0.2, 0.0)]   # rel. movement onset (w1=0 -> ends at onset)
# Cue-anchored analogs: same lengths, ending ~at the median onset (RT ~0.4 s post-cue).
CUE_WINDOWS = [(-0.1, 0.4), (0.1, 0.4), (0.2, 0.4)]       # rel. cue

CONDITIONS = [
    dict(name="offline",      filt="acausal", anchor="onset", scale="feature", resamp="acausal"),
    dict(name="c:filter",     filt="causal",  anchor="onset", scale="feature", resamp="acausal"),
    dict(name="c:scale",      filt="acausal", anchor="onset", scale="channel", resamp="acausal"),
    dict(name="c:resamp",     filt="acausal", anchor="onset", scale="feature", resamp="causal"),
    dict(name="causal-onset", filt="causal",  anchor="onset", scale="channel", resamp="causal"),
    dict(name="causal-cue",   filt="causal",  anchor="cue",   scale="channel", resamp="causal"),
]

# Lowered from sklearn's 5000 ceiling: weak-signal cells otherwise grind to the
# lbfgs iteration limit, which is the dominant runtime cost. C=0.1 (strong L2)
# converges well under this; pass --max-iter 5000 to restore the exact §12 estimator.
DEFAULT_MAX_ITER = 1000


# --------------------------------------------------------------------------- #
# per-channel scaler (online-faithful: no per-timepoint template) + estimators
# --------------------------------------------------------------------------- #
class ChannelScaler(BaseEstimator, TransformerMixin):
    """z-score each channel using TRAIN statistics over (trials, time). Unlike a
    flattened StandardScaler (which standardises each channel x timepoint, baking
    in the window's temporal template), this is what a sliding online window with
    a fixed calibration could actually reproduce."""
    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


def _plsda(k):
    """Supervised PLS-DA reducer (one-hot y -> PLSRegression -> X-scores) used as the
    manifold bottleneck. Imported lazily from the neural_manifold_learning branch so
    the default `direct` arm carries no dependency on that sibling tree; if the branch
    is absent the manifold arm raises a clear error rather than failing at import."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "neural_manifold_learning"))
        from reducers import PLSDATransformer
    except Exception as e:                                   # pragma: no cover
        raise ImportError(
            "manifold decoder needs neural_manifold_learning/reducers.py "
            f"(PLSDATransformer); not importable: {e}")
    return PLSDATransformer(n_components=k)


def _estimator(scale, sfreq, max_iter=DEFAULT_MAX_ITER, decoder="direct", k=16):
    """The pre-movement estimator with the requested normalisation and decoder arm.

    decoder='direct' replicates build_decoder('tdlinear') (a flat L2 logistic over all
    channel x time features). decoder='manifold' inserts a supervised PLS-DA bottleneck
    of dimension k between the scaled features and the SAME logistic head, so the only
    difference is the low-dim projection -- the manifold arm decodes X -> z (k dims) -> y
    while direct decodes X -> y over the full ~ch*t feature space. Both heads share
    C=0.1 so the comparison isolates the bottleneck, not the regulariser.

    'feature' scaling = per (channel x timepoint) z-score (Flatten -> StandardScaler,
    the §12 acausal template); 'channel' = per-channel z-score (online-faithful, no
    per-timepoint template). The PLS reducer centres internally, so the channel arm
    keeps its template-free property (no StandardScaler inserted). max_iter caps lbfgs;
    sfreq is accepted for signature symmetry."""
    clf = LogisticRegression(C=0.1, max_iter=max_iter)
    if scale == "feature":
        steps = [Flatten(), StandardScaler()]
    else:
        steps = [ChannelScaler(), Flatten()]
    if decoder == "manifold":
        steps.append(_plsda(k))
    steps.append(clf)
    return make_pipeline(*steps)


# --------------------------------------------------------------------------- #
# signal ops: causal vs acausal filtering, windowing, decimation
# --------------------------------------------------------------------------- #
def _make_sos(sfreq, band, order=4):
    if np.isscalar(band):
        return butter(order, float(band), btype="low", fs=sfreq, output="sos")
    lo, hi = band
    if lo is None:
        return butter(order, float(hi), btype="low", fs=sfreq, output="sos")
    return butter(order, [float(lo), float(hi)], btype="band", fs=sfreq, output="sos")


def _filter_full(X, sfreq, band, causal):
    """Band-filter the FULL epoch along time (axis=2). causal -> sosfilt (forward
    only); acausal -> sosfiltfilt (zero-phase). band=None -> passthrough."""
    if band is None:
        return X.astype(np.float32)
    sos = _make_sos(sfreq, band)
    f = sosfilt if causal else sosfiltfilt
    return f(sos, X, axis=2).astype(np.float32)


def _window(Xf, sfreq, win, anchor, onsets, cue_idx, y):
    """Slice a fixed-length window per trial. onset anchor uses each trial's accel
    onset (drops trials with no onset or an out-of-bounds window); cue anchor uses
    the shared cue index. Returns (segments[n, ch, wlen], y_kept)."""
    T = Xf.shape[2]
    w0 = int(round(win[0] * sfreq))
    w1 = int(round(win[1] * sfreq))
    segs, yk = [], []
    if anchor == "onset":
        for i in range(Xf.shape[0]):
            on = onsets[i]
            s, e = on + w0, on + w1
            if on >= 0 and s >= 0 and e <= T and e > s:
                segs.append(Xf[i, :, s:e]); yk.append(y[i])
    else:                                       # cue: identical slice every trial
        s, e = cue_idx + w0, cue_idx + w1
        if not (s >= 0 and e <= T and e > s):
            raise ValueError(f"cue window {win} out of bounds (cue@{cue_idx}, T={T})")
        for i in range(Xf.shape[0]):
            segs.append(Xf[i, :, s:e]); yk.append(y[i])
    return np.stack(segs).astype(np.float32), np.array(yk, dtype=int)


def _decimate(X, sfreq, target_fs, band, causal):
    """Downsample along time (axis=2) to ~target_fs. acausal -> resample_poly
    (= load_yeom). causal -> causal anti-alias low-pass (only if the band isn't
    already below the new Nyquist) + integer subsample."""
    if target_fs is None or abs(target_fs - sfreq) < 1e-6:
        return X, sfreq
    if not causal:
        frac = Fraction(target_fs / sfreq).limit_denominator(1000)
        Xr = resample_poly(X, frac.numerator, frac.denominator, axis=2).astype(np.float32)
        return Xr, float(target_fs)
    factor = max(1, int(round(sfreq / target_fs)))
    nyq_new = target_fs / 2.0
    hi = band if np.isscalar(band) else (band[1] if band else None)
    if hi is None or hi > 0.8 * nyq_new:        # need causal anti-aliasing
        sos = butter(4, 0.8 * nyq_new, btype="low", fs=sfreq, output="sos")
        X = sosfilt(sos, X, axis=2).astype(np.float32)
    return X[:, :, ::factor], float(sfreq / factor)


def causal_lag_ms(sfreq, band):
    """How far the CAUSAL filter lags real time, in ms -- the intrinsic latency of
    going causal. Measured empirically (robust for narrow band-passes, unlike a
    single-frequency group_delay): filter an impulse causally (sosfilt) and with
    the zero-phase reference (sosfiltfilt), then take the lag that maximises the
    cross-correlation of their envelopes. 0 for band=None."""
    if band is None:
        return 0.0
    n = int(round(2.0 * sfreq))                  # 2 s test buffer
    imp = np.zeros(n, dtype=float); imp[n // 4] = 1.0
    sos = _make_sos(sfreq, band)
    hc = np.abs(sosfilt(sos, imp))               # causal impulse response (envelope)
    hz = np.abs(sosfiltfilt(sos, imp))           # zero-phase reference (envelope)
    xc = np.correlate(hc, hz, mode="full")
    lag = int(np.argmax(xc) - (len(hz) - 1))     # samples the causal output trails by
    return lag / sfreq * 1000.0


# --------------------------------------------------------------------------- #
# one condition x band x window -> accuracy + 95% CI
# --------------------------------------------------------------------------- #
def run_cell(X_full, y, sfreq, onsets, cue_idx, cond, band, win,
             target_fs=50.0, folds=5, seed=0, max_iter=DEFAULT_MAX_ITER, n_jobs=1,
             decoder="direct", k=16):
    Xf = _filter_full(X_full, sfreq, band, causal=(cond["filt"] == "causal"))
    Xw, yw = _window(Xf, sfreq, win, cond["anchor"], onsets, cue_idx, y)
    Xd, fs_d = _decimate(Xw, sfreq, target_fs, band, causal=(cond["resamp"] == "causal"))

    counts = np.bincount(yw)
    nz = counts[counts > 0]
    if len(nz) < 2 or nz.min() < 2:
        return dict(acc=float("nan"), lo=float("nan"), hi=float("nan"),
                    n=len(yw), nfold=0)
    nfold = int(min(folds, nz.min()))
    cv = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
    est = _estimator(cond["scale"], fs_d, max_iter, decoder=decoder, k=k)
    y_pred = cross_val_predict(est, Xd, yw, cv=cv, n_jobs=n_jobs)
    err, lo, hi = bootstrap_error_ci(yw, y_pred, seed=seed)
    return dict(acc=100 * (1 - err), lo=100 * (1 - hi), hi=100 * (1 - lo),
                n=len(yw), nfold=nfold)


def run_grid(X_full, y, sfreq, onsets, cue_idx, class_names,
             target_fs=50.0, folds=5, seed=0, bands=None,
             max_iter=DEFAULT_MAX_ITER, n_jobs=1, decoder="direct", k=16):
    bands = BANDS if bands is None else bands
    chance = 100.0 / len(class_names)

    print(f"\nCausal-filter lag (how far the causal filter trails real time):")
    for b in bands:
        print(f"   band {_bstr(b):8s}: {causal_lag_ms(sfreq, b):6.1f} ms")

    rows = []
    for cond in CONDITIONS:
        wins = CUE_WINDOWS if cond["anchor"] == "cue" else ONSET_WINDOWS
        for band in bands:
            for win in wins:
                r = run_cell(X_full, y, sfreq, onsets, cue_idx, cond, band, win,
                             target_fs=target_fs, folds=folds, seed=seed,
                             max_iter=max_iter, n_jobs=n_jobs, decoder=decoder, k=k)
                r.update(cond=cond["name"], band=_bstr(band),
                         win=f"{win[0]:+.2f}..{win[1]:+.2f}")
                rows.append(r)
                print(f"  {cond['name']:13s} band {r['band']:8s} win {r['win']:13s}: "
                      f"{r['acc']:5.1f}%  CI[{r['lo']:4.1f},{r['hi']:4.1f}]  "
                      f"n={r['n']} cv={r['nfold']}", flush=True)

    _summary(rows, chance)
    return rows


def _bstr(b):
    return "none" if b is None else (f"<{b:g}" if np.isscalar(b) else f"{b[0]:g}-{b[1]:g}")


def _summary(rows, chance):
    print("\n================  HEADLINE (best band x window per condition)  ================")
    print(f"chance = {chance:.1f}%")
    best = {}
    for r in rows:
        if np.isnan(r["acc"]):
            continue
        if r["cond"] not in best or r["acc"] > best[r["cond"]]["acc"]:
            best[r["cond"]] = r
    ref = best.get("offline", {}).get("acc", float("nan"))
    print(f"{'condition':13s} {'best acc':>9s}  {'95% CI':>14s}  {'band/win':>22s}  {'Δ vs offline':>12s}")
    for cond in CONDITIONS:
        c = cond["name"]
        if c not in best:
            print(f"{c:13s}       (no valid cell)")
            continue
        r = best[c]
        d = "" if c == "offline" else f"{r['acc'] - ref:+5.1f} pts"
        print(f"{c:13s} {r['acc']:8.1f}%  [{r['lo']:4.1f},{r['hi']:4.1f}]  "
              f"{r['band']+' '+r['win']:>22s}  {d:>12s}")
    on = best.get("causal-onset", {}).get("acc")
    cue = best.get("causal-cue", {}).get("acc")
    if on is not None and not np.isnan(ref):
        print(f"\n  total causal penalty (offline -> causal-onset): {on - ref:+.1f} pts")
    if on is not None and cue is not None:
        print(f"  alignment penalty   (causal-onset -> causal-cue): {cue - on:+.1f} pts")
    print("==============================================================================")


# --------------------------------------------------------------------------- #
# NESTED CV: select band x window on calibration folds, LOCK, decode test.
# The honest "online" protocol -- hyperparameters are chosen WITHOUT touching the
# test trials, so the result is a deployable estimate rather than the selection-
# inflated grid maximum (which, by picking the best of N cells on the same data it
# is scored on, can't be reproduced online where the band/window must be fixed
# before the test trials exist).
# --------------------------------------------------------------------------- #
def _kept_mask(T, sfreq, win, anchor, onsets, cue_idx):
    """Boolean trial mask: which trials this window can be cut from. onset anchor
    drops trials whose [onset+w0, onset+w1] leaves [0, T); cue anchor keeps all if
    the shared slice is in-bounds."""
    w0 = int(round(win[0] * sfreq)); w1 = int(round(win[1] * sfreq))
    n = len(onsets)
    mask = np.zeros(n, dtype=bool)
    if anchor == "onset":
        for i in range(n):
            on = onsets[i]
            if on >= 0 and on + w0 >= 0 and on + w1 <= T and w1 > w0:
                mask[i] = True
    else:
        s, e = cue_idx + w0, cue_idx + w1
        if s >= 0 and e <= T and e > s:
            mask[:] = True
    return mask


def _extract(Xf, sfreq, win, anchor, onsets, cue_idx, idx):
    """Cut the window for the given trial indices (assumed all valid for `win`).
    Returns segments (len(idx), ch, wlen)."""
    w0 = int(round(win[0] * sfreq)); w1 = int(round(win[1] * sfreq))
    if anchor == "onset":
        segs = [Xf[i, :, onsets[i] + w0:onsets[i] + w1] for i in idx]
    else:
        s, e = cue_idx + w0, cue_idx + w1
        segs = [Xf[i, :, s:e] for i in idx]
    return np.stack(segs).astype(np.float32)


def _inner_score(scale, fs_d, Xtr, ytr, inner, max_iter, decoder="direct", k=16):
    """Inner-CV accuracy of one band x window candidate on the calibration trials.
    Top-level (not a closure) so it is picklable for joblib parallelism."""
    est = _estimator(scale, fs_d, max_iter, decoder=decoder, k=k)
    yhat = cross_val_predict(est, Xtr, ytr, cv=inner, n_jobs=1)
    return float(np.mean(yhat == ytr))


def run_condition_nested(X_full, y, sfreq, onsets, cue_idx, cond, bands,
                         target_fs, k_outer, k_inner, seed,
                         max_iter=DEFAULT_MAX_ITER, n_jobs=1, decoder="direct", k=16):
    """Nested CV for one condition. Outer folds give held-out test predictions;
    for each outer fold an inner CV over the band x window grid (on outer-train
    ONLY) picks the combo, which is locked and refit on all outer-train before
    predicting outer-test. Returns dict(acc, lo, hi, n, picks) or None if too few
    trials. Feature extraction (filter/window/decimate) is per-epoch and leak-free,
    so it is precomputed once per cell; only the scaler+logistic see fold splits."""
    T = X_full.shape[2]
    wins = CUE_WINDOWS if cond["anchor"] == "cue" else ONSET_WINDOWS
    # common trial pool: survive EVERY candidate window so all cells share trials
    mask = np.ones(len(y), dtype=bool)
    for w in wins:
        mask &= _kept_mask(T, sfreq, w, cond["anchor"], onsets, cue_idx)
    pool = np.where(mask)[0]
    yp = y[pool]
    counts = np.bincount(yp); nz = counts[counts > 0]
    if len(nz) < 2 or nz.min() < k_outer:
        return None

    causal_f = cond["filt"] == "causal"
    causal_r = cond["resamp"] == "causal"
    Xf_band = {b: _filter_full(X_full, sfreq, b, causal_f) for b in bands}
    feats = {}                                       # (band, win) -> (X[pool], fs)
    for b in bands:
        for w in wins:
            seg = _extract(Xf_band[b], sfreq, w, cond["anchor"], onsets, cue_idx, pool)
            seg, fs_d = _decimate(seg, sfreq, target_fs, b, causal_r)
            feats[(b, w)] = (seg, fs_d)
    candidates = [(b, w) for b in bands for w in wins]

    outer = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=seed)
    y_pred = np.empty(len(pool), dtype=int)
    picks = []
    dummy = np.zeros(len(yp))
    for tr, te in outer.split(dummy, yp):
        ytr = yp[tr]
        nz_in = np.bincount(ytr); nz_in = nz_in[nz_in > 0]
        kin = max(2, min(k_inner, int(nz_in.min())))
        inner = StratifiedKFold(n_splits=kin, shuffle=True, random_state=seed)
        # inner grid scan over candidates (calibration trials only)
        if n_jobs == 1:
            accs = [_inner_score(cond["scale"], feats[(b, w)][1], feats[(b, w)][0][tr],
                                 ytr, inner, max_iter, decoder, k) for (b, w) in candidates]
        else:
            from joblib import Parallel, delayed
            accs = Parallel(n_jobs=n_jobs, inner_max_num_threads=1)(
                delayed(_inner_score)(cond["scale"], feats[(b, w)][1],
                                      feats[(b, w)][0][tr], ytr, inner, max_iter, decoder, k)
                for (b, w) in candidates)
        best = candidates[int(np.argmax(accs))]      # locked on calibration only
        b, w = best
        Xc, fs_d = feats[(b, w)]
        est = _estimator(cond["scale"], fs_d, max_iter, decoder=decoder, k=k)
        est.fit(Xc[tr], ytr)
        y_pred[te] = est.predict(Xc[te])
        picks.append(best)
    err, lo, hi = bootstrap_error_ci(yp, y_pred, seed=seed)
    return dict(acc=100 * (1 - err), lo=100 * (1 - hi), hi=100 * (1 - lo),
                n=len(pool), picks=picks)


def _pick_str(picks):
    c = Counter(f"{_bstr(b)} {w[0]:+.2f}..{w[1]:+.2f}" for (b, w) in picks)
    top, cnt = c.most_common(1)[0]
    return f"{top} ({cnt}/{len(picks)})"


def run_nested(X_full, y, sfreq, onsets, cue_idx, class_names,
               target_fs=50.0, k_outer=5, k_inner=4, seed=0, bands=None,
               max_iter=DEFAULT_MAX_ITER, n_jobs=1, decoder="direct", k=16):
    bands = BANDS if bands is None else bands
    chance = 100.0 / len(class_names)
    print(f"\nCausal-filter lag (how far the causal filter trails real time):")
    for b in bands:
        print(f"   band {_bstr(b):8s}: {causal_lag_ms(sfreq, b):6.1f} ms")

    print(f"\n=== NESTED-CV (select band x window on calibration folds, lock, decode test) ===")
    print(f"chance = {chance:.1f}%   |   {k_outer}x outer, {k_inner}x inner "
          f"over {len(bands)} bands x windows   |   max_iter={max_iter}  jobs={n_jobs}"
          f"  decoder={decoder}" + (f" k={k}" if decoder == "manifold" else ""))
    res = {}
    for cond in CONDITIONS:
        r = run_condition_nested(X_full, y, sfreq, onsets, cue_idx, cond, bands,
                                 target_fs, k_outer, k_inner, seed,
                                 max_iter=max_iter, n_jobs=n_jobs, decoder=decoder, k=k)
        res[cond["name"]] = r
        if r is None:
            print(f"  {cond['name']:13s}   (too few trials)")
        else:
            print(f"  {cond['name']:13s} {r['acc']:5.1f}%  CI[{r['lo']:4.1f},{r['hi']:4.1f}]  "
                  f"n={r['n']:3d}  picked {_pick_str(r['picks'])}", flush=True)

    ref = res["offline"]["acc"] if res.get("offline") else None
    on = res["causal-onset"]["acc"] if res.get("causal-onset") else None
    cue = res["causal-cue"]["acc"] if res.get("causal-cue") else None
    if ref is not None and on is not None:
        print(f"\n  total causal penalty (offline -> causal-onset): {on - ref:+.1f} pts "
              f"(same onset trial pool)")
    if on is not None and cue is not None:
        print(f"  alignment penalty   (causal-onset -> causal-cue): {cue - on:+.1f} pts "
              f"(different anchor/pool -- approximate)")
    print("==============================================================================")
    return res


# --------------------------------------------------------------------------- #
# synthetic full-epoch generator (no accel needed; for the smoke test)
# --------------------------------------------------------------------------- #
def make_synthetic_full(n_per_class=40, n_ch=32, sfreq=200.0, n_classes=4,
                        tmin=-1.0, tmax=2.0, rt_mean=0.4, rt_sd=0.08, seed=0):
    """Full epochs with a class-specific SLOW (near-DC) pre-movement ramp in the
    0.3 s before a per-trial jittered onset, plus broadband noise -- so a low-freq
    decoder finds it, causal~=acausal (signal sits well inside the epoch), and
    cue-anchoring is slightly worse (RT jitter). Returns the load_full_epochs
    contract: X, y, sfreq, onsets, cue_idx, class_names."""
    rng = np.random.default_rng(seed)
    T = int(round((tmax - tmin) * sfreq))
    cue = int(round(-tmin * sfreq))
    W = rng.standard_normal((n_classes, n_ch))           # class spatial patterns
    X, y, onsets = [], [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            x = rng.standard_normal((n_ch, T)).astype(np.float32)
            on = cue + int(round(rng.normal(rt_mean, rt_sd) * sfreq))
            on = int(min(max(on, cue + 5), T - 5))
            w0 = max(on - int(round(0.3 * sfreq)), 0)
            ramp = np.zeros(T, dtype=np.float32)
            ramp[w0:on] = np.linspace(0.0, 1.0, on - w0, dtype=np.float32)
            x += 3.0 * (W[c][:, None] * ramp[None, :]).astype(np.float32)
            X.append(x); y.append(c); onsets.append(on)
    X = np.stack(X); y = np.array(y, int); onsets = np.array(onsets, int)
    perm = rng.permutation(len(y))
    return (X[perm], y[perm], float(sfreq), onsets[perm], cue,
            [f"dir{c}" for c in range(n_classes)])


def smoke():
    print("=== pseudo-online causal replay | SYNTHETIC smoke ===")
    X, y, sfreq, onsets, cue, names = make_synthetic_full(seed=0)
    print(f"synthetic: {X.shape[0]} trials | {X.shape[1]} ch | {X.shape[2]} samp | "
          f"sfreq={sfreq:.0f} | cue@{cue} | classes {names}")
    chance = 100.0 / len(names)

    # ---- grid protocol (diagnostic ceiling: max over band x window) ----
    rows = run_grid(X, y, sfreq, onsets, cue, names, target_fs=50.0, folds=5, seed=0)
    best = {}
    for r in rows:
        if not np.isnan(r["acc"]) and (r["cond"] not in best or r["acc"] > best[r["cond"]]):
            best[r["cond"]] = r["acc"]
    assert {c["name"] for c in CONDITIONS} <= set(best), set(best)
    assert best["offline"] > chance + 10, f"grid offline {best['offline']:.1f} not above chance"
    assert best["causal-onset"] > chance, f"grid causal-onset {best['causal-onset']:.1f} <= chance"
    # acausal filter on a signal well inside the epoch is closely reproduced by the
    # causal filter (the lag is small vs the window)
    assert best["causal-onset"] >= best["offline"] - 25, \
        f"grid causal-onset {best['causal-onset']:.1f} collapsed vs offline {best['offline']:.1f}"

    # ---- nested protocol (honest "online": select -> lock -> decode) ----
    res = run_nested(X, y, sfreq, onsets, cue, names, target_fs=50.0,
                     k_outer=5, k_inner=4, seed=0)
    present = {k for k, v in res.items() if v is not None}
    assert {c["name"] for c in CONDITIONS} <= present, present
    assert res["offline"]["acc"] > chance + 10, f"nested offline {res['offline']['acc']:.1f}"
    assert res["causal-onset"]["acc"] > chance, f"nested causal-onset {res['causal-onset']['acc']:.1f}"
    # nested is never higher than the grid ceiling (no winner's-curse inflation)
    assert res["offline"]["acc"] <= best["offline"] + 1e-6, \
        f"nested {res['offline']['acc']:.1f} > grid ceiling {best['offline']:.1f}"

    # ---- parallel path must match serial EXACTLY (determinism across n_jobs) ----
    off = [c for c in CONDITIONS if c["name"] == "offline"][0]
    r1 = run_condition_nested(X, y, sfreq, onsets, cue, off, BANDS, 50.0, 5, 4, 0,
                              max_iter=DEFAULT_MAX_ITER, n_jobs=1)
    r4 = run_condition_nested(X, y, sfreq, onsets, cue, off, BANDS, 50.0, 5, 4, 0,
                              max_iter=DEFAULT_MAX_ITER, n_jobs=4)
    assert abs(r1["acc"] - r4["acc"]) < 1e-9, (r1["acc"], r4["acc"])
    print(f"[parallel check] n_jobs 1 vs 4 -> {r1['acc']:.1f}% == {r4['acc']:.1f}%")
    print("\nSMOKE OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["synthetic", "yeom"], default="synthetic")
    p.add_argument("--protocol", choices=["nested", "grid"], default="nested",
                   help="nested = select band/window on calibration folds, lock, "
                        "decode test (honest online estimate); grid = per-cell CV "
                        "(diagnostic ceiling, selection-inflated)")
    p.add_argument("--yeom-path", default="./yeom_data")
    p.add_argument("--subject", default=None)
    p.add_argument("--session", default=None)
    p.add_argument("--sensor-type", choices=["mag", "grad", "all"], default="all")
    p.add_argument("--classes", nargs="+", default=None)
    p.add_argument("--target-fs", type=float, default=50.0)
    p.add_argument("--folds", type=int, default=5, help="grid protocol: CV folds")
    p.add_argument("--k-outer", type=int, default=5, help="nested protocol: outer folds")
    p.add_argument("--k-inner", type=int, default=4, help="nested protocol: inner folds")
    p.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                   help="logistic-regression lbfgs cap (lower = faster on weak-signal "
                        "cells; pass 5000 for the exact §12 estimator)")
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel workers over the band x window grid (BLAS pinned to "
                        "1 thread/worker to avoid oversubscription; 1 = serial)")
    p.add_argument("--decoder", choices=["direct", "manifold"], default="direct",
                   help="direct = flat L2 logistic over all features (tdlinear); "
                        "manifold = PLS-DA bottleneck (dim --k) before the same head")
    p.add_argument("--k", type=int, default=16,
                   help="manifold latent dimension (PLS-DA components); ignored for direct")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="optional .csv path to save results")
    args = p.parse_args()

    if args.source == "synthetic":
        smoke()
        return

    from loaders.yeom import load_full_epochs
    X, y, sfreq, onsets, cue, names = load_full_epochs(
        data_path=args.yeom_path, subject=args.subject, session=args.session,
        sensor_type=args.sensor_type,
        classes=tuple(args.classes) if args.classes else None, seed=args.seed)
    print(f"\n=== pseudo-online causal replay | source=yeom subject={args.subject} "
          f"protocol={args.protocol} ===")
    import csv
    if args.protocol == "nested":
        res = run_nested(X, y, sfreq, onsets, cue, names, target_fs=args.target_fs,
                         k_outer=args.k_outer, k_inner=args.k_inner, seed=args.seed,
                         max_iter=args.max_iter, n_jobs=args.jobs,
                         decoder=args.decoder, k=args.k)
        if args.out:
            with open(args.out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["cond", "acc", "lo", "hi", "n", "picked"])
                for c in CONDITIONS:
                    r = res.get(c["name"])
                    if r:
                        w.writerow([c["name"], r["acc"], r["lo"], r["hi"],
                                    r["n"], _pick_str(r["picks"])])
            print(f"saved {args.out}")
    else:
        rows = run_grid(X, y, sfreq, onsets, cue, names,
                        target_fs=args.target_fs, folds=args.folds, seed=args.seed,
                        max_iter=args.max_iter, n_jobs=args.jobs,
                        decoder=args.decoder, k=args.k)
        if args.out:
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["cond", "band", "win", "acc", "lo", "hi", "n", "nfold"])
                w.writeheader()
                for r in rows:
                    w.writerow({k: r[k] for k in w.fieldnames})
            print(f"saved {args.out}")


if __name__ == "__main__":
    main()
