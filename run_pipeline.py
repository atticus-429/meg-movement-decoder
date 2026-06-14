#!/usr/bin/env python
"""End-to-end HCP MEG Motor decoding pipeline.

  load epochs  ->  decode (cross-validated)  ->  evaluate (error rate + 95% CI)

Goal: decode which movement was made directly from MEG brain signals.

Examples
--------
# Smoke-test the whole pipeline on synthetic data (no download, runs anywhere):
python hcp_motor_decoder/run_pipeline.py --source synthetic --decoder csp
python hcp_motor_decoder/run_pipeline.py --source synthetic --decoder bandpower
python hcp_motor_decoder/run_pipeline.py --source synthetic --decoder eegnet

# Run on a real downloaded HCP subject (see download_hcp.md):
python hcp_motor_decoder/run_pipeline.py --source hcp \
       --hcp-path ./hcp_data --subject 100307 --decoder csp --classes LH RH
"""
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# quiet MNE's verbose covariance/rank logging during CSP fitting
logging.getLogger("mne").setLevel(logging.ERROR)

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config
from decode import build_decoder
from evaluate import (error_rate, bootstrap_error_ci, format_confusion,
                      per_class_accuracy)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["synthetic", "hcp", "yeom", "rathee"],
                   default="synthetic")
    p.add_argument("--decoder",
                   choices=["csp", "bandpower", "eegnet", "convtransformer", "tdlinear"],
                   default="csp")
    p.add_argument("--cv", choices=["kfold", "holdout"], default="kfold",
                   help="kfold = 5-fold CV; holdout = single split (torch trains once)")
    p.add_argument("--classes", nargs="+", default=list(config.DEFAULT_CLASSES),
                   help="movement classes to decode, e.g. LH RH (or LH RH LF RF)")
    p.add_argument("--folds", type=int, default=config.DEFAULT_FOLDS)
    p.add_argument("--bootstrap", type=int, default=config.DEFAULT_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=0)
    # synthetic options
    p.add_argument("--n-per-class", type=int, default=80)
    p.add_argument("--n-channels", type=int, default=64)
    # hcp options
    p.add_argument("--hcp-path", default="./hcp_data")
    p.add_argument("--subject", default="100307")
    # yeom options
    p.add_argument("--yeom-path", default="./yeom_data")
    p.add_argument("--rathee-path", default="./rathee_bids",
                   help="BIDS root for the Rathee MEG-imagery dataset")
    p.add_argument("--session", default=None)
    p.add_argument("--sensor-type", choices=["mag", "grad", "all"], default="all",
                   help="all=306 ch (unambiguous default); mag/grad need verified ordering")
    p.add_argument("--crop", nargs=2, type=float, default=None,
                   metavar=("LO", "HI"), help="crop window in seconds")
    p.add_argument("--resample", type=float, default=None, help="downsample to this Hz")
    p.add_argument("--band", nargs="+", type=float, default=None, metavar="HZ",
                   help="low-pass <H (one value) or band-pass L H (two values), "
                        "applied to full epochs before windowing (yeom source)")
    p.add_argument("--align", choices=["cue", "movement"], default="cue",
                   help="cue=fixed window from cue onset; movement=accel-gated pre-movement window")
    p.add_argument("--onset-k", type=float, default=4.0,
                   help="accel onset threshold (baseline_mean + k*std) for --align movement")
    p.add_argument("--tier1", action="store_true",
                   help="enable the Tier-1 regularization preset for convtransformer "
                        "(cosine LR+warmup, label smoothing, early stopping, light "
                        "augmentation); generalizable, no dataset-specific tuning")
    p.add_argument("--out", default=None, help="optional .npz path to save results")
    return p.parse_args()


def load_data(args):
    if args.source == "synthetic":
        from loaders.synthetic import make_synthetic_motor
        return make_synthetic_motor(
            n_trials_per_class=args.n_per_class, n_channels=args.n_channels,
            classes=tuple(args.classes), seed=args.seed)
    elif args.source == "hcp":
        from loaders.hcp import load_hcp_motor
        return load_hcp_motor(subject=args.subject, hcp_path=args.hcp_path,
                              classes=tuple(args.classes),
                              tmin=config.DEFAULT_TMIN, tmax=config.DEFAULT_TMAX)
    elif args.source == "yeom":
        from loaders.yeom import load_yeom
        # --classes defaults to the synthetic/hcp class set; for yeom let the
        # loader use its 4-direction default unless the user overrode it.
        yeom_classes = (None if list(args.classes) == list(config.DEFAULT_CLASSES)
                        else tuple(args.classes))
        band = None
        if args.band:
            band = args.band[0] if len(args.band) == 1 else tuple(args.band[:2])
        return load_yeom(data_path=args.yeom_path, subject=args.subject,
                         session=args.session, sensor_type=args.sensor_type,
                         classes=yeom_classes,
                         crop=tuple(args.crop) if args.crop else None,
                         resample=args.resample, align=args.align, onset_k=args.onset_k,
                         band=band)
    else:  # rathee (MEG motor/cognitive imagery; BIDS .fif)
        from loaders.rathee import load_rathee
        # default to all 4 imagery classes unless the user overrode --classes
        rathee_classes = (None if list(args.classes) == list(config.DEFAULT_CLASSES)
                          else tuple(args.classes))
        tmin, tmax = (args.crop if args.crop else (0.5, 3.5))   # imagery window (s)
        return load_rathee(bids_root=args.rathee_path, subject=args.subject,
                           session=(args.session or "1"), sensor_type=args.sensor_type,
                           classes=rathee_classes, tmin=tmin, tmax=tmax,
                           resample=args.resample)


def main():
    args = parse_args()
    print(f"\n=== MEG movement decoder | source={args.source} "
          f"decoder={args.decoder} cv={args.cv} ===")

    X, y, sfreq, class_names = load_data(args)
    n_classes = len(class_names)
    print(f"Loaded {X.shape[0]} trials | {X.shape[1]} channels | "
          f"{X.shape[2]} samples | sfreq={sfreq:.1f} Hz")
    counts = {class_names[c]: int((y == c).sum()) for c in range(n_classes)}
    print(f"Class counts: {counts}")

    dec_kwargs = {}
    if args.tier1:
        if args.decoder != "convtransformer":
            print(f"[warn] --tier1 only affects convtransformer; "
                  f"ignoring for decoder={args.decoder}")
        else:
            # generalizable regularization preset (not Yeom-tuned)
            dec_kwargs = dict(lr_schedule="cosine", warmup_frac=0.1,
                              label_smoothing=0.1, early_stopping=True,
                              val_frac=0.2, patience=15,
                              noise_std=0.1, time_jitter=3,
                              channel_drop=0.1, mixup_alpha=0.2)
            print(f"[tier1] convtransformer regularization preset: {dec_kwargs}")
    decoder = build_decoder(args.decoder, sfreq=sfreq,
                            beta_band=config.BETA_BAND, mu_band=config.MU_BAND,
                            **dec_kwargs)

    # Held-out predictions. kfold = every trial held out once (analogous to the
    # parent repo's "HeldOutTrials"); holdout = one split so a torch decoder
    # trains once instead of n_folds times.
    if args.cv == "holdout":
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                                  stratify=y, random_state=args.seed)
        print(f"Running single stratified holdout split "
              f"({len(tr)} train / {len(te)} test, train once)...")
        decoder.fit(X[tr], y[tr])
        y_pred, y_eval = decoder.predict(X[te]), y[te]
    else:
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        print(f"Running {args.folds}-fold cross-validation...")
        y_pred, y_eval = cross_val_predict(decoder, X, y, cv=cv, n_jobs=1), y

    # ---- evaluation (error rate + bootstrap 95% CI, per repo's technique) ----
    err, lo, hi = bootstrap_error_ci(y_eval, y_pred, n_resamples=args.bootstrap,
                                     seed=args.seed)
    acc = 1.0 - err
    chance = 1.0 / n_classes
    conf_str, cm = format_confusion(y_eval, y_pred, class_names)
    pca = per_class_accuracy(cm)

    # accuracy CI is the mirror of the error-rate CI
    acc_lo, acc_hi = 1.0 - hi, 1.0 - lo

    print("\n---------------- RESULTS ----------------")
    print(f"Decoding accuracy : {100*acc:5.2f}%   95% CI [{100*acc_lo:.2f}, {100*acc_hi:.2f}]"
          f"   (chance = {100*chance:.1f}%)")
    print(f"Error rate        : {100*err:5.2f}%   95% CI [{100*lo:.2f}, {100*hi:.2f}]")
    print("Per-class accuracy: " +
          ", ".join(f"{class_names[c]} {100*pca[c]:.1f}%" for c in range(n_classes)))
    print("\nConfusion matrix:")
    print(conf_str)
    print("-----------------------------------------")

    above = acc_lo > chance
    print(f"Verdict: decoding is {'ABOVE chance' if above else 'NOT above chance'} "
          f"(accuracy CI {'excludes' if above else 'includes'} the {100*chance:.1f}% chance level).")

    if args.out:
        np.savez(args.out, y_true=y_eval, y_pred=y_pred, cm=cm,
                 acc=acc, err=err, ci=(lo, hi), class_names=class_names)
        print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
