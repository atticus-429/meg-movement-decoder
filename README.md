# HCP MEG Motor Decoder

Decode **which movement a person made directly from their MEG brain signals**,
using the [HCP Young-Adult](https://www.humanconnectome.org/study/hcp-young-adult)
"Motor" task (visually-cued left-hand / right-hand / left-foot / right-foot
movements). The default problem is **hand laterality** (left vs right hand),
driven by contralateral mu/beta desynchronization over sensorimotor cortex.

This is a self-contained companion to the handwriting-BCI pipeline in the parent
repo. It is a *trial-classification* decoder (discrete cued movements), not a
sequence-to-text decoder, but it **borrows this repo's evaluation technique**:
the error-rate + 95% bootstrap-CI method from `SummarizeRNNPerformance.ipynb`
(resample held-out trials 10,000× → 2.5/97.5 percentiles).

## What it does

```
load epochs ─▶ decode (cross-validated) ─▶ evaluate (error rate + 95% CI)
(n_trials, n_channels, n_times)            accuracy, confusion, above-chance?
```

Two data sources behind one array contract `(X, y, sfreq, class_names)`:
- **`synthetic`** — a physiologically-motivated MEG generator (contralateral beta
  ERD + movement-evoked field + noise + behavioral lapses). Needs no data/GPU;
  verifies the whole pipeline runs. **This is what makes the repo runnable
  end-to-end out of the box.**
- **`yeom`** — Yeom 2023 MEG 3D-reaching loader (`loaders/yeom.py`): 4 cued reach
  directions, **openly downloadable** (figshare, no credentials, plain scipy — no
  `mne-hcp`). The primary dataset for the conv→transformer (Rung 1). See
  `download_yeom.md`.
- **`hcp`** — the real HCP Motor loader (`loaders/hcp.py`, via `mne-hcp`); secondary.

Four decoders, all exposed as scikit-learn estimators so they share one
cross-validation + evaluation path:

| decoder | what it is | deps |
|---|---|---|
| `csp` *(default)* | band-pass → Common Spatial Patterns → LDA — gold standard for sensorimotor-rhythm decoding | mne |
| `bandpower` | log band-power per channel → logistic regression | numpy/scipy/sklearn only |
| `eegnet` | compact EEGNet-style CNN | torch (optional) |
| `convtransformer` | Brain2Qwerty/Défossez-style conv front-end → temporal transformer → pooled classifier (Rung 1) | torch |

## Quickstart (no data, runs anywhere)

```bash
pip install -r hcp_motor_decoder/requirements.txt   # numpy scipy scikit-learn mne (torch optional)

python hcp_motor_decoder/run_pipeline.py --source synthetic --decoder csp
python hcp_motor_decoder/run_pipeline.py --source synthetic --decoder bandpower
python hcp_motor_decoder/run_pipeline.py --source synthetic --decoder csp --classes LH RH LF RF
# conv->transformer (torch); --cv holdout trains once instead of 5x:
python hcp_motor_decoder/run_pipeline.py --source synthetic --decoder convtransformer --cv holdout
```

For the real Yeom reaching data (4 directions), see `download_yeom.md`, then e.g.
`--source yeom --yeom-path ./yeom_data --subject <id> --decoder convtransformer --resample 150`.

### Train on Colab (GPU)

`colab_runner.ipynb` is a ready-to-run Colab notebook that **mounts Google Drive →
downloads the Yeom archive and extracts your subject to Drive → loads from Drive →
trains the conv→transformer on the GPU** (full 5-fold CV becomes practical there).
Set the runtime to GPU; the code is cloned from the public repo (no token needed).
The Yeom `.mat` are MATLAB v7.3, so the notebook installs `mat73` to read them.

### Verified synthetic results (seed 0, 5-fold CV, 160 trials)

| decoder | accuracy | 95% CI | chance | verdict |
|---|---|---|---|---|
| `csp` | 61.9% | [54.4, 69.4] | 50% | above chance ✓ |
| `bandpower` | 92.5% | [88.1, 96.2] | 50% | above chance ✓ |
| `eegnet` | 91.2% | [86.9, 95.6] | 50% | above chance ✓ |
| `convtransformer` † | 95.8% | [87.5, 100] | 50% | above chance ✓ |
| `csp` (4-class LH/RH/LF/RF) | 35.0% | [30.0, 40.3] | 25% | above chance ✓ |

† `convtransformer` is reported with `--cv holdout` (single split — 5-fold × 60 epochs is
slow on CPU; use a GPU or `--cv holdout`). Its multi-class head is separately confirmed at
100% on a genuinely-separable 4-class toy. The *synthetic* 4-class is degenerate by
construction (LH≡LF, RH≡RF share a hemisphere, ~50% ceiling), so it is a plumbing test only —
the real 4-class evaluation is the Yeom reaching data.

The synthetic deliberately contains both signal types real movement produces: an
**induced beta ERD** (a power drop — what `csp`/`bandpower` exploit) and a
**phase-locked movement-evoked field** (what the time-domain `eegnet` CNN
exploits). It also injects noise and behavioral "lapse" trials so no decoder is
degenerate. These numbers only prove the mechanics — real-MEG accuracy depends
on the subject and on tuning bands/window/components.

## Running on real HCP data

1. Download a subject — see **[download_hcp.md](download_hcp.md)** (free, credentialed).
2. ```bash
   pip install mne-hcp
   python hcp_motor_decoder/run_pipeline.py \
       --source hcp --hcp-path ./hcp_data --subject 100307 \
       --decoder csp --classes LH RH
   ```

## Files

```
hcp_motor_decoder/
  run_pipeline.py     # CLI orchestrator (load → decode → evaluate)
  config.py           # classes, bands, CV/bootstrap defaults
  features.py         # band-pass + log band-power (numpy/scipy)
  decode.py           # CSP+LDA, bandpower+LogReg, EEGNet — sklearn estimators
  evaluate.py         # error rate + bootstrap 95% CI (from SummarizeRNNPerformance.ipynb)
  loaders/
    synthetic.py      # synthetic MEG-Motor generator  (verified path)
    hcp.py            # real HCP loader via mne-hcp      (real-data path)
  download_hcp.md     # credentialed S3 download steps
  requirements.txt
```

## Honest scope / limitations

- **CPU is fine** — MEG trial classification (CSP/bandpower) needs no GPU.
- These are **discrete, cued** movements → laterality / movement-class decoding.
  This is *not* continuous handwriting-trajectory decoding; HCP has no such labels.
- The real-data loader (`loaders/hcp.py`) was written against the documented
  `mne-hcp` API but **not executed against real data here** (requires your
  credentialed download). Check the printed trigger-code table on first run.
- The synthetic path exists to verify pipeline *mechanics*, not to predict
  real accuracy. Real single-subject HCP hand-laterality MEG decoding with CSP
  typically lands well above chance; expect to tune bands/window/CSP components.
```
