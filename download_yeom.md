# Downloading the Yeom 2023 MEG 3D-reaching dataset

The PRIMARY Rung-1 dataset. Unlike HCP, it is **openly downloadable — no account,
no data-use agreement** (CC-BY 4.0).

- Paper: Yeom, Kim, Chung (2023), *Scientific Data* 10:552 —
  https://www.nature.com/articles/s41597-023-02454-y (open access mirror:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10444808/)
- Data: figshare collection **DOI 10.6084/m9.figshare.c.6431021**
- Analysis code (MATLAB, reference): https://github.com/honggi82/Scientific_Data_2023

## What it is

- 306-ch Elekta/MEGIN MEG (102 magnetometers + 204 planar gradiometers), 600.615 Hz.
- 9 subjects × 2 sessions; **4 cued reach directions** (upper-left, upper-right,
  bottom-left, bottom-right); ~30 trials/direction/session; epochs −1..+2 s from cue.
- Three forms provided: raw `.fif`, **epoched `.mat`** (what this loader reads),
  ICA-cleaned epoched `.mat`. Each epoched `.mat` is a 4-cell array (one cell per
  direction), each cell `channels × time × trials`, 319 channels
  (1-306 MEG, 307-315 triggers, 316 EOG, 317-319 accelerometer).

## On Colab (recommended for training)

Use **`colab_runner.ipynb`** — it mounts Drive, downloads only the subject you
need straight from figshare (via HTTP range requests, ~0.5–1 GB instead of the
full 9.3 GB), and trains on the GPU. The figshare files are single ~9.3 GB zips
(`ica` file id `41898840`, `epoched` `41898714`); the notebook lists the archive
and extracts just your subject's `.mat` to Drive.

## Get it manually

1. Open the figshare collection (DOI above) in a browser and download the
   **epoched .mat** file(s) for at least one subject/session.
2. Put them in `./yeom_data/` (any filename; the loader globs `*.mat` and you
   select with `--subject` / `--session` substring matching):
   ```
   mkdir -p yeom_data
   # move/download the epoched .mat files into ./yeom_data/
   ```

## Run

```bash
# cheap first pass: downsample + crop + single split (torch trains once)
python hcp_motor_decoder/run_pipeline.py --source yeom --yeom-path ./yeom_data \
       --subject <id> --session <s> --decoder convtransformer \
       --resample 150 --crop 0.0 1.5 --cv holdout

# full 5-fold + a CSP baseline for comparison
python hcp_motor_decoder/run_pipeline.py --source yeom --yeom-path ./yeom_data \
       --subject <id> --decoder convtransformer --resample 150
python hcp_motor_decoder/run_pipeline.py --source yeom --yeom-path ./yeom_data \
       --subject <id> --decoder csp --sensor-type all
```

Default decodes all **4 directions** (chance = 25%). The loader prints the matched
filename, the `.mat` variable it used, and per-class trial counts — check these
look right (4 classes, ~30 trials each) before trusting accuracy.

## Gotchas

- **v7.3 `.mat`**: if `scipy.io.loadmat` errors, `pip install mat73` (the loader
  falls back to it automatically).
- **Channel selection**: `--sensor-type all` (default, 306 ch) is unambiguous and
  barely costs more than mag-only here. `--sensor-type mag`/`grad` assume MEGIN
  triplet ordering (mags at `2::3`) — verify against the dataset's channel names
  first, or pass an explicit `mag_idx` in code.
- **Direction labels** are positional (`dir0..dir3`); confirm the physical
  direction mapping from the dataset docs before claiming *what* is decoded.
- **sfreq** is hard-coded to 600.615 Hz (the epoched `.mat` may not store it);
  override via the loader's `sfreq=` if a file differs.
