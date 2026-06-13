# Downloading HCP MEG "Motor" data

The decoder needs one (or more) HCP Young-Adult subjects that have **MEG** data.
HCP MEG is **open access**, but you must accept the data-use terms once and use
the AWS credentials it gives you. I (the assistant) cannot do this for you — it
is tied to your own ConnectomeDB account.

## 1. One-time setup (free)

> As of 2025–2026, HCP folded ConnectomeDB INTO BALSA: visiting
> `db.humanconnectome.org` now redirects to `balsa.wustl.edu` — that is expected.
> Your BALSA account is the correct account. The MEG data lives in the original
> 2017 **S1200** release, which is reached via the **ConnectomeDB tab** inside
> BALSA and downloaded over **AWS S3** (S3 key generation re-enabled 2026-03-31).
> The newer "HCP-YA 2025" packages are BALSA-only MRI repackaging — NOT what you
> want for MEG.

1. Log in, then open the HCP-YA project page:
   **https://balsa.wustl.edu/project?project=HCP_YA**
   (db.humanconnectome.org redirects into BALSA; this is the right page.)
2. Click **"Data Use Terms"** and accept the *WU-Minn HCP Consortium Open Access
   Data Use Terms*.
3. Click **"Get/Reset AWS S3 Access"** to mint your keys. Save all three values
   (the secret is shown only once):
   **USERNAME, ACCESS KEY ID, SECRET ACCESS KEY**.
4. Configure AWS locally and confirm the keys work:
   ```bash
   pip install awscli
   aws configure          # paste Access Key ID / Secret Access Key; region us-east-1
   aws s3 ls s3://hcp-openaccess/HCP_1200/100307/MEG/   # should list Motor/ etc.
   ```

## 2. Pick a subject with MEG

Not all subjects have MEG. Known MEG subjects include **100307, 102816, 105923,
106521, 108323, 109123, 111514, 113922**. Start with `100307`.

List what MEG packages exist for a subject:
```bash
aws s3 ls s3://hcp-openaccess/HCP_1200/100307/MEG/
```

## 3. Download the Motor task (+ what mne-hcp needs)

`mne-hcp` expects a local tree `<hcp_path>/<subject>/...` containing the
unprocessed Motor runs and the metadata. Download into `./hcp_data`:

```bash
SUBJ=100307
DEST=./hcp_data/$SUBJ
mkdir -p $DEST

# Unprocessed Motor MEG runs (what loaders/hcp.py reads):
aws s3 cp --recursive \
  s3://hcp-openaccess/HCP_1200/$SUBJ/unprocessed/MEG/ \
  $DEST/unprocessed/MEG/ \
  --exclude "*" --include "*Motor*"

# MEG metadata / anatomy (head model, sensor info):
aws s3 cp --recursive \
  s3://hcp-openaccess/HCP_1200/$SUBJ/MEG/anatomy/ \
  $DEST/MEG/anatomy/
```

(If you prefer the already-epoched preprocessed data, also grab
`s3://hcp-openaccess/HCP_1200/$SUBJ/MEG/Motor/tmegpreproc/` and adapt the loader —
that route is smaller, ~hundreds of MB.)

**Size:** the unprocessed Motor runs for one subject are ~1.5–3 GB. We decode in
SENSOR space, so you do NOT need the anatomy/MRI + head models (skip the
`MEG/anatomy/` download above unless you later want source localization). Check
the exact size first with:
```bash
aws s3 ls --summarize --human-readable --recursive \
  s3://hcp-openaccess/HCP_1200/$SUBJ/MEG/Motor/
```
For reference: one subject's *complete* MEG (all tasks + all processing levels) is
~20–30 GB; the entire ~95-subject MEG release is ~2–3 TB.

## 4. Run the decoder on it

```bash
pip install mne-hcp
python hcp_motor_decoder/run_pipeline.py \
    --source hcp --hcp-path ./hcp_data --subject 100307 \
    --decoder csp --classes LH RH
```

## Notes / gotchas

- **Trigger codes:** on first run, `loaders/hcp.py` prints the movement-condition
  codes it finds. HCP field conventions vary slightly across releases; if the
  class counts look wrong, edit `DEFAULT_CODE_MAP` in `loaders/hcp.py` to match.
- **mne-hcp + recent MNE:** `mne-hcp` is older; if it errors against MNE ≥ 1.6,
  create an isolated env (e.g. `conda create -n hcp python=3.9 mne=1.4 mne-hcp`).
- **Foot classes** (`LF`, `RF`) are harder to decode than hand laterality; start
  with `--classes LH RH`.
