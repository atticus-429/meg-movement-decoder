"""Shared configuration for the HCP MEG Motor decoder.

The decoding goal: classify which movement a subject made directly from the
MEG sensor signals (decode motor movement from brain activity).

HCP "Motor" task: visually-cued movements of left hand / right hand /
left foot / right foot. The flagship, high-SNR contrast is hand LATERALITY
(left hand vs right hand), driven by contralateral mu/beta desynchronization
over sensorimotor cortex -- that is the default target here.
"""

# Canonical HCP Motor movement classes.
ALL_CLASSES = ("LH", "RH", "LF", "RF")          # left/right hand, left/right foot
CLASS_LONG = {
    "LH": "left hand", "RH": "right hand",
    "LF": "left foot", "RF": "right foot",
}

# Default decoding problem: hand laterality (binary, easiest & most robust).
DEFAULT_CLASSES = ("LH", "RH")

# Time window (seconds, relative to movement cue) to analyse.
DEFAULT_TMIN = -0.5
DEFAULT_TMAX = 1.5

# Frequency band used by the CSP / band-power decoders (sensorimotor beta).
# A second (mu) band is added automatically by the band-power decoder.
BETA_BAND = (13.0, 30.0)
MU_BAND = (8.0, 13.0)

# Cross-validation: number of stratified folds over trials.
# This mirrors the repo's "HeldOutTrials" idea -- single trials held out for test.
DEFAULT_FOLDS = 5

# Bootstrap resamples for the 95% CI on the error rate.
# Technique borrowed from this repo's SummarizeRNNPerformance.ipynb.
DEFAULT_BOOTSTRAP = 10000
