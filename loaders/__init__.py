"""Data loaders. Every loader returns the same contract:

    X     : float ndarray, shape (n_trials, n_channels, n_times)   -- MEG epochs
    y     : int ndarray,   shape (n_trials,)                       -- class index
    sfreq : float                                                  -- sampling rate (Hz)
    class_names : list[str]                                        -- name per class index

Downstream code (features / decode / evaluate) is loader-agnostic, so the
synthetic generator and the real HCP loader are fully interchangeable.
"""
