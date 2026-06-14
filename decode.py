"""Decoders that map MEG epochs (n_trials, n_channels, n_times) -> class.

Three options, all exposed as scikit-learn estimators so they share one
cross-validation / evaluation path:

  - "csp"       : band-pass -> Common Spatial Patterns -> LDA   (gold-standard
                  sensorimotor-rhythm decoder; uses MNE's CSP)
  - "bandpower" : log band-power per channel -> LogisticRegression
                  (dependency-light: numpy/scipy/sklearn only)
  - "eegnet"    : a compact EEGNet-style CNN (PyTorch, CPU-friendly, optional)
  - "convtransformer" : Brain2Qwerty/Defossez-style conv front-end -> temporal
                  transformer -> pooled classifier (PyTorch). Decodes a single
                  movement event per trial; the conv->transformer architecture
                  is the Rung-1 stepping stone toward sequence/language decoding.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from features import bandpass, band_power_features


# --------------------------------------------------------------------------- #
# sklearn transformers operating on 3-D (trials, channels, times) arrays
# --------------------------------------------------------------------------- #
class BandPass(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=250.0, l_freq=13.0, h_freq=30.0, order=4):
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return bandpass(X, self.sfreq, self.l_freq, self.h_freq, self.order)


class BandPowerFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=250.0, bands=((8, 13), (13, 30)), order=4):
        self.sfreq = sfreq
        self.bands = bands
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return band_power_features(X, self.sfreq, self.bands, self.order)


# --------------------------------------------------------------------------- #
# Optional EEGNet-style CNN (PyTorch). Wrapped as an sklearn classifier.
# --------------------------------------------------------------------------- #
class TorchEEGNet(BaseEstimator, ClassifierMixin):
    def __init__(self, sfreq=250.0, n_epochs=40, lr=1e-3, batch_size=32, seed=0):
        self.sfreq = sfreq
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed

    def _build(self, n_channels, n_times, n_classes):
        import torch
        import torch.nn as nn

        F1, D, F2 = 8, 2, 16
        kern = max(int(self.sfreq // 2), 8)        # ~half-second temporal kernel

        class EEGNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.firstconv = nn.Sequential(
                    nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
                    nn.BatchNorm2d(F1),
                )
                self.depthwise = nn.Sequential(
                    nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
                    nn.BatchNorm2d(F1 * D), nn.ELU(),
                    nn.AvgPool2d((1, 4)), nn.Dropout(0.25),
                )
                self.separable = nn.Sequential(
                    nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
                    nn.BatchNorm2d(F2), nn.ELU(),
                    nn.AvgPool2d((1, 8)), nn.Dropout(0.25),
                )
                self.classify = None
                self._n_classes = n_classes

            def forward(self, x):
                x = self.firstconv(x)
                x = self.depthwise(x)
                x = self.separable(x)
                x = x.flatten(1)
                if self.classify is None:
                    self.classify = nn.Linear(x.shape[1], self._n_classes).to(x.device)
                return self.classify(x)

        return EEGNet()

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_channels, n_times = X.shape[1], X.shape[2]
        self.net_ = self._build(n_channels, n_times, n_classes)

        # standardize per channel using train statistics
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-6
        Xn = (X - self.mean_) / self.std_

        xb = torch.tensor(Xn[:, None, :, :], dtype=torch.float32)
        yb = torch.tensor(y, dtype=torch.long)
        self.net_(xb[:2])                            # lazily build classifier head
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        lossf = torch.nn.CrossEntropyLoss()
        n = len(y)
        rng = np.random.default_rng(self.seed)
        self.net_.train()
        for _ in range(self.n_epochs):
            for s in range(0, n, self.batch_size):
                idx = rng.permutation(n)[s:s + self.batch_size]
                opt.zero_grad()
                out = self.net_(xb[idx])
                loss = lossf(out, yb[idx])
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        import torch
        Xn = (X - self.mean_) / self.std_
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(torch.tensor(Xn[:, None, :, :], dtype=torch.float32))
        return self.classes_[out.argmax(1).numpy()]


# --------------------------------------------------------------------------- #
# Brain2Qwerty / Defossez-style conv -> transformer (PyTorch), as an sklearn
# classifier so it shares the same CV / evaluation path as the other decoders.
# Input convention (batch, n_channels, n_times): channels are the conv feature
# dimension (NOT EEGNet's image layout). One movement event per trial.
# --------------------------------------------------------------------------- #
class TorchConvTransformer(BaseEstimator, ClassifierMixin):
    def __init__(self, sfreq=250.0, n_spatial=128,
                 conv_dim=128, n_conv_blocks=4, conv_kernel=7,
                 conv_dilations=(1, 2, 4, 8), conv_stride=4,
                 d_model=128, n_heads=4, n_layers=3, ff_mult=4,
                 pool="attention", dropout=0.2,
                 n_epochs=60, lr=1e-3, weight_decay=1e-4, batch_size=32,
                 device="auto", max_seq_len=2048, class_weight=True, seed=0,
                 # --- Tier-1 regularization (all OFF by default -> default
                 #     pipeline is byte-for-byte unchanged; opt in per-arg) ---
                 label_smoothing=0.0, lr_schedule="none", warmup_frac=0.1,
                 early_stopping=False, val_frac=0.2, patience=15, min_epochs=10,
                 noise_std=0.0, time_jitter=0, channel_drop=0.0, mixup_alpha=0.0):
        # store every arg verbatim (clone-safe); no derived state here
        self.sfreq = sfreq
        self.n_spatial = n_spatial
        self.conv_dim = conv_dim
        self.n_conv_blocks = n_conv_blocks
        self.conv_kernel = conv_kernel
        self.conv_dilations = conv_dilations
        self.conv_stride = conv_stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_mult = ff_mult
        self.pool = pool
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = device
        self.max_seq_len = max_seq_len
        self.class_weight = class_weight
        self.seed = seed
        # Tier-1 knobs (generalizable; no Yeom-specific tuning)
        self.label_smoothing = label_smoothing   # soft targets in CE
        self.lr_schedule = lr_schedule           # "none" | "cosine" (warmup+decay)
        self.warmup_frac = warmup_frac           # cosine warmup as frac of steps
        self.early_stopping = early_stopping     # hold out val_frac, keep best
        self.val_frac = val_frac
        self.patience = patience                 # epochs w/o val improvement
        self.min_epochs = min_epochs             # never stop before this
        self.noise_std = noise_std               # additive Gaussian (z-units)
        self.time_jitter = time_jitter           # +/- samples circular shift
        self.channel_drop = channel_drop         # frac of channels zeroed/sample
        self.mixup_alpha = mixup_alpha           # Beta(a,a) mixup; 0 = off

    def _build(self, n_channels, n_classes):
        import torch
        import torch.nn as nn

        cfg = self
        dilations = [cfg.conv_dilations[i % len(cfg.conv_dilations)]
                     for i in range(cfg.n_conv_blocks)]

        def sinusoidal_pe(max_len, d):
            pe = torch.zeros(max_len, d)
            pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32)
                            * (-np.log(10000.0) / d))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe.unsqueeze(0)                       # (1, max_len, d)

        class ConvBlock(nn.Module):
            def __init__(self, dim, kernel, dilation, dropout):
                super().__init__()
                pad = (kernel - 1) // 2 * dilation       # "same" length, stride 1
                self.conv1 = nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation)
                self.conv2 = nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation)
                self.norm = nn.GroupNorm(1, dim)         # batch-size-robust norm
                self.drop = nn.Dropout(dropout)
                self.act = nn.GELU()

            def forward(self, x):
                r = x
                x = self.drop(self.act(self.conv1(x)))
                x = self.conv2(x)
                return self.act(self.norm(x + r))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.spatial = nn.Sequential(            # 1x1-over-channels mixer
                    nn.Conv1d(n_channels, cfg.n_spatial, 1), nn.GELU())
                self.proj_in = nn.Conv1d(cfg.n_spatial, cfg.conv_dim, 1)
                self.blocks = nn.ModuleList([
                    ConvBlock(cfg.conv_dim, cfg.conv_kernel, d, cfg.dropout)
                    for d in dilations])
                self.downsample = nn.Conv1d(cfg.conv_dim, cfg.d_model,
                                            cfg.conv_stride, stride=cfg.conv_stride)
                enc = nn.TransformerEncoderLayer(
                    cfg.d_model, cfg.n_heads, dim_feedforward=cfg.ff_mult * cfg.d_model,
                    dropout=cfg.dropout, activation="gelu", batch_first=True,
                    norm_first=True)
                self.transformer = nn.TransformerEncoder(enc, cfg.n_layers)
                self.register_buffer("pe", sinusoidal_pe(cfg.max_seq_len, cfg.d_model))
                self._pool = cfg.pool
                if cfg.pool == "attention":
                    self.query = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
                self.classify = nn.Linear(cfg.d_model, n_classes)

            def forward(self, x):                        # x: (B, C, T)
                x = self.proj_in(self.spatial(x))
                for b in self.blocks:
                    x = b(x)
                x = self.downsample(x).transpose(1, 2)   # (B, T_tok, d_model)
                t_tok = x.shape[1]
                assert t_tok <= cfg.max_seq_len, (
                    f"token sequence length {t_tok} > max_seq_len {cfg.max_seq_len}; "
                    f"raise max_seq_len or conv_stride")
                x = self.transformer(x + self.pe[:, :t_tok])
                if self._pool == "attention":
                    q = self.query.expand(x.shape[0], -1, -1)
                    attn = torch.softmax((q @ x.transpose(1, 2))
                                         / (x.shape[-1] ** 0.5), dim=-1)
                    pooled = (attn @ x).squeeze(1)
                else:
                    pooled = x.mean(1)
                return self.classify(pooled)

        return Net()

    def _augment(self, x, y, g):
        """Per-batch augmentation (training only). Returns (x, y_a, y_b, lam):
        lam is None unless mixup is active, in which case the caller mixes the
        loss as lam*CE(out,y_a) + (1-lam)*CE(out,y_b). All ops no-op when their
        hyperparameter is 0/off, so the default path is untouched."""
        import torch
        y_a, y_b, lam = y, y, None
        if self.noise_std and self.noise_std > 0:
            x = x + self.noise_std * torch.randn(x.shape, generator=g,
                                                 device=x.device, dtype=x.dtype)
        if self.time_jitter and self.time_jitter > 0:          # circular shift
            B, C, T = x.shape
            ms = int(self.time_jitter)
            shifts = torch.randint(-ms, ms + 1, (B,), generator=g, device=x.device)
            ar = torch.arange(T, device=x.device)
            idx = (ar.unsqueeze(0) - shifts.unsqueeze(1)) % T   # (B, T)
            x = torch.gather(x, 2, idx.unsqueeze(1).expand(B, C, T))
        if self.channel_drop and self.channel_drop > 0:
            B, C, _ = x.shape
            keep = (torch.rand(B, C, 1, generator=g, device=x.device)
                    >= self.channel_drop).to(x.dtype)
            x = x * keep / (1.0 - self.channel_drop)            # keep expectation
        if self.mixup_alpha and self.mixup_alpha > 0:
            a = float(self.mixup_alpha)
            lam = float(torch.distributions.Beta(a, a).sample())
            lam = max(lam, 1.0 - lam)                           # label stays nearer y_a
            perm = torch.randperm(x.shape[0], generator=g, device=x.device)
            x = lam * x + (1.0 - lam) * x[perm]
            y_b = y[perm]
        return x, y_a, y_b, lam

    def _run_training(self, xb, yb, xb_all, yb_all, va, weight, lossf,
                      opt, sched, n_epochs, augment=True):
        """Shared minibatch training loop for fit() (pretrain) and finetune()
        (calibrate). Trains net_ in place for n_epochs; if `va` is given, tracks
        held-out val loss and restores the best weights (early stopping)."""
        import torch
        n = len(yb)
        rng = np.random.default_rng(self.seed)
        g = torch.Generator(device=self.device_)
        g.manual_seed(int(self.seed))
        best_val, best_state, bad = np.inf, None, 0
        for epoch in range(n_epochs):
            self.net_.train()
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                xbatch = xb[idx].to(self.device_)
                ybatch = yb[idx].to(self.device_)
                if augment:
                    xbatch, y_a, y_b, lam = self._augment(xbatch, ybatch, g)
                else:
                    y_a, y_b, lam = ybatch, ybatch, None
                opt.zero_grad()
                out = self.net_(xbatch)
                if lam is None:
                    loss = lossf(out, y_a)
                else:
                    loss = lam * lossf(out, y_a) + (1.0 - lam) * lossf(out, y_b)
                loss.backward()
                opt.step()
                if sched is not None:
                    sched.step()
            # early stopping on held-out val loss (keep best weights)
            if va is not None and (epoch + 1) >= self.min_epochs:
                self.net_.eval()
                with torch.no_grad():
                    vout = self.net_(xb_all[va].to(self.device_))
                    vloss = torch.nn.functional.cross_entropy(
                        vout, yb_all[va].to(self.device_), weight=weight,
                        label_smoothing=self.label_smoothing).item()
                if vloss < best_val - 1e-4:
                    best_val, bad = vloss, 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.net_.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.net_.load_state_dict(best_state)

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if self.device == "auto":
            self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_ = self.device

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)        # map labels -> 0..K-1

        # per-channel standardization (train statistics)
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-6
        Xn = ((X - self.mean_) / self.std_).astype(np.float32)

        self.net_ = self._build(X.shape[1], n_classes).to(self.device_)

        xb_all = torch.tensor(Xn)                        # (N, C, T) on CPU
        yb_all = torch.tensor(y_idx, dtype=torch.long)

        # optional stratified validation split for early stopping (opt-in)
        va = None
        if self.early_stopping and self.val_frac and self.val_frac > 0:
            try:
                from sklearn.model_selection import train_test_split
                tr, va = train_test_split(np.arange(len(y_idx)),
                                          test_size=self.val_frac, stratify=y_idx,
                                          random_state=self.seed)
            except Exception:                            # too few per class -> skip
                tr, va = np.arange(len(y_idx)), None
        else:
            tr = np.arange(len(y_idx))
        xb, yb = xb_all[tr], yb_all[tr]

        if self.class_weight:
            counts = np.bincount(yb.numpy(), minlength=n_classes).astype(np.float32)
            w = counts.sum() / (n_classes * np.maximum(counts, 1.0))
            weight = torch.tensor(w, dtype=torch.float32, device=self.device_)
        else:
            weight = None
        lossf = torch.nn.CrossEntropyLoss(weight=weight,
                                          label_smoothing=self.label_smoothing)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)

        n = len(yb)
        steps_per_epoch = max(1, (n + self.batch_size - 1) // self.batch_size)
        total_steps = steps_per_epoch * self.n_epochs
        if self.lr_schedule == "cosine":
            warmup = max(1, int(self.warmup_frac * total_steps))

            def lr_lambda(step):
                if step < warmup:
                    return (step + 1) / warmup
                prog = (step - warmup) / max(1, total_steps - warmup)
                return float(0.5 * (1.0 + np.cos(np.pi * min(prog, 1.0))))

            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        else:
            sched = None

        self._run_training(xb, yb, xb_all, yb_all, va, weight, lossf,
                           opt, sched, self.n_epochs, augment=True)
        return self

    def finetune(self, X, y, lr=1e-4, n_epochs=30, freeze_trunk=False,
                 restandardize=True, augment=False):
        """Calibrate a PRETRAINED model on a small target-subject dataset.
        Requires fit() first (Tier-2 cross-subject transfer):
          - freeze_trunk=False -> fine-tune ALL weights (flavor A)
          - freeze_trunk=True  -> train only the input adapter (spatial) + pool
            query + classifier head, trunk frozen (flavor C)
        Re-standardizes per-channel on the target by default (a parameter-free
        subject adaptation). Trains on ALL of (X, y) -- no val split, since
        calibration data is scarce. Returns self."""
        import torch
        if not hasattr(self, "net_"):
            raise RuntimeError("finetune() requires a pretrained model; call fit() first")
        unknown = set(np.unique(y).tolist()) - set(self.classes_.tolist())
        if unknown:
            raise ValueError(f"finetune labels {unknown} not in pretrained "
                             f"classes {self.classes_.tolist()}")
        y_idx = np.searchsorted(self.classes_, y)
        n_classes = len(self.classes_)

        # parameter-free subject adaptation: re-standardize on the target
        if restandardize:
            self.mean_ = X.mean(axis=(0, 2), keepdims=True)
            self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-6
        Xn = ((X - self.mean_) / self.std_).astype(np.float32)
        xb_all = torch.tensor(Xn)
        yb_all = torch.tensor(y_idx, dtype=torch.long)

        # reset any prior freeze, then optionally freeze the trunk
        for p in self.net_.parameters():
            p.requires_grad_(True)
        if freeze_trunk:
            for mod in (self.net_.proj_in, self.net_.blocks,
                        self.net_.downsample, self.net_.transformer):
                for p in mod.parameters():
                    p.requires_grad_(False)

        if self.class_weight:
            counts = np.bincount(yb_all.numpy(), minlength=n_classes).astype(np.float32)
            w = counts.sum() / (n_classes * np.maximum(counts, 1.0))
            weight = torch.tensor(w, dtype=torch.float32, device=self.device_)
        else:
            weight = None
        lossf = torch.nn.CrossEntropyLoss(weight=weight,
                                          label_smoothing=self.label_smoothing)
        params = [p for p in self.net_.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)

        n = len(yb_all)
        steps_per_epoch = max(1, (n + self.batch_size - 1) // self.batch_size)
        total_steps = steps_per_epoch * n_epochs
        if self.lr_schedule == "cosine":
            warmup = max(1, int(self.warmup_frac * total_steps))

            def lr_lambda(step):
                if step < warmup:
                    return (step + 1) / warmup
                prog = (step - warmup) / max(1, total_steps - warmup)
                return float(0.5 * (1.0 + np.cos(np.pi * min(prog, 1.0))))

            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        else:
            sched = None

        # calibrate on all target trials (va=None -> no early stopping)
        self._run_training(xb_all, yb_all, xb_all, yb_all, None, weight, lossf,
                           opt, sched, n_epochs, augment=augment)
        return self

    def predict(self, X):
        import torch
        Xn = ((X - self.mean_) / self.std_).astype(np.float32)
        self.net_.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(Xn), 256):             # chunk to bound memory
                out = self.net_(torch.tensor(Xn[s:s + 256]).to(self.device_))
                preds.append(out.argmax(1).cpu().numpy())
        return self.classes_[np.concatenate(preds)]


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_decoder(name, sfreq, beta_band=(13.0, 30.0), mu_band=(8.0, 13.0),
                  n_csp=6, **kwargs):
    name = name.lower()
    if name != "convtransformer" and kwargs:
        raise ValueError(f"extra decoder kwargs {list(kwargs)} are only supported "
                         f"for convtransformer, not '{name}'")
    if name == "csp":
        import mne
        from mne.decoding import CSP
        mne.set_log_level("ERROR")            # silence per-fold covariance logging
        return make_pipeline(
            BandPass(sfreq=sfreq, l_freq=beta_band[0], h_freq=beta_band[1]),
            CSP(n_components=n_csp, reg="ledoit_wolf", log=True, norm_trace=False),
            LinearDiscriminantAnalysis(),
        )
    if name == "bandpower":
        return make_pipeline(
            BandPowerFeatures(sfreq=sfreq, bands=(mu_band, beta_band)),
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        )
    if name == "eegnet":
        return TorchEEGNet(sfreq=sfreq)
    if name == "convtransformer":
        return TorchConvTransformer(sfreq=sfreq, **kwargs)
    raise ValueError(f"unknown decoder '{name}' "
                     f"(choose csp / bandpower / eegnet / convtransformer)")
