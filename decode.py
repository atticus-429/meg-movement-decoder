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
                 device="auto", max_seq_len=2048, class_weight=True, seed=0):
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

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
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

        xb = torch.tensor(Xn)                            # (N, C, T)
        yb = torch.tensor(y_idx, dtype=torch.long)

        if self.class_weight:
            counts = np.bincount(y_idx, minlength=n_classes).astype(np.float32)
            w = counts.sum() / (n_classes * np.maximum(counts, 1.0))
            weight = torch.tensor(w, dtype=torch.float32, device=self.device_)
        else:
            weight = None
        lossf = torch.nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)

        n = len(y_idx)
        rng = np.random.default_rng(self.seed)
        self.net_.train()
        for _ in range(self.n_epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                opt.zero_grad()
                out = self.net_(xb[idx].to(self.device_))
                loss = lossf(out, yb[idx].to(self.device_))
                loss.backward()
                opt.step()
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
                  n_csp=6):
    name = name.lower()
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
        return TorchConvTransformer(sfreq=sfreq)
    raise ValueError(f"unknown decoder '{name}' "
                     f"(choose csp / bandpower / eegnet / convtransformer)")
