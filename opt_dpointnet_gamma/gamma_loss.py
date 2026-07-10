"""The gamma objective as a dpointnet **loss module**.

The loss has two differentiable terms (see Section 4 of the notebook):

  * **Gamma band-contrast** — rewards a *coordinated* rhythm. Take the population-mean firing rate,
    compute its power spectrum with a differentiable Welch estimate, and reward the fraction of power
    in the target band (default 30-50 Hz) relative to a reference. With ``spectral_ref='full'``
    (default) the reference is the average power across the *whole* spectrum, so power anywhere else
    — low-frequency drift or higher-frequency noise — is penalized; this gives a smoother, more
    monotonic training signal than the alternative ``spectral_ref='flanks'`` (which compares only to
    the immediate 20-30 / 50-60 Hz shoulders). Using the spectrum of the *population mean*
    (n_groups=1) means band power requires many cells to fluctuate together — genuine coordination,
    not isolated regular firing.

  * **Rate-distribution EMD** — keeps firing rates realistic. Push the distribution of per-cell rates
    toward a target lognormal (median ``rate_median_hz``) via the Earth-Mover's / Wasserstein-1
    distance, so cells stay sparsely active with no silent cells and no runaway high-rate cells.

Everything is built from ``tf.signal.frame`` / ``rfft``, so gradients flow through the surrogate
spikes back to the trainable weights. ``band_contrast_np`` is the matching numpy scorer used for
plots and progress metrics (same pipeline, so the reported number matches what is optimized).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from bmtk.simulator.dpointnet import register_loss_module

_EPS = 1e-8


# ── sub-group rates ──────────────────────────────────────────────────────────────────────────────
def _group_rates_tf(spikes_BTN, group_idx, dt_ms):
    """spikes [B,T,N] + group_idx [K,gs] -> per-group rate [M=B*K, T] in Hz."""
    x = tf.cast(spikes_BTN, tf.float32)
    K, gs = group_idx.shape
    xg = tf.gather(x, tf.reshape(group_idx, [-1]), axis=2)          # [B,T,K*gs]
    T = tf.shape(xg)[1]
    xg = tf.reshape(xg, [tf.shape(xg)[0], T, int(K), int(gs)])       # [B,T,K,gs]
    rate = tf.reduce_mean(xg, axis=3) * (1000.0 / dt_ms)             # [B,T,K] Hz
    rate = tf.transpose(rate, [0, 2, 1])                             # [B,K,T]
    return tf.reshape(rate, [-1, T])                                 # [B*K, T]


# ── Welch PSD (tf, differentiable) ───────────────────────────────────────────────────────────────
def _welch_psd_tf(pr_MT, dt_ms, frame_len, frame_step):
    """[M,T] -> freqs [F], power [F] (mean over M samples and over frames)."""
    frames = tf.signal.frame(pr_MT, frame_len, frame_step, axis=1)  # [M, nf, L]
    frames = frames - tf.reduce_mean(frames, axis=-1, keepdims=True)
    frames = frames * tf.signal.hann_window(frame_len, periodic=False)
    spec = tf.signal.rfft(frames)
    power = tf.math.real(spec * tf.math.conj(spec))                 # [M, nf, F]
    power = tf.reduce_mean(power, axis=[0, 1])                       # [F]
    freqs = tf.range(frame_len // 2 + 1, dtype=tf.float32) / (frame_len * dt_ms / 1000.0)
    return freqs, power


def _band_mean_tf(freqs, power, lo, hi):
    m = tf.cast((freqs >= lo) & (freqs < hi), tf.float32)
    return tf.reduce_sum(power * m) / tf.maximum(tf.reduce_sum(m), 1.0)


# ── numpy version (scoring / plots) ───────────────────────────────────────────────────────────────
def band_contrast_np(spikes_TN, *, mask=None, group_idx=None, in_band=(30., 50.),
                     flanks=((20., 30.), (50., 60.)), dt_ms=1.0, t_skip=0, spectral_ref="flanks",
                     frame_len=200, frame_step=50):   # frame_len=200 @ dt=1ms => 5 Hz bins (0,5,10,..)
    """Return (contrast in [0,1], peak_freq_Hz, freqs, power) using the SAME Welch pipeline as the
    loss, so the reported metric matches what is optimized. ``spectral_ref`` selects the reference the
    in-band power is compared against: 'full' (whole spectrum, default) or 'flanks' (the shoulders)."""
    sp = np.asarray(spikes_TN, np.float32)[t_skip:]                 # [T, N]
    idx = np.arange(sp.shape[1]) if mask is None else np.flatnonzero(mask)
    if group_idx is None:
        groups = [idx]
    else:
        groups = [np.asarray(g) for g in group_idx]
    win = np.hanning(frame_len)
    psds = []
    for g in groups:
        r = sp[:, g].mean(1) * (1000.0 / dt_ms)
        for s in range(0, len(r) - frame_len + 1, frame_step):
            seg = r[s:s + frame_len]; seg = (seg - seg.mean()) * win
            psds.append(np.abs(np.fft.rfft(seg)) ** 2)
    P = np.mean(psds, axis=0) if psds else np.zeros(frame_len // 2 + 1)
    f = np.fft.rfftfreq(frame_len, dt_ms / 1000.0)

    def bmean(lo, hi):
        sel = (f >= lo) & (f < hi)
        return P[sel].mean() if sel.any() else 0.0

    inb = bmean(*in_band)
    if spectral_ref == "full":
        ref = float(P.mean())                                        # in-band vs the whole spectrum
    else:
        ref = float(np.mean([bmean(lo, hi) for lo, hi in flanks]))   # in-band vs the flanks
    contrast = float(inb / (inb + ref + _EPS))
    span = (min(in_band[0], *[b[0] for b in flanks]), max(in_band[1], *[b[1] for b in flanks]))
    sel = (f >= span[0]) & (f <= span[1])
    peak = float(f[sel][np.argmax(P[sel])]) if sel.any() else 0.0
    return contrast, peak, f, P


@register_loss_module(module_name="GammaLoss")
class GammaLoss:
    """Config kwargs (defaults = the tutorial setup):

        in_band=[30,50], flanks=[[20,30],[50,60]]   target band + shoulders (Hz)
        spectral_ref='full'                         reference for the band-contrast: 'full' (whole
                                                    spectrum) or 'flanks' (shoulders only)
        n_groups=1                                  1 => spectrum of the population mean (coordination)
        frame_len=200, frame_step=50                Welch window / hop (ms at dt=1)
        rate_dist_weight=0.02                        weight of the rate-distribution EMD term
        rate_median_hz=10, rate_sigma=0.6            target lognormal rate distribution
        t_skip_ms=150, pop='all', seed=0
    """

    def __init__(self, rnn, in_band=(30., 50.), flanks=((20., 30.), (50., 60.)),
                 n_groups=1, frame_len=200, frame_step=50,
                 rate_dist_weight=0.02, rate_median_hz=10.0, rate_sigma=0.6, rate_max_hz=120.0,
                 emd_k=0.5, t_skip_ms=150, pop="all", spectral_ref="flanks", seed=0, **kwargs):
        self.rnn = rnn
        self.in_band = (float(in_band[0]), float(in_band[1]))
        self.flanks = [(float(a), float(b)) for a, b in flanks]
        self.spectral_ref = str(spectral_ref)
        self.dt_ms = float(rnn.dt)
        self.t_skip = int(t_skip_ms / self.dt_ms)
        self.frame_len, self.frame_step = int(frame_len), int(frame_step)

        # Rate-distribution EMD (Wasserstein-1) toward a target LOGNORMAL (median rate_median_hz,
        # spread rate_sigma): realistic sparse rates, penalizing both runaway (>100 Hz) and silent
        # cells. Graph-safe soft-CDF form: EMD = integral |softCDF_obs(r) - CDF_target(r)| dr.
        self.rate_dist_weight = float(rate_dist_weight)
        if self.rate_dist_weight > 0:
            samp = np.random.default_rng(1234).lognormal(mean=np.log(rate_median_hz), sigma=rate_sigma, size=200000)
            thr = np.linspace(0.0, rate_max_hz, 121).astype(np.float32)
            tcdf = np.array([(samp <= t).mean() for t in thr], np.float32)
            self._thr = tf.constant(thr[:, None]); self._tcdf = tf.constant(tcdf)
            self._dthr = float(thr[1] - thr[0]); self._emd_k = float(emd_k)

        # Which population's rhythm to score: 'all' (default), 'exc', or 'inh'.
        df = rnn.get_recurrent_network().get_nodes_df()
        ei = df["ei"].astype(str).values if "ei" in df.columns else np.array(["e"] * len(df))
        self.mask = {"exc": ei == "e", "inh": ei == "i", "all": np.ones(len(df), bool)}[pop]
        self.mask = np.asarray(self.mask, bool)
        idx = np.flatnonzero(self.mask)
        # fixed random partition into n_groups equal sub-groups (n_groups=1 => the whole population)
        rng = np.random.default_rng(seed)
        shuf = rng.permutation(idx)
        K = max(1, int(n_groups)); gs = max(1, len(idx) // K)
        self.group_idx_np = shuf[:K * gs].reshape(K, gs)
        self.group_idx = tf.constant(self.group_idx_np, tf.int32)

    @staticmethod
    def module():
        return "GammaLoss"

    def __call__(self, spikes, **kwargs):
        if self.t_skip > 0:
            spikes = spikes[:, self.t_skip:, :]
        # (1) gamma band-contrast on the population-mean spectrum
        pr = _group_rates_tf(spikes, self.group_idx, self.dt_ms)          # [B*K, T]
        freqs, power = _welch_psd_tf(pr, self.dt_ms, self.frame_len, self.frame_step)
        inb = _band_mean_tf(freqs, power, *self.in_band)
        if self.spectral_ref == "full":
            ref = tf.reduce_mean(power)                              # in-band vs the whole spectrum
        else:
            ref = tf.add_n([_band_mean_tf(freqs, power, lo, hi) for lo, hi in self.flanks]) / len(self.flanks)
        contrast = inb / (inb + ref + _EPS)
        total = -contrast
        # (2) rate-distribution EMD toward the target lognormal
        if self.rate_dist_weight > 0:
            cell_hz = tf.reduce_mean(tf.cast(spikes, tf.float32), axis=[0, 1]) * (1000.0 / self.dt_ms)  # [N]
            cdf_obs = tf.reduce_mean(tf.sigmoid(self._emd_k * (self._thr - cell_hz[None, :])), axis=1)   # [F]
            emd = tf.reduce_sum(tf.abs(cdf_obs - self._tcdf)) * self._dthr
            total = total + self.rate_dist_weight * emd
        return total
