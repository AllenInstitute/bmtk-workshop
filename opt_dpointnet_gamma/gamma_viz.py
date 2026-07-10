"""Small, self-contained plotting helpers for the gamma tutorial.

Each takes numpy spikes ``(T, N)`` for one rollout plus the boolean E/I mask (``is_exc``) and draws
onto a matplotlib axis. Excitatory cells are red, inhibitory blue — the usual E/I convention. The
notebook computes its own spectrum / rate distribution inline; these helpers are just the raster and
the population-rate trace, so there's no dependency on the loss module.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

E_COLOR, I_COLOR = "#c1272d", "#0072b2"   # red exc, blue inh


def raster(spikes_TN, is_exc, *, ax=None, t0=0, t1=None):
    """Spike raster with excitatory rows on top (red) and inhibitory rows below (blue)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    sp = np.asarray(spikes_TN)
    t1 = t1 or sp.shape[0]
    order = np.concatenate([np.where(is_exc)[0], np.where(~is_exc)[0]])  # E rows first
    row_of = np.empty(sp.shape[1], int); row_of[order] = np.arange(sp.shape[1])
    is_exc_row = is_exc[order]
    tt, nn = np.where(sp[t0:t1] > 0)
    rows = row_of[nn]
    ec = is_exc_row[rows]
    ax.scatter(tt[ec] + t0, rows[ec], s=1.5, c=E_COLOR, marker="|", linewidths=0.5)
    ax.scatter(tt[~ec] + t0, rows[~ec], s=1.5, c=I_COLOR, marker="|", linewidths=0.5)
    ax.axhline(is_exc.sum(), color="k", lw=0.5, ls=":")   # E/I divider
    ax.set(xlabel="time (ms)", ylabel="neuron (I top / E bottom)", xlim=(t0, t1))
    return ax


def pop_rate(spikes_TN, is_exc, *, ax=None, dt_ms=1.0, smooth_ms=3):
    """Population firing-rate trace (Hz), excitatory and inhibitory populations separately."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))
    sp = np.asarray(spikes_TN, float)
    t = np.arange(sp.shape[0]) * dt_ms
    k = np.ones(max(1, int(smooth_ms))) / max(1, int(smooth_ms))
    for mask, c, lab in [(is_exc, E_COLOR, "E"), (~is_exc, I_COLOR, "I")]:
        r = sp[:, mask].mean(1) * 1000.0 / dt_ms
        ax.plot(t, np.convolve(r, k, "same"), c=c, lw=0.9, label=lab)
    ax.set(xlabel="time (ms)", ylabel="pop. rate (Hz)"); ax.legend(loc="upper right", fontsize=8)
    return ax
