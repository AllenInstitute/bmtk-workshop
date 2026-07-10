# dpointnet gamma tutorial

Train a small excitatory–inhibitory GLIF point-neuron network, **by gradient descent through its
spikes**, to produce a coordinated **gamma-band (~30–50 Hz) rhythm** — using BMTK's differentiable
point-neuron simulator `dpointnet`. The focus of the tutorial is the **loss function**: gamma is a
*population* phenomenon, so the objective rewards genuine coordination across cells (a spectral
band-contrast on the population-mean firing rate) while an Earth-Mover's term keeps per-cell firing
rates realistic.

## Contents

- `dpointnet_gamma_tutorial.ipynb` — the tutorial: build → run untrained → define the loss → train →
  results, followed by optional open-ended problems to explore.
- `dpointnet_gamma_tutorial_solutions.ipynb` — worked versions of those optional problems.
- `build_net.py` — builds the SONATA network (240 E + 60 I + background/drive virtual populations).
- `gamma_loss.py` — the custom `GammaLoss` module (band-contrast + rate-EMD) plus a matching numpy scorer.
- `gamma_viz.py` — small raster / population-rate plotting helpers.
- `configs/config.train.json` — the `dpointnet` training config.

The network and all outputs are generated at run time (`build_net.py` writes `net/`; the notebook
writes `saved/`), so no large files are committed.

## Run

In a `dpointnet`-capable environment (BMTK with the `dpointnet` simulator + TensorFlow-GPU), from this
directory open `dpointnet_gamma_tutorial.ipynb`, or run it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace dpointnet_gamma_tutorial.ipynb
```

Build + 30 training epochs is a few minutes on a single GPU.
