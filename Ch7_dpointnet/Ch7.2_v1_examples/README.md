# dpointnet workshop tutorial

This repository contains the lightweight workshop materials for the dpointnet tutorial:

- Jupyter notebooks
- Python helper scripts
- SONATA/BMTK JSON configs
- Small target-data files used by the tutorial losses
- GLIF model component metadata
- Small figures and CSV summaries

Large network, cache, training-output, and inference-output files are intentionally not included. In particular, this repo excludes folders such as `GLIF_network/network/`, `GLIF_network_l4_cutout/`, `lgn_cache/`, `output_*`, and `training_callbacks_*`.

To run the notebooks end-to-end, copy or mount the full V1 network into `GLIF_network/network/`. The notebook can then generate the L4 cutout network locally from that downloaded network.
