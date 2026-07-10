"""Build the gamma tutorial's excitatory-inhibitory point-neuron network as a SONATA model that
BMTK's ``dpointnet`` reads directly.

The network is deliberately small and simple: 240 excitatory + 60 inhibitory GLIF point neurons
with random recurrent connectivity, plus a background input population that keeps every cell active.
``dpointnet`` *trains* the synaptic weights, so what this script sets are the **initial** weights
and the fixed cell / synapse properties.

Two design choices give the network a gamma-band timescale for training to exploit: (a) **fast**
inhibitory cells with a **fast-decaying** inhibitory synapse, and (b) a short
**inhibitory→excitatory conduction delay**. A delayed inhibitory feedback loop of this kind tends to
settle into a rhythm in the gamma range. These are ordinary knobs — see the constants below and
Section 7 of the notebook.

Run (from this tutorial's directory, in a ``dpointnet`` environment):
    python build_net.py
or import and call ``build(out_dir, ...)``.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

# ── network sizes ──────────────────────────────────────────────────────────────────────────────
N_E = 240              # excitatory neurons  ) 300 total, a 4:1 E:I ratio (as in cortex)
N_I = 60               # inhibitory neurons  )
N_DRIVE = 100          # external drive input units (unused here; see W_DRIVE below)
N_BKG = 100            # background-noise units (dpointnet generates their Poisson spikes internally)

# ── initial weight gains ─────────────────────────────────────────────────────────────────────────
# dpointnet divides each synaptic weight by (V_th - E_L) = 20 mV, and each cell here has only
# ~50-100 synapses (a small network), so the per-synapse weights are relatively large — the
# untrained network already fires. Training then reshapes these weights into a rhythm. All three
# gains are free knobs.
DRIVE_GAIN = 25.0      # (drive pathway is disabled here — see W_DRIVE)
REC_GAIN = 250.0       # recurrent coupling: strong enough that each spike matters to its partners,
                       # so the network can coordinate at low per-cell firing rates.
BKG_GAIN = 8.0         # background is the only drive here; a high background rate (config: 2000 Hz)
                       # makes it a smooth input rather than a noisy one.

# ── recurrent connectivity (random Bernoulli) ─────────────────────────────────────────────────
P_EE, P_EI, P_IE, P_II = 0.15, 0.15, 0.60, 0.60   # I->* at 0.60 gives ~36 inhibitory inputs per cell
W_EE, W_EI, W_IE, W_II = 0.02, 0.02, 0.06, 0.06    # magnitudes; the SIGN (set below) fixes E vs I
# Conduction delays (ms). The inhibitory->excitatory (and I->I) delay is the key timescale: a
# definite inhibitory feedback delay is what lets the E-I loop ring rather than simply settle.
# Excitatory synapses stay fast. A small per-edge jitter (DELAY_JITTER) adds heterogeneity.
DELAY_E, DELAY_IE = 1.5, 4.0
DELAY_JITTER = 1.5     # +/- ms uniform jitter per edge (so the I->E delay spans ~2.5-5.5 ms)

# ── input connectivity ─────────────────────────────────────────────────────────────────────────
# W_DRIVE=0: the external "drive" pathway is disabled for the gamma tutorial (it carries the
# auditory input in Tutorial 2; the gamma network is driven by the background alone). The drive
# population/edges are still built, for parity with Tutorial 2, but contribute nothing here.
P_DRIVE, W_DRIVE, DELAY_DRIVE = 0.5, 0.0, 1.5
P_BKG, W_BKG, DELAY_BKG = 1.0, 0.05, 1.5           # P_BKG=1.0: every cell gets background => all active

# ── GLIF point-neuron cell parameters (dpointnet format) ─────────────────────────────────────────
# tau_m = C_m / g. Pyramidal-like E cells: slower membrane (~17 ms). PV-like inhibitory cells:
# FAST membrane (~4 ms), brief refractory, higher threshold — the fast-spiking interneurons that
# pace the rhythm. Adaptation is off (asc_amps=0) for a clean sustained oscillation. build() adds
# heterogeneity by jittering each base cell into N_E_TYPES / N_I_TYPES variants.
CELL_E = dict(V_th=-50.0, g=13.0, E_L=-70.0, C_m=220.0, V_reset=-65.0, t_ref=3.0,
              asc_decay=[0.003, 0.03], asc_amps=[0.0, 0.0])   # pyramidal-like, tau_m ~ 17 ms
CELL_I = dict(V_th=-47.0, g=20.0, E_L=-70.0, C_m=80.0,  V_reset=-65.0, t_ref=1.0,
              asc_decay=[0.003, 0.03], asc_amps=[0.0, 0.0])   # PV-like (fast-spiking), tau_m ~ 4 ms
N_E_TYPES, N_I_TYPES = 4, 4    # number of jittered variants of each base cell
CELL_JITTER = 0.15             # +/- fractional jitter on C_m, g, t_ref, V_th

# ── synaptic kinetics ────────────────────────────────────────────────────────────────────────────
# dpointnet reads each synapse's `basis_weights` (a weighting of the fixed `tau_basis` timescales in
# the config) straight from its JSON. These kernels are fits to recorded synaptic currents.
TAU_BASIS = [0.5, 2.46621207433047, 12.164403991146798, 60.0]
BASIS_EXC = [0.4978443360372444, 0.7669305776661263, 0.3457013392585465, -0.02210257926138741]
# Excitatory input ONTO inhibitory cells — faster-peaking than the generic E kernel (weighted toward
# the fast 0.5/2.5 ms timescales). Fast E->I recruitment tightens the E-I loop.
BASIS_E2I = [0.90, 0.50, 0.40, 0.0]
# The inhibitory kernel sets the rhythm frequency: faster inhibitory decay -> higher frequency.
# This ~12 ms-weighted kernel gives a ~40 Hz resonance; BASIS_INH_FAST is faster (~80 Hz).
BASIS_INH = [0.1, 0.3, 1.0, 0.0]          # default: ~40 Hz
BASIS_INH_FAST = [0.2, 1.2, 0.15, 0.0]    # faster inhibition -> ~80 Hz high gamma
RECEPTOR_EXC, RECEPTOR_INH = 3, 6

# component dynamics_params filenames
SYN_EXC_JSON = "exc.json"       # E->E (and drive/bkg) excitatory synapses
SYN_E2I_JSON = "e2i.json"       # E->I excitatory synapses (fast-peaking, onto inhibitory cells)
SYN_INH_JSON = "inh.json"       # inhibitory-source synapses
CELL_E_JSON = "cell_exc.json"
CELL_I_JSON = "cell_inh.json"


def _rng(seed):
    return np.random.default_rng(seed)


def _edge_set(n_src, n_trg, p, rng, *, no_autapse=False):
    """Precompute a random Bernoulli(p) edge list as a set of (src_id, trg_id) node-id pairs
    (deterministic given the seed, unlike a stochastic connection_rule whose result depends on
    BMTK's pair-iteration order)."""
    mask = rng.random((n_src, n_trg)) < p
    if no_autapse and n_src == n_trg:
        np.fill_diagonal(mask, False)
    src, trg = np.where(mask)
    return set(zip(src.tolist(), trg.tolist()))


def _explicit_rule(edge_set, src_off, trg_off):
    """BMTK connection_rule(source, target) -> 0/1 from a precomputed set. Offsets map a
    population-local pair back to the (src_off+i, trg_off+j) ids the set was built with."""
    def rule(source, target):
        return 1 if (source.node_id - src_off, target.node_id - trg_off) in edge_set else 0
    return rule


def _const(value):
    def rule(source, target):
        return value
    return rule


def build(out_dir: str, *, seed: int = 0, force: bool = True, basis_inh=None) -> str:
    """Build the SONATA network under ``out_dir`` (contains network/ + components/). Returns out_dir.

    ``basis_inh`` overrides the inhibitory synapse's 4-value basis_weights (e.g. BASIS_INH_FAST for
    faster inhibition, a higher rhythm frequency); defaults to BASIS_INH.
    """
    from bmtk.builder import NetworkBuilder
    basis_inh = BASIS_INH if basis_inh is None else list(basis_inh)

    net_path = os.path.join(out_dir, "network")
    comp_cell = os.path.join(out_dir, "components", "cell_models")
    comp_syn = os.path.join(out_dir, "components", "synaptic_models")
    if force and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    for d in (net_path, comp_cell, comp_syn):
        os.makedirs(d, exist_ok=True)

    rng = _rng(seed)

    # ---- heterogeneous cell variants (jittered inhibitory / excitatory cells) -----------------
    def _variants(base, n_types, prefix):
        """n_types jittered copies of a base cell dict; returns [(json_name, params), ...]."""
        out = []
        for k in range(n_types):
            p = dict(base)
            for key in ("C_m", "g", "t_ref", "V_th"):
                p[key] = float(base[key] * (1.0 + rng.uniform(-CELL_JITTER, CELL_JITTER)))
            out.append((f"{prefix}_{k}.json", p))
        return out

    e_variants = _variants(CELL_E, N_E_TYPES, "cell_exc")
    i_variants = _variants(CELL_I, N_I_TYPES, "cell_inh")

    # ---- recurrent E-I network (point_neuron) -------------------------------------------------
    # E node_ids are 0..N_E-1 (all E variants added first), I are N_E..N_E+N_I-1 (BMTK preserves add
    # order; dpointnet keeps native SONATA order), so the ei-based edge selectors/offsets hold.
    stg = NetworkBuilder("stg")
    for k, (jname, _) in enumerate(e_variants):
        n = N_E // N_E_TYPES + (1 if k < N_E % N_E_TYPES else 0)
        stg.add_nodes(N=n, ei="e", pop_name=f"exc{k}", model_type="point_neuron",
                      model_template="glif_lif_asc", dynamics_params=jname)
    for k, (jname, _) in enumerate(i_variants):
        n = N_I // N_I_TYPES + (1 if k < N_I % N_I_TYPES else 0)
        stg.add_nodes(N=n, ei="i", pop_name=f"inh{k}", model_type="point_neuron",
                      model_template="glif_lif_asc", dynamics_params=jname)

    def _delay_rule(base):
        def rule(source, target):
            return float(max(1.0, base + rng.uniform(-DELAY_JITTER, DELAY_JITTER)))
        return rule

    def _wjit_rule(w):  # per-edge weight jitter (heterogeneity), sign preserved
        def rule(source, target):
            return float(w * (1.0 + rng.uniform(-0.2, 0.2)))
        return rule

    def _add_rec(pre_ei, post_ei, p, w, no_auto):
        pre_n, post_n = (N_E if pre_ei == "e" else N_I), (N_E if post_ei == "e" else N_I)
        pre_off, post_off = (0 if pre_ei == "e" else N_E), (0 if post_ei == "e" else N_E)
        eset = _edge_set(pre_n, post_n, p, rng, no_autapse=no_auto and pre_ei == post_ei)
        if not eset:
            return
        # SIGN fixes each synapse's type: E sources start positive, I sources negative. dpointnet's
        # SignedConstraint then holds that sign fixed through training (E stays E, I stays I).
        sign = 1.0 if pre_ei == "e" else -1.0
        if pre_ei == "e":
            syn = SYN_E2I_JSON if post_ei == "i" else SYN_EXC_JSON   # fast E->I recruitment
        else:
            syn = SYN_INH_JSON
        base_delay = DELAY_IE if pre_ei == "i" else DELAY_E     # I->* carries the feedback delay
        cm = stg.add_edges(
            source={"ei": pre_ei}, target={"ei": post_ei},
            connection_rule=_explicit_rule(eset, pre_off, post_off),
            dynamics_params=syn, model_template="static_synapse",
        )
        cm.add_properties("syn_weight", rule=_wjit_rule(sign * w * REC_GAIN), dtypes=np.float32)
        cm.add_properties("delay", rule=_delay_rule(base_delay), dtypes=np.float32)

    _add_rec("e", "e", P_EE, W_EE, no_auto=True)
    _add_rec("e", "i", P_EI, W_EI, no_auto=False)
    _add_rec("i", "e", P_IE, W_IE, no_auto=False)
    _add_rec("i", "i", P_II, W_II, no_auto=True)

    # ---- external drive + background (virtual) ------------------------------------------------
    def _add_virtual(name, n_src, p, w, delay):
        vnet = NetworkBuilder(name)
        vnet.add_nodes(N=n_src, ei="e", model_type="virtual")
        for post_ei in ("e", "i"):
            post_n, post_off = (N_E if post_ei == "e" else N_I), (0 if post_ei == "e" else N_E)
            eset = _edge_set(n_src, post_n, p, rng)
            if not eset:
                continue
            cm = vnet.add_edges(
                source=vnet.nodes(), target=stg.nodes(ei=post_ei),
                connection_rule=_explicit_rule(eset, 0, post_off),
                delay=delay, dynamics_params=SYN_EXC_JSON, model_template="static_synapse",
            )
            cm.add_properties("syn_weight", rule=_const(w), dtypes=np.float32)
        return vnet

    drive = _add_virtual("drive", N_DRIVE, P_DRIVE, W_DRIVE * DRIVE_GAIN, DELAY_DRIVE)
    bkg = _add_virtual("bkg", N_BKG, P_BKG, W_BKG * BKG_GAIN, DELAY_BKG)

    # ---- save SONATA (NetworkBuilder emits <name>_nodes.h5 / _node_types.csv / edges files) ----
    for net in (stg, drive, bkg):
        net.build()
        net.save(net_path)

    # ---- component JSONs ----------------------------------------------------------------------
    for jname, params in e_variants + i_variants:
        _write_json(os.path.join(comp_cell, jname), params)
    _write_json(os.path.join(comp_syn, SYN_EXC_JSON),
                {"receptor_type": RECEPTOR_EXC, "basis_weights": BASIS_EXC})
    _write_json(os.path.join(comp_syn, SYN_E2I_JSON),
                {"receptor_type": RECEPTOR_EXC, "basis_weights": BASIS_E2I})
    _write_json(os.path.join(comp_syn, SYN_INH_JSON),
                {"receptor_type": RECEPTOR_INH, "basis_weights": basis_inh})

    print(f"[gamma/dpn] built SONATA net_dir: {out_dir}  ({N_E} E + {N_I} I, "
          f"{N_E_TYPES}E/{N_I_TYPES}I cell variants, I->E delay {DELAY_IE}ms, drive={N_DRIVE}, bkg={N_BKG})")
    return out_dir


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "net")
    build(out, force=True)
