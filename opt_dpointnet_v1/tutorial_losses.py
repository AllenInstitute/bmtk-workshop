from pathlib import Path
import importlib

import numpy as np
import tensorflow as tf

from bmtk.simulator import dpointnet
from bmtk.simulator.dpointnet.loss_functions import loss_utils

import workshop_dpointnet as ws


class OverallSpikeRateDistributionTarget:
    def __init__(
        self,
        rnn,
        rate_cost=10000.0,
        stimulus_type="drifting_gratings",
        pre_delay=0.0,
        post_delay=0.0,
        neuropixels_df=None,
        seed=42,
        dtype=tf.float32,
        **kwargs,
    ):
        target_rates_hz = ws.load_neuropixels_rates(
            neuropixels_df,
            stimulus_type=stimulus_type,
        )["firing_rate_hz"].dropna().to_numpy(dtype=np.float32)
        target_rates = loss_utils.sample_firing_rates(
            np.append(target_rates_hz / 1000.0, 0.0),
            rnn.recurrent_network["n_nodes"],
            seed,
        )
        self.target_rates = tf.constant(np.sort(target_rates), dtype=dtype)
        self.rate_cost = tf.constant(float(rate_cost), dtype=dtype)
        self.pre_delay = int(pre_delay or 0)
        self.post_delay = int(post_delay or 0)
        self.dtype = dtype

    @staticmethod
    def module():
        return "OverallSpikeRateDistributionTarget"

    def __call__(self, spikes, trim=True, **kwargs):
        spikes = loss_utils.spike_trimming(
            spikes,
            pre_delay=self.pre_delay,
            post_delay=self.post_delay,
            trim=trim,
        )
        spikes = tf.cast(spikes, self.dtype)
        model_rates = tf.reduce_mean(spikes, axis=tf.range(tf.rank(spikes) - 1))
        loss = loss_utils.compute_spike_rate_distribution_loss(
            model_rates,
            self.target_rates,
            dtype=self.dtype,
        )
        return self.rate_cost * tf.reduce_mean(loss)


class L5ZAxisBiasLoss:
    def __init__(self, rnn, cost=1.0, dgsign=1.0, data_dir="GLIF_network", layer="L5", z_scale=400.0, **kwargs):
        metadata = ws.load_v1_metadata(Path(data_dir) / "network").set_index("node_id")
        tf_id_map = importlib.import_module("bmtk.simulator.dpointnet.id_maps").TFIDMap()
        node_ids = tf_id_map.tf2bmtk_id_map().loc[np.arange(rnn.recurrent_network["n_nodes"]), "node_id"].to_numpy()
        metadata = metadata.loc[node_ids].reset_index(drop=True)

        selected = metadata["layer"].to_numpy() == layer
        z_weight = metadata.loc[selected, "z"].to_numpy(dtype=np.float32) / float(z_scale)
        self.mask = tf.constant(selected, dtype=tf.bool)
        self.spatial_weight = tf.constant(float(dgsign) * z_weight, dtype=tf.float32)
        self.cost = tf.constant(float(cost), dtype=tf.float32)
        self.dt = tf.constant(rnn.dt, dtype=tf.float32)

    @staticmethod
    def module():
        return "L5ZAxisBiasLoss"

    def __call__(self, spikes, **kwargs):
        l5_spikes = tf.boolean_mask(spikes, self.mask, axis=2)
        firing_rates = tf.reduce_mean(tf.cast(l5_spikes, tf.float32), axis=[0, 1]) * (1000.0 / self.dt)
        return -self.cost * tf.reduce_mean(firing_rates * self.spatial_weight)


def register_tutorial_losses():
    dpointnet.add_loss_module(OverallSpikeRateDistributionTarget, overwrite=True)
    dpointnet.add_loss_module(L5ZAxisBiasLoss, overwrite=True)


register_tutorial_losses()