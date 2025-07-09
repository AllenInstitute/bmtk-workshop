import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
import seaborn as sns
from . import other_v1_utils, toolkit
import shutil

# Set style parameters for publication quality
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.labelpad': 8,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'legend.frameon': True,
    'legend.fontsize': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': True
})

sns.set(style="ticks")
plt.rcParams['text.usetex'] = shutil.which('latex') is not None

class InputActivityFigure:
    def __init__(
        self,
        network,
        data_dir,
        images_dir="Images",
        filename="Raster_plot",
        batch_ind=0,
        scale=3.0,
        frequency=2,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        reverse=False,
        plot_core_only=True,
        core_radius=400,
    ):
        self.figure = plt.figure(
            figsize=toolkit.cm2inch((15 * scale, 11 * scale)))
        gs = self.figure.add_gridspec(10, 1)
        self.input_ax = self.figure.add_subplot(gs[:3])
        self.activity_ax = self.figure.add_subplot(gs[3:-1])
        self.drifting_grating_ax = self.figure.add_subplot(gs[-1])

        self.inputs_plot = RasterPlot(
            batch_ind=batch_ind,
            stimuli_init_time=stimuli_init_time,
            stimuli_end_time=stimuli_end_time,
            scale=scale,
            y_label="LGN Neuron ID",
            alpha=0.05,
        )
        self.laminar_plot = LaminarPlot(
            network,
            data_dir,
            batch_ind=batch_ind,
            stimuli_init_time=stimuli_init_time,
            stimuli_end_time=stimuli_end_time,
            scale=scale,
            alpha=0.4,
            plot_core_only=plot_core_only,
            core_radius=core_radius,
        )
        self.drifting_grating_plot = DriftingGrating(
            frequency=frequency,
            stimuli_init_time=stimuli_init_time,
            stimuli_end_time=stimuli_end_time,
            reverse=reverse,
            scale=scale,
        )

        self.tightened = True  # False
        self.scale = scale
        self.n_neurons = network["n_nodes"]
        self.batch_ind = batch_ind
        self.images_dir = images_dir
        self.filename = filename

    def __call__(self, inputs, spikes):
        self.input_ax.clear()
        self.activity_ax.clear()
        self.drifting_grating_ax.clear()

        self.inputs_plot(self.input_ax, inputs)
        self.input_ax.set_xticklabels([])
        toolkit.apply_style(self.input_ax, scale=self.scale)

        self.laminar_plot(self.activity_ax, spikes)
        self.activity_ax.set_xticklabels([])
        toolkit.apply_style(self.activity_ax, scale=self.scale)

        simulation_length = spikes.shape[1]
        self.drifting_grating_plot(self.drifting_grating_ax, simulation_length)
        toolkit.apply_style(self.drifting_grating_ax, scale=self.scale)

        if not self.tightened:
            self.figure.tight_layout()
            self.tightened = True

        self.figure.savefig(
            os.path.join(self.images_dir, self.filename), dpi=300, transparent=False
        )

        plt.close(self.figure)

        # return self.figure


class InputActivityFigureWithoutStimulus:
    def __init__(
        self,
        network,
        data_dir,
        images_dir="Images",
        filename="Raster_plot",
        batch_ind=0,
        scale=3.0,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        plot_core_only=True,
        core_radius=400,
    ):
        self.figure = plt.figure(
            figsize=toolkit.cm2inch((15 * scale, 11 * scale)))
        gs = self.figure.add_gridspec(10, 1)
        self.input_ax = self.figure.add_subplot(gs[:3])
        self.activity_ax = self.figure.add_subplot(gs[3:])

        self.inputs_plot = RasterPlot(
            batch_ind=batch_ind,
            stimuli_init_time=500,
            stimuli_end_time=1500,
            scale=scale,
            y_label="LGN Neuron ID",
            alpha=0.05,
        )
        self.laminar_plot = LaminarPlot(
            network,
            data_dir,
            batch_ind=batch_ind,
            stimuli_init_time=500,
            stimuli_end_time=1500,
            scale=scale,
            alpha=0.2,
            plot_core_only=plot_core_only,
            core_radius=core_radius,
        )

        self.tightened = True  # False
        self.scale = scale
        self.n_neurons = network["n_nodes"]
        self.batch_ind = batch_ind
        self.images_dir = images_dir
        self.filename = filename

    def __call__(self, inputs, spikes):
        self.input_ax.clear()
        self.activity_ax.clear()

        self.inputs_plot(self.input_ax, inputs)
        self.input_ax.set_xticklabels([])
        toolkit.apply_style(self.input_ax, scale=self.scale)

        self.laminar_plot(self.activity_ax, spikes)
        self.activity_ax.set_xticklabels([])
        toolkit.apply_style(self.activity_ax, scale=self.scale)

        # self.drifting_grating_plot(self.drifting_grating_ax, spikes)
        # toolkit.apply_style(self.drifting_grating_ax, scale=self.scale)

        if not self.tightened:
            self.figure.tight_layout()
            self.tightened = True

        self.figure.savefig(
            os.path.join(self.images_dir, self.filename), dpi=300, transparent=False
        )

        # return self.figure
        plt.close(self.figure)


def pop_ordering(pop_name):
    layer_order = 2 if '23' in pop_name[1:3] else int(pop_name[1:2])
    
    if pop_name[0] == "e":
        inter_order = 4 
    elif pop_name.count("Vip") or pop_name.count("Htr3a") > 0:
        inter_order = 1
    elif pop_name.count("Sst") > 0:
        inter_order = 2
    elif pop_name.count("Pvalb") > 0:
        inter_order = 3
    else:
        print(pop_name)
        raise ValueError()

    return layer_order * 10 + inter_order

def model_name_to_cell_type(pop_name):
    # Convert pop_name in the old format to cell types. E.g., 'e4Rorb' -> 'L4 Exc', 'i4Pvalb' -> 'L4 PV', 'i23Sst' -> 'L2/3 SST'
    shift = 0  # letter shift for L23
    ei = pop_name[0]
    layer = pop_name[1]
    if layer == "2":
        layer = "23"
        shift = 1
    elif layer == "1":
        return "i1Htr3a"  # special case
    class_name = pop_name[2 + shift :]
    if ei == 'e':  # excitatory
        class_name = ""
    elif (class_name == "Vip") or (class_name == "Htr3a"):
        class_name = "Vip"

    return f"{ei}{layer}{class_name}"


class RasterPlot:
    def __init__(
        self,
        batch_ind=0,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        scale=2.0,
        marker_size=1.0,
        alpha=0.03,
        color="r",
        y_label="Neuron ID",
    ):
        self.batch_ind = batch_ind
        self.stimuli_init_time = stimuli_init_time
        self.stimuli_end_time = stimuli_end_time
        self.scale = scale
        self.marker_size = marker_size
        self.alpha = alpha
        self.color = color
        self.y_label = y_label

    def __call__(self, ax, spikes):
        # This method plots the spike train (spikes) that enters the network
        n_elements = np.prod(spikes.shape)
        non_binary_frac = (
            np.sum(np.logical_and(spikes > 1e-3, spikes < 1 - 1e-3)) / n_elements
        )
        if non_binary_frac > 0.01:
            rate = -np.log(1 - spikes[self.batch_ind] / 1.3) * 1000
            # rate = rate.reshape((rate.shape[0], int(rate.shape[1] / 100), 100)).mean(-1)
            p = ax.pcolormesh(rate.T, cmap="cividis")
            toolkit.do_inset_colorbar(ax, p, "")
            ax.set_ylim([0, rate.shape[-1]])
            ax.set_yticks([0, rate.shape[-1]])
            # ax.set_yticklabels([0, rate.shape[-1] * 100])
            ax.set_yticklabels([0, rate.shape[-1]])
            ax.set_ylabel(self.y_label, fontsize=20)
        else:
            # Take the times where the spikes occur
            times, ids = np.where(
                spikes[self.batch_ind].astype(float) > 0.5)
            ax.plot(
                times, ids, ".", color=self.color, ms=self.marker_size, alpha=self.alpha
            )
            ax.set_ylim([0, spikes.shape[-1]])
            ax.set_yticks([0, spikes.shape[-1]])
            ax.set_ylabel(self.y_label, fontsize=24)

        ax.axvline(
            self.stimuli_init_time,
            linestyle="dashed",
            color="k",
            linewidth=1.5,
            alpha=1,
        )
        ax.axvline(
            self.stimuli_end_time, linestyle="dashed", color="k", linewidth=1.5, alpha=1
        )
        ax.set_xlim([0, spikes.shape[1]])
        ax.set_xticks([0, spikes.shape[1]])
        ax.tick_params(axis="both", which="major", labelsize=20)


class LaminarPlot:
    def __init__(
        self,
        network,
        data_dir,
        batch_ind=0,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        scale=2.0,
        marker_size=1.0,
        alpha=0.2,
        plot_core_only=True,
        core_radius=400,
    ):
        self.batch_ind = batch_ind
        self.stimuli_init_time = stimuli_init_time
        self.stimuli_end_time = stimuli_end_time
        self.scale = scale
        self.marker_size = marker_size
        self.alpha = alpha
        self.data_dir = data_dir
        self.network = network
        self.n_neurons = network["n_nodes"]
        # self.core_neurons = 65871

        if plot_core_only:
            # if self.n_neurons > self.core_neurons:
                # self.n_neurons = self.core_neurons
            # core_neurons = 16679 #65871 
            # core_radius = 400 #200
            self.core_mask = other_v1_utils.isolate_core_neurons(
                self.network, radius=core_radius, data_dir=self.data_dir
            )
            core_neurons = np.sum(self.core_mask)
            self.n_neurons = core_neurons
        else:
            self.core_mask = np.full(self.n_neurons, True)

         # use the true_pop_names, true_node_type_ids to create a dictionary with the node_type_id as key and the pop_name as value
        # since many of them are repeated we can use the unique function to get the unique pop_names

        node_types = pd.read_csv(os.path.join(self.data_dir, "network/v1_node_types.csv"), sep=" ")
        path_to_h5 = os.path.join(self.data_dir, "network/v1_nodes.h5")
        with h5py.File(path_to_h5, mode='r') as node_h5:
            # Create mapping from node_type_id to pop_name
            node_types.set_index('node_type_id', inplace=True)
            node_type_id_to_pop_name = node_types['pop_name'].to_dict()

            # Map node_type_id to pop_name for all neurons and select population names of neurons in the present network 
            node_type_ids = node_h5['nodes']['v1']['node_type_id'][()][network['tf_id_to_bmtk_id']]
            true_pop_names = np.array([node_type_id_to_pop_name[nid] for nid in node_type_ids])

            # Select population names of neurons in the present network (core)
            true_pop_names = true_pop_names[self.core_mask]
            true_node_type_ids = node_type_ids[self.core_mask]
    
        # Now order the pop_names
        #  according to their layer and type
        pop_orders = dict(sorted(node_type_id_to_pop_name.items(), key=lambda item: pop_ordering(item[1])))
        reversed_pop_orders = {model_name_to_cell_type(v): [] for k, v in pop_orders.items()}
        for k, v in pop_orders.items():
            reversed_pop_orders[model_name_to_cell_type(v)].append(k)

        # Now we convert the neuron id (related to its pop_name) to an index related to its position in the y axis
        neuron_id_to_y = np.zeros(self.n_neurons, np.int32) - 1  # rest 1 to check at the end if every neuron has an index
        current_ind = 0

        self.e_mask = np.zeros(self.n_neurons, np.bool_)
        self.htr3a_mask = np.zeros(self.n_neurons, np.bool_)
        self.vip_mask = np.zeros(self.n_neurons, np.bool_)
        self.sst_mask = np.zeros(self.n_neurons, np.bool_)
        self.pvalb_mask = np.zeros(self.n_neurons, np.bool_)

        layer_bounds = []
        ie_bounds = []
        current_pop_name = "e0"

        for pop_name, pop_ids in reversed_pop_orders.items():
            # choose all the neurons of the cell_type
            sel = np.isin(true_node_type_ids, pop_ids)
            _n = np.sum(sel)
            pop_y_positions = np.arange(current_ind, current_ind + _n)
            tuning_angles = network['tuning_angle'][self.core_mask][sel]
            sorted_indices = np.argsort(tuning_angles)
            # Correctly map the sorted y positions to the selected neurons
            # Convert boolean mask 'sel' to indices
            sel_indices = np.where(sel)[0]
            # Assign y positions based on sorted order
            neuron_id_to_y[sel_indices[sorted_indices]] = pop_y_positions
            # pop_y_positions = pop_y_positions[sorted_indices]
            # order the neurons by type and tuning angle in the y axis
            # neuron_id_to_y[sel] = pop_y_positions

            if int(pop_name[1]) > int(current_pop_name[1]):
                # register the change of layer
                layer_bounds.append(current_ind)
            if current_pop_name[0] == "i" and pop_name[0] == "e":
                # register the change of neuron type: exc -> inh
                ie_bounds.append(current_ind)

            # #Now introduce the masks for the different neuron types
            if pop_name[0] == "e":
                self.e_mask = np.logical_or(self.e_mask, sel)
            elif pop_name.count("Htr3a") > 0:
                self.htr3a_mask = np.logical_or(self.htr3a_mask, sel)
            elif pop_name.count("Vip") > 0:
                self.vip_mask = np.logical_or(self.vip_mask, sel)
            elif pop_name.count("Sst") > 0:
                self.sst_mask = np.logical_or(self.sst_mask, sel)
            elif pop_name.count("Pvalb") > 0:
                self.pvalb_mask = np.logical_or(self.pvalb_mask, sel)
            else:
                raise ValueError(f"Unknown population {pop_name}")
            current_ind += _n
            current_pop_name = pop_name
        # check that an y id has been given to every neuron
        assert np.sum(neuron_id_to_y < 0) == 0
        self.layer_bounds = layer_bounds

        ######### For l5e neurons  ###########
        # l5e_min, l5e_max = ie_bounds[-2], layer_bounds[-1]
        # n_l5e = l5e_max - l5e_min

        # n_readout_pops = network['readout_neuron_ids'].shape[0]
        # dist = int(n_l5e / n_readout_pops)
        # #####################################

        y_to_neuron_id = np.zeros(self.n_neurons, np.int32)
        y_to_neuron_id[neuron_id_to_y] = np.arange(self.n_neurons)
        assert np.all(y_to_neuron_id[neuron_id_to_y] == np.arange(self.n_neurons))
        # y_to_neuron_id: E.g., la neurona séptima por orden de capas tiene id 0, y_to_neuron_id[7]=0
        # neuron_id_to_y: E.g., la neurona con id 0 es la séptima por orden de capas, neuron_id_to_y[0] = 7

        # ##### For l5e neurons #####
        # neurons_per_readout = network['readout_neuron_ids'].shape[1]

        # for i in range(n_readout_pops):
        #     desired_y = np.arange(neurons_per_readout) + \
        #         int(dist / 2) + dist * i + l5e_min
        #     for j in range(neurons_per_readout):
        #         other_id = y_to_neuron_id[desired_y[j]]
        #         readout_id = network['readout_neuron_ids'][i, j]
        #         old_readout_y = neuron_id_to_y[readout_id]
        #         neuron_id_to_y[readout_id], neuron_id_to_y[other_id] = desired_y[j], neuron_id_to_y[readout_id]
        #         y_to_neuron_id[old_readout_y], y_to_neuron_id[desired_y[j]
        #                                                       ] = other_id, readout_id
        ###########################

        # plot the L1 top and L6 bottom
        self.neuron_id_to_y = self.n_neurons - neuron_id_to_y  

    def __call__(self, ax, spikes):
        scale = self.scale
        ms = self.marker_size
        alpha = self.alpha
        seq_len = spikes.shape[1]
        layer_label = ["1", "2/3", "4", "5", "6"]
        for i, (y, h) in enumerate(zip(self.layer_bounds, np.diff(self.layer_bounds, append=[self.n_neurons]))):
            ax.annotate(
                f"L{layer_label[i]}",
                (5, (self.n_neurons - y - h / 2)),
                fontsize=6 * scale,
                color="k",
                fontweight="bold",
                va="center",
            )

            if i % 2 != 0:
                continue
            rect = patches.Rectangle(
                (0, self.n_neurons - y - h), seq_len, h, color="gray", alpha=0.1
            )
            ax.add_patch(rect)

        spikes = np.array(spikes)
        spikes = np.transpose(spikes[self.batch_ind, :, self.core_mask])

        # e
        times, ids = np.where(spikes * self.e_mask[None, :].astype(float))
        _y = self.neuron_id_to_y[ids]
        ax.plot(times, _y, ".", color="r", ms=ms, alpha=alpha)

        # Htr3a
        times, ids = np.where(
            spikes * self.htr3a_mask[None, :].astype(float))
        _y = self.neuron_id_to_y[ids]
        ax.plot(times, _y, ".", color="pink", ms=ms, alpha=alpha)

        # vip
        times, ids = np.where(spikes * self.vip_mask[None, :].astype(float))
        _y = self.neuron_id_to_y[ids]
        ax.plot(times, _y, ".", color="darkviolet", ms=ms, alpha=alpha)

        # sst
        times, ids = np.where(spikes * self.sst_mask[None, :].astype(float))
        _y = self.neuron_id_to_y[ids]
        ax.plot(times, _y, ".", color="g", ms=ms, alpha=alpha)

        # pvalb
        times, ids = np.where(
            spikes * self.pvalb_mask[None, :].astype(float))
        _y = self.neuron_id_to_y[ids]
        ax.plot(times, _y, ".", color="b", ms=ms, alpha=alpha)

        ##### For l5e neurons #####

        # for i, readout_neuron_ids in enumerate(self.network['readout_neuron_ids']):
        #     if len(self.network['readout_neuron_ids']) == 2 and i == 0:
        #         continue
        #     sel = np.zeros(self.n_neurons)
        #     sel[readout_neuron_ids] = 1.
        #     times, ids = np.where(
        #         spikes[self.batch_ind] * sel[None, :].astype(float))
        #     _y = self.neuron_id_to_y[ids]
        #     ax.plot(times, _y, '.', color='k', ms=ms, alpha=alpha)

        ###########################
        ax.plot([-1, -1], [-1, -1], ".", color="pink",
                ms=6, alpha=0.9, label="Htr3a")
        ax.plot([-1, -1], [-1, -1], ".", color="darkviolet", 
                ms=6, alpha=0.9, label="Vip")
        ax.plot([-1, -1], [-1, -1], ".", color="g",
                ms=6, alpha=0.9, label="Sst")
        ax.plot([-1, -1], [-1, -1], ".", color="b",
                ms=6, alpha=0.9, label="Pvalb")
        ax.plot([-1, -1], [-1, -1], ".", color="r",
                ms=6, alpha=0.9, label="Excitatory")
        # ax.plot([-1, -1], [-1, -1], '.', color='k',
        #         ms=4, alpha=.9, label='Readout (L5e)')

        # bg = patches.Rectangle((480 / 2050 * seq_len, 0), 300 / 2050 * seq_len,
        #                        220 / 1000 * self.n_neurons, color='white', alpha=.9, zorder=101)
        # ax.add_patch(bg)
        # ax.legend(frameon=True, facecolor='white', framealpha=.9, edgecolor='white',
        #           fontsize=5 * scale, loc='center', bbox_to_anchor=(.3, .12)).set_zorder(102)
        ax.axvline(
            self.stimuli_init_time,
            linestyle="dashed",
            color="k",
            linewidth=1.5,
            alpha=1,
        )
        ax.axvline(
            self.stimuli_end_time, linestyle="dashed", color="k", linewidth=1.5, alpha=1
        )
        ax.set_ylim([0, self.n_neurons])
        ax.set_yticks([0, self.n_neurons])
        ax.set_ylabel("Network Neuron ID", fontsize=24)
        ax.set_xlim([0, seq_len])
        ax.set_xticks([0, seq_len])
        ax.tick_params(axis="both", which="major", labelsize=20)


class DriftingGrating:
    def __init__(
        self,
        scale=2.0,
        frequency=2.0,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        reverse=False,
        marker_size=1.0,
        alpha=1,
        color="g",
    ):
        self.marker_size = marker_size
        self.alpha = alpha
        self.color = color
        self.scale = scale
        self.stimuli_init_time = stimuli_init_time
        self.stimuli_end_time = stimuli_end_time
        self.reverse = reverse
        self.frequency = frequency

    def __call__(self, ax, simulation_length, stimulus_length=None):
        if stimulus_length is None:
            stimulus_length = simulation_length

        times = np.arange(stimulus_length)
        stimuli_speed = np.zeros((stimulus_length))
        if self.reverse:
            stimuli_speed[: self.stimuli_init_time] = self.frequency
            stimuli_speed[self.stimuli_end_time:] = self.frequency
        else:
            stimuli_speed[
                self.stimuli_init_time: self.stimuli_end_time
            ] = self.frequency

        ax.plot(
            times,
            stimuli_speed,
            color=self.color,
            ms=self.marker_size,
            alpha=self.alpha,
            linewidth=2 * self.scale,
        )
        ax.set_ylabel("TF \n [Hz]")
        ax.set_yticks([0, self.frequency])
        ax.set_yticklabels(["0", f"{self.frequency}"])
        ax.set_xlim([0, stimulus_length])
        ax.set_xticks(np.linspace(0, stimulus_length, 6))
        ax.set_xticklabels([str(int(x))
                           for x in np.linspace(0, simulation_length, 6)])
        ax.set_xlabel("Time [ms]", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)


class LGN_sample_plot:
    # Plot one realization of the LGN units response
    def __init__(
        self,
        firing_rates,
        spikes,
        stimuli_init_time=500,
        stimuli_end_time=1500,
        images_dir="Images",
        n_samples=2,
        directory="LGN units",
    ):
        self.firing_rates = firing_rates[0, :, :]
        self.spikes = spikes
        self.stimuli_init_time = stimuli_init_time
        self.stimuli_end_time = stimuli_end_time
        self.firing_rates_shape = self.firing_rates.shape
        self.n_samples = n_samples
        self.images_dir = images_dir
        self.directory = directory

    def __call__(self):
        for neuron_idx in np.random.choice(
            range(self.firing_rates_shape[1]), size=self.n_samples
        ):
            times = np.linspace(0, self.firing_rates_shape[0], self.firing_rates_shape[0])

            fig, axs = plt.subplots(2, sharex=True)
            axs[0].plot(times, self.firing_rates[:, neuron_idx], color="r", ms=1, alpha=0.7)
            axs[0].set_ylabel("Firing rate [Hz]")
            axs[1].plot(times, self.spikes[0, :, neuron_idx], color="b", ms=1, alpha=0.7)
            axs[1].set_yticks([0, 1])
            axs[1].set_ylim(0, 1)
            axs[1].set_xlabel("Time [ms]")
            axs[1].set_ylabel("Spikes")

            for subplot in range(2):
                axs[subplot].axvline(
                    self.stimuli_init_time,
                    linestyle="dashed",
                    color="gray",
                    linewidth=3,
                )
                axs[subplot].axvline(
                    self.stimuli_end_time, linestyle="dashed", color="gray", linewidth=3
                )

            fig.suptitle(f"LGN unit idx:{neuron_idx}")
            path = os.path.join(self.images_dir, self.directory)
            os.makedirs(path, exist_ok=True)
            fig.savefig(os.path.join(
                path, f"LGN unit idx_{neuron_idx}.png"), dpi=300)
            # close figure
            plt.close(fig)


class PopulationActivity:
    def __init__(self, n_neurons, network, stimuli_init_time=500, stimuli_end_time=1500,
                image_path="", data_dir="", filename='', core_radius=400):
        self.data_dir = data_dir
        self.n_neurons = n_neurons
        self.network = network
        self.stimuli_init_time = stimuli_init_time
        self.stimuli_end_time = stimuli_end_time
        self.filename = filename
        self.images_path = image_path
        self.core_radius = core_radius
        os.makedirs(self.images_path, exist_ok=True)

    def __call__(self, spikes, plot_core_only=True, bin_size=10):
        if plot_core_only:
            self.core_mask = other_v1_utils.isolate_core_neurons(
                self.network, radius=self.core_radius, data_dir=self.data_dir
            )
            self.n_neurons = np.sum(self.core_mask)
            # if self.n_neurons > 65871:
            #     self.n_neurons = 65871
            #     core_radius = 400
            #     self.core_mask = other_v1_utils.isolate_core_neurons(self.network, radius=core_radius, data_dir=self.data_dir)
            # else:
            #     self.core_mask = np.full(self.n_neurons, True)
        else:
            self.core_mask = np.full(self.n_neurons, True)

        self.spikes = np.array(spikes)[0, :, self.core_mask]
        self.spikes = np.transpose(self.spikes)
        self.neurons_ordering()
        # self.plot_populations_activity(bin_size)
        self.subplot_populations_activity(bin_size)

    def neurons_ordering(self):
        node_types = pd.read_csv(os.path.join(self.data_dir, "network/v1_node_types.csv"), sep=" ")
        path_to_h5 = os.path.join(self.data_dir, "network/v1_nodes.h5")

        with h5py.File(path_to_h5, mode='r') as node_h5:
            # Create mapping from node_type_id to pop_name
            node_types.set_index('node_type_id', inplace=True)
            node_type_id_to_pop_name = node_types['pop_name'].to_dict()

            # Map node_type_id to pop_name for all neurons and select population names of neurons in the present network 
            node_type_ids = node_h5['nodes']['v1']['node_type_id'][()][self.network['tf_id_to_bmtk_id']]
            true_node_type_ids = node_type_ids[self.core_mask]

        # Now order the pop_names according to their layer and type
        pop_orders = dict(sorted(node_type_id_to_pop_name.items(), key=lambda item: pop_ordering(item[1])))

        # Now we convert the neuron id (related to its pop_name) to an index related to its position in the y axis
        neuron_id_to_y = np.zeros(self.n_neurons, np.int32) - 1  # rest 1 to check at the end if every neuron has an index
        current_ind = 0
        self.layer_bounds = []
        self.ie_bounds = []
        current_pop_name = "e0"

        for pop_id, pop_name in pop_orders.items():
            # choose all the neurons of the given pop_id
            sel = true_node_type_ids == pop_id
            _n = np.sum(sel)
            # order the neurons by type in the y axis
            neuron_id_to_y[sel] = np.arange(current_ind, current_ind + _n)
            if int(pop_name[1]) > int(current_pop_name[1]):
                # register the change of layer
                self.layer_bounds.append(current_ind)
            if current_pop_name[0] == "i" and pop_name[0] == "e":
                # register the change of neuron type: exc -> inh
                self.ie_bounds.append(current_ind)
            current_ind += _n
            current_pop_name = pop_name

        # check that an y id has been given to every neuron
        assert np.sum(neuron_id_to_y < 0) == 0
        self.y_to_neuron_id = np.zeros(self.n_neurons, np.int32)
        self.y_to_neuron_id[neuron_id_to_y] = np.arange(self.n_neurons)
        assert np.all(self.y_to_neuron_id[neuron_id_to_y] == np.arange(self.n_neurons))

    def plot_populations_activity(self, bin_size=10):
        layers_label = ["i1", "i23", "e23", "i4", "e4", "i5", "e5", "i6", "e6"]
        neuron_class_bounds = np.concatenate((self.ie_bounds, self.layer_bounds))
        neuron_class_bounds = np.append(neuron_class_bounds, self.n_neurons)
        neuron_class_bounds.sort()

        for idx, label in enumerate(layers_label):
            init_idx = neuron_class_bounds[idx]
            end_idx = neuron_class_bounds[idx + 1]
            neuron_ids = self.y_to_neuron_id[init_idx:end_idx]
            n_neurons_class = len(neuron_ids)
            class_spikes = self.spikes[:, neuron_ids]
            m, n = class_spikes.shape
            H, W = int(m / bin_size), 1  # block-size
            n_spikes_bin = class_spikes.reshape(H, m // H, W, n // W).sum(axis=(1, 3))
            population_activity = n_spikes_bin / (n_neurons_class * bin_size * 0.001)

            fig = plt.figure()
            plt.plot(np.arange(0, self.spikes.shape[0], bin_size), population_activity)
            plt.axvline(
                self.stimuli_init_time,
                linestyle="dashed",
                color="gray",
                linewidth=1,
                zorder=10,
            )
            plt.axvline(
                self.stimuli_end_time,
                linestyle="dashed",
                color="gray",
                linewidth=1,
                zorder=10,
            )
            plt.xlabel("Time (ms)")
            plt.ylabel("Population activity (Hz)")
            plt.suptitle(f"Population activity of {label} neurons")
            path = os.path.join(self.images_path, "Populations activity")
            os.makedirs(path, exist_ok=True)
            fig.tight_layout()
            fig.savefig(os.path.join(path, f"{label}_population_activity.png"), dpi=300)
            plt.close(fig)

    def subplot_populations_activity(self, bin_size=10):
        layers_label = [
            "Inhibitory L1 neurons",
            "Inhibitory L23 neurons",
            "Excitatory L23 neurons",
            "Inhibitory L4 neurons",
            "Excitatory L4 neurons",
            "Inhibitory L5 neurons",
            "Excitatory L5 neurons",
            "Inhibitory L6 neurons",
            "Excitatory L6 neurons",
        ]
        neuron_class_bounds = np.concatenate((self.ie_bounds, self.layer_bounds))
        neuron_class_bounds = np.append(neuron_class_bounds, self.n_neurons)
        neuron_class_bounds.sort()

        population_activity_dict = {}

        for idx, label in enumerate(layers_label):
            init_idx = neuron_class_bounds[idx]
            end_idx = neuron_class_bounds[idx + 1]
            neuron_ids = self.y_to_neuron_id[init_idx:end_idx]
            n_neurons_class = len(neuron_ids)
            class_spikes = self.spikes[:, neuron_ids]
            m, n = class_spikes.shape
            H, W = int(m / bin_size), 1  # block-size
            n_spikes_bin = class_spikes.reshape(H, m // H, W, n // W).sum(axis=(1, 3))
            population_activity = n_spikes_bin / (n_neurons_class * bin_size * 0.001)
            population_activity_dict[label] = population_activity

        time = np.arange(0, self.spikes.shape[0], bin_size)
        fig = plt.figure(constrained_layout=False)
        # fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0.15, wspace=0.15)
        ax1 = plt.subplot(5, 1, 1)
        plt.plot(time, population_activity_dict["Inhibitory L1 neurons"],
            label="Inhibitory L1 neurons", color="b",
        )
        plt.legend(fontsize=6)
        plt.tick_params(axis="both", labelsize=7)
        # plt.xlabel('Time (ms)', fontsize=7)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.ylabel("Population \n activity (Hz)", fontsize=7)
        plt.axvline(
            self.stimuli_init_time,
            linestyle="dashed",
            color="gray",
            linewidth=1,
            zorder=10,
        )
        plt.axvline(
            self.stimuli_end_time,
            linestyle="dashed",
            color="gray",
            linewidth=1,
            zorder=10,
        )

        ax2 = None
        for i in range(3, 9):
            if i % 2 == 1:
                ax1 = plt.subplot(5, 2, i, sharex=ax1, sharey=ax1)
                plt.plot(time, population_activity_dict[layers_label[i - 2]],
                    label=layers_label[i - 2], color="b",)
                plt.ylabel("Population \n activity (Hz)", fontsize=7)
                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.legend(fontsize=6, loc="upper right")
                plt.tick_params(axis="both", labelsize=7)
                plt.axvline(
                    self.stimuli_init_time,
                    linestyle="dashed",
                    color="gray",
                    linewidth=1,
                    zorder=10,
                )
                plt.axvline(
                    self.stimuli_end_time,
                    linestyle="dashed",
                    color="gray",
                    linewidth=1,
                    zorder=10,
                )
            else:
                if ax2 == None:
                    ax2 = plt.subplot(5, 2, i, sharex=ax1)
                else:
                    ax2 = plt.subplot(5, 2, i, sharex=ax2, sharey=ax2)
                plt.plot(time, population_activity_dict[layers_label[i - 2]],
                    label=layers_label[i - 2], color="r",)
                plt.setp(ax2.get_xticklabels(), visible=False)
                plt.legend(fontsize=6, loc="upper right")
                plt.tick_params(axis="both", labelsize=7)
                plt.axvline(
                    self.stimuli_init_time,
                    linestyle="dashed",
                    color="gray",
                    linewidth=1,
                    zorder=10,
                )
                plt.axvline(
                    self.stimuli_end_time,
                    linestyle="dashed",
                    color="gray",
                    linewidth=1,
                    zorder=10,
                )

        ax1 = plt.subplot(5, 2, 9, sharex=ax1, sharey=ax1)
        plt.plot(
            time,
            population_activity_dict[layers_label[7]],
            label=layers_label[7],
            color="b",
        )
        plt.ylabel("Population \n activity (Hz)", fontsize=7)
        plt.xlabel("Time [ms]", fontsize=7)
        plt.tick_params(axis="both", labelsize=7)
        plt.legend(fontsize=6, loc="upper right")
        plt.axvline(
            self.stimuli_init_time,
            linestyle="dashed",
            color="gray",
            linewidth=1,
            zorder=10,
        )
        plt.axvline(
            self.stimuli_end_time,
            linestyle="dashed",
            color="gray",
            linewidth=1,
            zorder=10,
        )

        ax2 = plt.subplot(5, 2, 10, sharex=ax2, sharey=ax2)
        plt.plot(
            time,
            population_activity_dict[layers_label[8]],
            label=layers_label[8],
            color="r",
        )
        plt.xlabel("Time [ms]", fontsize=7)
        plt.tick_params(axis="both", labelsize=7)
        plt.legend(fontsize=6, loc="upper right")
        plt.axvline(
            self.stimuli_init_time,
            linestyle="dashed",
            color="gray",
            linewidth=1,
            zorder=10,
        )
        plt.axvline(
            self.stimuli_end_time,
            linestyle="dashed",
            color="gray",
            linewidth=1,
            zorder=10,
        )

        plt.subplots_adjust(
            left=0.1, bottom=0.07, right=0.99, top=0.99, wspace=0.17, hspace=0.17
        )

        path = os.path.join(self.images_path, "Populations activity")
        os.makedirs(path, exist_ok=True)
        # fig.tight_layout()
        fig.savefig(os.path.join(
            path, "subplot_population_activity.png"), dpi=300)
        plt.close(fig)


