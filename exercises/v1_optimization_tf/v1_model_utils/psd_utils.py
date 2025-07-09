
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal, stats
from collections import defaultdict
from statsmodels.stats.multitest import multipletests
import pandas as pd
import other_v1_utils
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


def compute_band_power(freqs, psd, band):
    idx = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapz(psd[idx], freqs[idx])

def bootstrap_relative_change(stim_trials, spont_trials, n_boot=1000):
    np.random.seed(42)
    boot_deltas = []
    for _ in range(n_boot):
        stim_sample = np.random.choice(stim_trials, size=len(stim_trials), replace=True)
        spont_sample = np.random.choice(spont_trials, size=len(spont_trials), replace=True)
        delta = (np.mean(stim_sample) - np.mean(spont_sample)) / np.mean(spont_sample)
        boot_deltas.append(delta)
    return np.mean(boot_deltas), np.percentile(boot_deltas, [2.5, 97.5])

def compare_band_power_bootstrap(stim_trials_list, spont_trials_list, bands, save_path=None):
    layer_order = ["L1", "L2/3", "L4", "L5", "L6"]
    band_colors = {
        "Theta (4-8 Hz)": "blue",
        "Alpha (8-12 Hz)": "green",
        "Beta (12-30 Hz)": "orange",
        "Gamma (30-80 Hz)": "red"
    }

    data = []
    for cell_type in stim_trials_list[0]:
        if cell_type not in spont_trials_list[0]:
            continue

        for band_name, band_range in bands.items():
            stim_band_vals = []
            spont_band_vals = []

            for trial in stim_trials_list:
                freqs = trial[cell_type]['frequencies']
                psd = trial[cell_type]['psd']
                stim_band_vals.append(compute_band_power(freqs, psd, band_range))

            for trial in spont_trials_list:
                freqs = trial[cell_type]['frequencies']
                psd = trial[cell_type]['psd']
                spont_band_vals.append(compute_band_power(freqs, psd, band_range))

            mean_delta, ci = bootstrap_relative_change(stim_band_vals, spont_band_vals)

            try:
                stat, pval = stats.wilcoxon(np.array(stim_band_vals) - np.array(spont_band_vals))
            except ValueError:
                pval = np.nan

            data.append({
                "Cell Type": cell_type,
                "Band": band_name,
                "Relative Delta Power": mean_delta,
                "CI lower": ci[0],
                "CI upper": ci[1],
                "p-value": pval
            })

    df = pd.DataFrame(data)

    reject, pvals_corrected, _, _ = multipletests(df["p-value"].fillna(1.0), method='fdr_bh')
    df["p-adj"] = pvals_corrected
    df["Significant"] = reject

    df["Layer"] = df["Cell Type"].apply(lambda x: x.split()[0] if x.startswith("L") else "Other")
    df = df.sort_values(by=["Layer", "Cell Type"], key=lambda col: col.map({k: i for i, k in enumerate(layer_order)}) if col.name == "Layer" else col, ascending=False)

    max_abs = np.nanmax(np.abs(df[["CI lower", "CI upper"]].values))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (band, color) in enumerate(band_colors.items()):
        ax = axes[i]
        subdf = df[df["Band"] == band].reset_index(drop=True)

        for idx, row in subdf.iterrows():
            ax.barh(
                y=idx,
                width=row["Relative Delta Power"],
                color=color, edgecolor='black',
                xerr=[[row["Relative Delta Power"] - row["CI lower"]],
                      [row["CI upper"] - row["Relative Delta Power"]]],
                capsize=3, alpha=0.7
            )
            if row["Significant"]:
                ax.text(max_abs * 1.05, idx, '*', fontsize=14, va='center', ha='left', color='black')

        ax.axvline(0, color='gray', linestyle='--')
        ax.set_yticks(range(len(subdf)))
        # increase the fontsize of both axis ticks
        ax.tick_params(axis='both', labelsize=14)
        ax.set_yticklabels(subdf["Cell Type"], fontsize=14)
        ax.set_title(f"{band}", fontsize=18, fontweight='bold')
        ax.set_facecolor(color)
        ax.patch.set_alpha(0.05)
        ax.set_xlim(-max_abs * 1.2, max_abs * 1.2)

        yticklabels = list(subdf["Cell Type"])
        layer_bounds = {}
        for i_, label in enumerate(yticklabels):
            layer = label.split()[0] if label.startswith("L") else None
            if layer not in layer_bounds:
                layer_bounds[layer] = [i_, i_]
            else:
                layer_bounds[layer][1] = i_

        for i_, (layer, (start, end)) in enumerate(layer_bounds.items()):
            if layer_order.index(layer) % 2 == 0:
                rect = patches.Rectangle(
                    (-max_abs * 1.2, start - 0.5),
                    width=2 * max_abs * 1.2,
                    height=(end - start + 1),
                    color='gray', alpha=0.05, zorder=0
                )
                ax.add_patch(rect)

    # fig.suptitle("Relative Change in Band Power with 95% CI and FDR-Corrected Significance (*q < 0.05)", fontsize=24)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'relative_band_power_comparison.png'), dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

def plot_absolute_band_power(stim_trials_list, spont_trials_list, bands, save_path=None):

    layer_order = ["L6", "L5", "L4", "L2/3", "L1"]  # reversed for top-down
    band_colors = {
        "Theta (4-8 Hz)": "blue",
        "Alpha (8-12 Hz)": "green",
        "Beta (12-30 Hz)": "orange",
        "Gamma (30-80 Hz)": "red"
    }

    def compute_sem(values):
        values = np.array(values)
        return np.std(values) / np.sqrt(len(values))

    def pval_to_asterisks(p):
        # if p < 0.001:
        #     return '***'
        # elif p < 0.01:
        #     return '**'
        if p < 0.05:
            return '*'
        else:
            return ''

    data = []
    pvals = []
    for cell_type in stim_trials_list[0]:
        if cell_type not in spont_trials_list[0]:
            continue

        for band_name, band_range in bands.items():
            stim_band_vals = []
            spont_band_vals = []

            for trial in stim_trials_list:
                freqs = trial[cell_type]['frequencies']
                psd = trial[cell_type]['psd']
                stim_band_vals.append(compute_band_power(freqs, psd, band_range))

            for trial in spont_trials_list:
                freqs = trial[cell_type]['frequencies']
                psd = trial[cell_type]['psd']
                spont_band_vals.append(compute_band_power(freqs, psd, band_range))

            try:
                stat, pval = stats.wilcoxon(np.array(stim_band_vals) - np.array(spont_band_vals))
            except ValueError:
                pval = 1.0

            pvals.append(pval)

            data.append({
                "Cell Type": cell_type,
                "Band": band_name,
                "Condition": "Stimulus",
                "Power": np.mean(stim_band_vals),
                "SEM": compute_sem(stim_band_vals),
                "pval": pval
            })
            data.append({
                "Cell Type": cell_type,
                "Band": band_name,
                "Condition": "Spontaneous",
                "Power": np.mean(spont_band_vals),
                "SEM": compute_sem(spont_band_vals),
                "pval": pval
            })

    reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh', alpha=0.01)

    for i, row in enumerate(data):
        row['p-adj'] = pvals_corrected[i // 2]
        row['Significant'] = reject[i // 2]

    df = pd.DataFrame(data)
    df["Layer"] = df["Cell Type"].apply(lambda x: x.split()[0] if x.startswith("L") else "Other")
    df = df.sort_values(by=["Layer", "Cell Type", "Condition"], key=lambda col: col.map({k: i for i, k in enumerate(layer_order[::-1])}) if col.name == "Layer" else col, ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharex=False, sharey=True)
    axes = axes.flatten()

    for i, (band, color) in enumerate(band_colors.items()):
        ax = axes[i]
        subdf = df[df["Band"] == band].reset_index(drop=True)

        spacing = 2.5
        cell_types = subdf["Cell Type"].unique()
        ypos = {}
        for idx, ct in enumerate(cell_types):
            ypos[(ct, "Spontaneous")] = idx * spacing
            ypos[(ct, "Stimulus")] = idx * spacing + 1

        max_x = 0
        handles = []
        labels = []
        for condition, c_label in zip(["Spontaneous", "Stimulus"], ["gray", color]):
            cond_df = subdf[subdf["Condition"] == condition]
            for j, (_, row) in enumerate(cond_df.iterrows()):
                y = ypos[(row["Cell Type"], condition)]
                width = row['Power']
                sem = row['SEM']
                error_left = error_right = sem
                max_x = max(max_x, width + error_right)

                bar = ax.barh(
                    y=y,
                    width=width,
                    height=0.8,
                    color=c_label,
                    edgecolor='black',
                    alpha=0.7,
                    label=condition if j == 0 else None,
                    xerr=[[error_left], [error_right]],
                    capsize=3
                )
                if j == 0:
                    handles.append(bar[0])
                    labels.append(condition)

        for _, row in subdf.iterrows():
            if row['Condition'] == 'Stimulus' and row['Significant']:
                y = ypos[(row["Cell Type"], "Stimulus")]
                stars = pval_to_asterisks(row['p-adj'])
                ax.text(max_x * 1.02, y, stars, fontsize=20, va='center', ha='left', color='black')

        yticks = [ypos[(ct, "Spontaneous")] + 0.5 for ct in cell_types]
        ax.set_yticks(yticks)
        ax.set_yticklabels(cell_types, fontsize=14)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xlim(0, max_x * 1.1)

        ax.set_title(f"{band}", fontsize=20, fontweight='bold')
        ax.set_facecolor(color)
        ax.patch.set_alpha(0.05)
        if i == 2 or i == 3:
            ax.set_xlabel("Mean Band Power", fontsize=18)
        ax.axvline(0, color='gray', linestyle='--')
        if i == 0:
            ax.legend(handles, labels, fontsize=14)

        layer_bounds = {}
        for ct in cell_types:
            layer = ct.split()[0]
            start = ypos[(ct, "Spontaneous")]
            end = ypos[(ct, "Stimulus")]
            if layer not in layer_bounds:
                layer_bounds[layer] = [start, end]
            else:
                layer_bounds[layer][1] = end

        for i_, (layer, (start, end)) in enumerate(layer_bounds.items()):
            if layer in layer_order[::-1] and layer_order[::-1].index(layer) % 2 == 0:
                rect = patches.Rectangle(
                    (0, start - 0.5),
                    width=max_x * 1.1,
                    height=(end - start + 1.5),
                    color='gray', alpha=0.05, zorder=0
                )
                ax.add_patch(rect)

    # fig.suptitle("Mean Band Power Across Conditions with SEM and FDR-Corrected Significance (* p < 0.05, ** p < 0.01, *** p < 0.001)", fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'absolute_band_power.png'), dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

def plot_psd_by_layer(psd_dict, normalize=False, normalize_by_n2=True, title_suffix="Spontaneous", save_path=None):
    colors = {
        'Exc': 'r', 'PV': 'b', 'SST': 'g', 'VIP': 'darkviolet',
        'Htr3a': 'pink', 'ET': 'firebrick', 'IT': 'tomato', 'NP': 'lightcoral'
    }

    def get_cell_type_color(cell_type):
        for sub in colors:
            if sub in cell_type:
                return colors[sub]
        return 'gray'

    # Group by layer
    layer_cell_types = defaultdict(list)
    for cell_type in psd_dict:
        if cell_type.startswith('L'):
            layer = cell_type.split()[0]
            layer_cell_types[layer].append(cell_type)

    expected_layers = ['L1', 'L2/3', 'L4', 'L5', 'L6']
    for layer in expected_layers:
        layer_cell_types.setdefault(layer, [])

    # Find global y-axis limits for consistent scale
    y_min, y_max = float('inf'), float('-inf')
    for cell_type, data in psd_dict.items():
        if cell_type.startswith('L'):  # Only consider cortical layers
            values = data['mean_psd'] + data['sem_psd']  # Upper bound
            y_max = max(y_max, np.max(values))
            values = data['mean_psd'] - data['sem_psd']  # Lower bound
            y_min = min(y_min, np.min(values[values > 0]))  # Exclude negative or zero values for log scale
    
    # Add some margin
    y_min *= 0.8
    y_max *= 1.2

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    subplot_positions = {
        'L1': gs[0, :], 'L2/3': gs[1, 0], 'L4': gs[1, 1],
        'L5': gs[2, 0], 'L6': gs[2, 1]
    }

    # Track axes for shared y-limits
    all_axes = []
    first_ax = None

    for layer_idx, (layer, pos) in enumerate(subplot_positions.items()):
        if layer_idx == 0:
            # Create the first subplot
            ax = fig.add_subplot(pos)
            first_ax = ax
        else:
            # Share y-axis with the first subplot
            ax = fig.add_subplot(pos, sharey=first_ax)
        
        all_axes.append(ax)

        for cell_type in layer_cell_types[layer]:
            color = get_cell_type_color(cell_type)
            ax.plot(psd_dict[cell_type]['frequencies'], psd_dict[cell_type]['mean_psd'],
                    color=color, linewidth=3, label=cell_type)
            ax.fill_between(
                psd_dict[cell_type]['frequencies'],
                psd_dict[cell_type]['mean_psd'] - psd_dict[cell_type]['sem_psd'],
                psd_dict[cell_type]['mean_psd'] + psd_dict[cell_type]['sem_psd'],
                color=color, alpha=0.2
            )

        is_first = (layer == 'L1')
        ax.axvspan(4, 8, alpha=0.1, color="blue", label="Theta (4-8 Hz)" if is_first else "_nolegend_")
        ax.axvspan(8, 12, alpha=0.1, color="green", label="Alpha (8-12 Hz)" if is_first else "_nolegend_")
        ax.axvspan(12, 30, alpha=0.1, color="orange", label="Beta (12-30 Hz)" if is_first else "_nolegend_")
        ax.axvspan(30, 80, alpha=0.1, color="red", label="Gamma (30-80 Hz)" if is_first else "_nolegend_")

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.5, 100)

        ylabel = "Power Spectral Density"
        if normalize_by_n2:
            ylabel += " / N²"
        elif normalize:
            ylabel = "Normalized Power Spectral Density"

        if layer in ['L1', 'L2/3', 'L5']:
            ax.set_ylabel(ylabel, fontsize=22, fontweight='bold')
        if layer in ['L5', 'L6']:
            ax.set_xlabel("Frequency (Hz)", fontsize=22, fontweight='bold')

        ax.set_title(f"Layer {layer[1:]}", fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, which='both', linestyle='--', alpha=0.4, linewidth=1)

        if layer == 'L1' or layer_cell_types[layer]:
            leg = ax.legend(loc='lower left', fontsize=16, frameon=True, framealpha=0.8, ncol=2)
            leg.get_frame().set_edgecolor('black')
            leg.get_frame().set_linewidth(1.5)

    # Set the y-limits for all plots
    for ax in all_axes:
        ax.set_ylim(y_min, y_max)

    # fig.suptitle(f"{'Normalized ' if normalize else ''}Power Spectral Density by Cell Type for {title_suffix}",
                #  fontsize=28, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'psd_by_layer_{title_suffix}.png'), dpi=300, bbox_inches='tight', transparent=False)
    plt.close()


class PSDAnalyzer:
    def __init__(self, network, fs=1.0, analyze_core_only=True, population_average=True, normalize=True, normalize_by_n2=False, normalize_by_rate=True,
                 save_path=None, data_dir='GLIF_network'):
        
        if analyze_core_only:
            # Isolate the core neurons
            pop_names = other_v1_utils.pop_names(network, core_radius=200, data_dir=data_dir)
            self.core_mask = other_v1_utils.isolate_core_neurons(network, radius=200, data_dir=data_dir)
            # spikes = spikes[:, :, :, core_mask]
        else:
            pop_names = other_v1_utils.pop_names(network, core_radius=400, data_dir=data_dir)
            n_core_neurons = len(pop_names)
            self.core_mask = np.ones(n_core_neurons, dtype=bool)

        cell_types = [other_v1_utils.pop_name_to_cell_type(name) for name in pop_names]
        self.cell_types_map = {i: cell_types[i] for i in range(len(cell_types))}
        self.fs = fs
        self.population_average = population_average
        self.normalize = normalize
        self.normalize_by_n2 = normalize_by_n2
        self.normalize_by_rate = normalize_by_rate
        self.save_path = save_path
        # check if save_path exists
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        self.BANDS = {
                        "Theta (4-8 Hz)": (4, 8),
                        "Alpha (8-12 Hz)": (8, 12),
                        "Beta (12-30 Hz)": (12, 30),
                        "Gamma (30-80 Hz)": (30, 80)
                    }

    def calculate_power_spectrum(self, spike_times, t_start, t_end, nperseg=1000, noverlap=500):
        duration = t_end - t_start
        bins = np.linspace(t_start, t_end, int(duration * self.fs) + 1)
        spike_train, _ = np.histogram(spike_times, bins=bins)

        freqs, psd = signal.welch(
            spike_train,
            fs=self.fs,
            nperseg=min(nperseg, len(spike_train) // 2),
            noverlap=min(noverlap, len(spike_train) // 4),
            window="hann",
            detrend="constant",
            scaling="density",
        )

        freqs *= 1000  # kHz to Hz
        if self.normalize:
            psd *= freqs

        return freqs, psd

    def compute_cell_type_psds(self, spike_data, node_ids, t_start, t_end):
        neuron_spikes = defaultdict(list)
        for t, nid in zip(spike_data, node_ids):
            neuron_spikes[nid].append(t)

        cell_type_neurons = defaultdict(list)
        for nid, spikes in neuron_spikes.items():
            cell_type = self.cell_types_map.get(nid)
            if cell_type:
                cell_type_neurons[cell_type].append(spikes)

        spectra = {}
        for cell_type, spike_lists in cell_type_neurons.items():
            all_spikes = [t for spikes in spike_lists for t in spikes]
            if not all_spikes:
                continue

            if self.population_average:
                freqs, psd = self.calculate_power_spectrum(all_spikes, t_start, t_end)
                if self.normalize_by_n2:
                    psd /= len(spike_lists) ** 2
                if self.normalize_by_rate and len(all_spikes) > 0:
                    psd /= len(all_spikes) ** 2
            else:
                individual_psds = [
                    self.calculate_power_spectrum(spikes, t_start, t_end)[1] for spikes in spike_lists if spikes
                ]
                psd = np.mean(individual_psds, axis=0)
                freqs = self.calculate_power_spectrum(spike_lists[0], t_start, t_end)[0]

            spectra[cell_type] = {
                'frequencies': freqs,
                'psd': psd,
                'num_neurons': len(spike_lists),
                'avg_spikes_per_neuron': len(all_spikes) / len(spike_lists)
            }

        return spectra

    def average_psd_over_trials(self, psd_list_by_trial):
        result = defaultdict(lambda: defaultdict(list))
        for trial_psd in psd_list_by_trial:
            for cell_type, metrics in trial_psd.items():
                for key, val in metrics.items():
                    result[cell_type][key].append(val)

        power_spectrum_dict = {}
        for cell_type, data in result.items():
            avg_psd = np.mean(data['psd'], axis=0)
            sem_psd = np.std(data['psd'], axis=0) / np.sqrt(len(data['psd']))
            power_spectrum_dict[cell_type] = {
                'frequencies': np.mean(data['frequencies'], axis=0),
                'mean_psd': avg_psd,
                'sem_psd': sem_psd,
                'num_neurons': np.mean(data['num_neurons']),
                'avg_spikes_per_neuron': np.mean(data['avg_spikes_per_neuron'])
            }
        return power_spectrum_dict

    def process_trials(self, spikes, t_start, t_end):
        self.t_start = t_start
        self.t_end = t_end

        psd_trials = []
        for trial in range(spikes.shape[0]):
            spike_times, node_ids = np.where(spikes[trial] > 0)
            psd = self.compute_cell_type_psds(
                spike_times, node_ids, t_start, t_end
            )
            psd_trials.append(psd)
        return psd_trials

    def __call__(self, spikes, t_spont_start=None, t_spont_end=None, t_evoked_start=None, t_evoked_end=None):
        # Mask core neurons
        if len(spikes.shape) == 4:  # Shape: [trials, angles, time, neurons]
            # Reshape spikes from [trials, angles, time, neurons] to [trials*angles, time, neurons]
            spikes = np.reshape(spikes, [spikes.shape[0] * spikes.shape[1], spikes.shape[2], spikes.shape[3]])
        else:
            raise ValueError(f"Unexpected spike array shape: {spikes.shape}. Expected 3D or 4D array.")

        # Mask core neurons
        spikes = spikes[:, :, self.core_mask]

        # Process evoked trials if time window is provided
        evoked_psds = None
        evoked_power_spectrum_dict = None
        if t_evoked_start is not None and t_evoked_end is not None and t_evoked_end > t_evoked_start:
            evoked_psds = self.process_trials(spikes, t_evoked_start, t_evoked_end)
            evoked_power_spectrum_dict = self.average_psd_over_trials(evoked_psds)
            plot_psd_by_layer(evoked_power_spectrum_dict, normalize=self.normalize, normalize_by_n2=self.normalize_by_n2, 
                            title_suffix='Evoked', save_path=self.save_path)
        else:
            print("Skipping evoked analysis: Invalid time window provided.")

        # Process spontaneous trials if time window is provided
        spont_psds = None
        spont_power_spectrum_dict = None
        if t_spont_start is not None and t_spont_end is not None and t_spont_end > t_spont_start:
            spont_psds = self.process_trials(spikes, t_spont_start, t_spont_end)
            spont_power_spectrum_dict = self.average_psd_over_trials(spont_psds)
            plot_psd_by_layer(spont_power_spectrum_dict, normalize=self.normalize, normalize_by_n2=self.normalize_by_n2, 
                            title_suffix='Spontaneous', save_path=self.save_path)

        else:
            print("Skipping spontaneous analysis: Invalid time window provided.")
    
        # Compare evoked vs spontaneous data if both are available
        if evoked_psds is not None and spont_psds is not None:
            compare_band_power_bootstrap(evoked_psds, spont_psds, self.BANDS, save_path=self.save_path)
            plot_absolute_band_power(evoked_psds, spont_psds, self.BANDS, save_path=self.save_path)
            