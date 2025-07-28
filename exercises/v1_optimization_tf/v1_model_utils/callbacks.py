import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime as dt
import subprocess
from time import time
import pickle as pkl
from numba import njit
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import ks_2samp
from v1_model_utils import other_v1_utils
from v1_model_utils.plotting_utils import InputActivityFigure, PopulationActivity
from v1_model_utils.model_metrics_analysis import ModelMetricsAnalysis
from v1_model_utils.model_metrics_analysis import calculate_Firing_Rate, get_borders, draw_borders
from v1_model_utils.psd_utils import PSDAnalyzer
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
if shutil.which('latex') is not None:
    use_tex = True
else:
    use_tex = False
    print("LaTeX not found. Using MathText for rendering.")
plt.rcParams['text.usetex'] = use_tex

def printgpu(gpu_id=0):
    if tf.config.list_physical_devices('GPU'):
        # Check TensorFlow memory info
        meminfo = tf.config.experimental.get_memory_info(f'GPU:{gpu_id}')
        current = meminfo['current'] / 1024**3
        peak = meminfo['peak'] / 1024**3
        print(f'    TensorFlow GPU {gpu_id} Memory Usage: {current:.2f} GiB, Peak Usage: {peak:.2f} GiB')
        # Check GPU memory using nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, encoding='utf-8'
        )  # MiB
        # Split output into lines for each GPU
        gpu_memory_info = result.stdout.strip().split('\n')
        if gpu_id is not None:
            # Display memory info for a specific GPU
            if gpu_id < len(gpu_memory_info):
                used, free, total = [float(x)/1024 for x in gpu_memory_info[gpu_id].split(',')]
                print(f"    Total GPU {gpu_id} Memory Usage: Used: {used:.2f} GiB, Free: {free:.2f} GiB, Total: {total:.2f} GiB")
            else:
                print(f"    Invalid GPU ID: {gpu_id}. Available GPUs: {len(gpu_memory_info)}")
        else:
            # Display memory info for all GPUs
            for i, info in enumerate(gpu_memory_info):
                used, free, total = [float(x)/1024 for x in info.split(',')]
                print(f"    Total GPU {gpu_id} Memory Usage: Used: {used:.2f} GiB, Free: {free:.2f} GiB, Total: {total:.2f} GiB")

def compose_str(metrics_values):
        # _acc, _loss, _rate, _rate_loss, _voltage_loss, _regularizers_loss, _osi_dsi_loss, _sync_loss = metrics_values
        _loss, _rate, _rate_loss, _voltage_loss, _regularizers_loss, _osi_dsi_loss, _sync_loss = metrics_values

        _s = f'Loss {_loss:.4f}, '
        _s += f'RLoss {_rate_loss:.4f}, '
        _s += f'VLoss {_voltage_loss:.4f}, '
        _s += f'RegLoss {_regularizers_loss:.4f}, '
        _s += f'OLoss {_osi_dsi_loss:.4f}, '
        _s += f'SLoss {_sync_loss:.4f}, '
        # _s += f'Accuracy {_acc:.4f}, '
        _s += f'Rate {_rate:.4f}'
        return _s

def compute_ks_statistics(df, metric='Weight', min_n_sample=15):
    """
    Compute the Kolmogorov-Smirnov statistic and similarity scores for each cell type in the dataframe.
    Parameters:
    - df: pd.DataFrame, contains data with columns 'data_type' and 'Evoked rate (Hz)', and indexed by cell type.
    Returns:
    - mean_similarity_score: float, the mean of the similarity scores computed across all cell types.
    """
    # Get unique cell types
    # cell_types = df.index.unique()
    cell_types = df['Post_names'].unique()
    # Initialize a dictionary to store the results
    ks_results = {}
    similarity_scores = {}
    # Iterate over cell types
    for cell_type in cell_types:
        # Filter data for current cell type from two different data types
        # df1 = df.loc[(df.index == cell_type) & (df['data_type'] == 'V1/LM GLIF model'), metric]
        # df2 = df.loc[(df.index == cell_type) & (df['data_type'] == 'Neuropixels'), metric]
        df1 = df.loc[(df['Post_names'] == cell_type) & (df['Weight Type'] == 'Initial weight'), metric]
        df2 = df.loc[(df['Post_names'] == cell_type) & (df['Weight Type'] == 'Final weight'), metric]
        # Drop NA values
        df1.dropna(inplace=True)
        df2.dropna(inplace=True)
        # Calculate the Kolmogorov-Smirnov statistic
        if len(df1) >= min_n_sample and len(df2) >= min_n_sample:
            ks_stat, p_value = ks_2samp(df1, df2)
            ks_results[cell_type] = (ks_stat, p_value)
            similarity_scores[cell_type] = 1 - ks_stat

    # Calculate the mean of the similarity scores and return it
    mean_similarity_score = np.mean(list(similarity_scores.values()))
    return mean_similarity_score

# Define a function to compute the exponential decay of a spike train
def exponential_decay_filter(spike_train, tau=20):
    decay_factor = np.exp(-1/tau)
    continuous_signal = np.zeros_like(spike_train, dtype=float)
    continuous_signal[0] = spike_train[0]
    for i in range(1, len(spike_train)):
        continuous_signal[i] = decay_factor * continuous_signal[i-1] + spike_train[i]
    return continuous_signal

@njit
def pop_fano(spikes, bin_sizes):
    fanos = np.zeros(len(bin_sizes))
    for i, bin_width in enumerate(bin_sizes):
        bin_size = int(np.round(bin_width * 1000))
        max_index = spikes.shape[0] // bin_size * bin_size
        # drop the last bin if it is not complete
        # sum over neurons to get the spike counts
        # trimmed_spikes = np.sum(spikes[:max_index, :], axis=1) 
        trimmed_spikes = spikes[:max_index]
        trimmed_spikes = np.reshape(trimmed_spikes, (max_index // bin_size, bin_size))
        # sum over the bins
        sp_counts = np.sum(trimmed_spikes, axis=1)
        # Calculate the mean of the spike counts
        mean_count = np.mean(sp_counts)
        if mean_count > 0:
            # Calculate the Fano Factor
            fanos[i] = np.var(sp_counts) / mean_count
                 
    return fanos

# # create a class for callbacks in other training sessions (e.g. validation, testing)
class OsiDsiCallbacks:
    def __init__(self, network, lgn_input, bkg_input, flags, logdir, current_epoch=0,
                pre_delay=50, post_delay=50, model_variables_init=None):
        self.n_neurons = flags.neurons
        self.network = network
        self.lgn_input = lgn_input
        self.bkg_input = bkg_input
        self.flags = flags
        self.logdir = logdir
        self.images_dir = self.logdir
        self.pre_delay = pre_delay
        self.post_delay = post_delay
        self.current_epoch = current_epoch
        self.model_variables_dict = model_variables_init
        # Analize changes in trainable variables.
        if self.model_variables_dict is not None:
            for var in self.model_variables_dict['Best'].keys():
                t0 = time()
                self.trainable_variable_change_heatmaps_and_distributions(var)
                print(f'Time spent in {var}: {time()-t0}')

    def trainable_variable_change_heatmaps_and_distributions(self, variable):
        node_types_voltage_scale = (self.network['node_params']['V_th'] - self.network['node_params']['E_L']).astype(np.float32)
        node_type_ids = self.network['node_type_ids']
        if 'rest_of_brain_weights' in variable:
            voltage_scale = node_types_voltage_scale[node_type_ids[self.bkg_input['indices'][:, 0]]]
            self.node_to_pop_weights_analysis(self.bkg_input['indices'], variable=variable, voltage_scale=voltage_scale)
        elif'sparse_input_weights' in variable:
            voltage_scale = node_types_voltage_scale[node_type_ids[self.lgn_input['indices'][:, 0]]]
            self.node_to_pop_weights_analysis(self.lgn_input['indices'], variable=variable, voltage_scale=voltage_scale)
        elif 'sparse_recurrent_weights' in variable:
            indices = self.network['synapses']['indices']
            voltage_scale = node_types_voltage_scale[node_type_ids[indices[:, 0]]]
            self.pop_to_pop_weights_analysis(indices, variable=variable, voltage_scale=voltage_scale)
            self.pop_to_pop_weights_distribution(indices, variable=variable, voltage_scale=voltage_scale)
            # self.pop_to_pop_weights_distribution_by_layer(indices, variable=variable, voltage_scale=voltage_scale)
    
    def node_to_pop_weights_analysis(self, indices, variable='', voltage_scale=None):
        pop_names = other_v1_utils.pop_names(self.network)
        target_cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in pop_names]
        post_indices = indices[:, 0]
        post_cell_types = [target_cell_types[i] for i in post_indices]

        # Map variable names to descriptive titles
        title_mapping = {
            'sparse_recurrent_weights:0': 'Recurrent Weights',
            'rest_of_brain_weights:0': 'Background Input Weights',
            'sparse_input_weights:0': 'LGN Input Weights'
        }
        # Get descriptive title or use original variable name if not in mapping
        descriptive_title = title_mapping.get(variable, variable)

        if voltage_scale is not None:
            # Normalize the weights by the voltage scale
            initial_weights = self.model_variables_dict['Initial'][variable] * voltage_scale
            final_weights = self.model_variables_dict['Best'][variable] * voltage_scale
        else:
            initial_weights = self.model_variables_dict['Initial'][variable]
            final_weights = self.model_variables_dict['Best'][variable]

        # Create DataFrame with all the necessary data
        df = pd.DataFrame({
            'Post_names': post_cell_types * 2,  # Duplicate node names for initial and final weights
            'Weight': initial_weights.tolist() + final_weights.tolist(),  # Combine initial and final weights
            'Weight Type': ['Initial weight'] * len(initial_weights) + ['Final weight'] * len(final_weights)  # Distinguish between initial and final weights
        })

        # Sort the dataframe by Node Name and then by Type to ensure consistent order
        df = df.sort_values(['Post_names', 'Weight Type'])

        # Plotting
        boxplots_dir = os.path.join(self.images_dir, f'Boxplots/{variable}')
        os.makedirs(boxplots_dir, exist_ok=True)
        
        # Increased fontsize for labels and ticks
        label_fontsize = 16
        tick_fontsize = 14
        title_fontsize = 18
        
        fig = plt.figure(figsize=(12, 6))
        ax = fig.gca()
        similarity_score = compute_ks_statistics(df, metric='Weight', min_n_sample=15)
        hue_order = ['Initial weight', 'Final weight']
        palette = {"Initial weight": "#87CEEB", "Final weight": "#FFA500"}
        
        sns.boxplot(x='Post_names', y='Weight', hue='Weight Type', hue_order=hue_order, 
                    data=df, ax=ax, width=0.7, fliersize=1., palette=palette)
        
        ax.set_yscale('log')
        ax.tick_params(axis="x", labelrotation=90, labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)

        ax.set_xlabel('Target Cell Type', fontsize=label_fontsize)
        ax.set_ylabel('Weight (pA)', fontsize=label_fontsize)
        ax.set_title(f'{descriptive_title}', fontsize=title_fontsize)
        # Apply shadings to each layer
        xticklabel = ax.get_xticklabels()
        borders = get_borders(xticklabel)
        # get the current ylim
        ylim = ax.get_ylim()
        # draw the borders
        draw_borders(ax, borders, ylim)
        
        # Make legend text larger
        ax.legend(loc='best', fontsize=tick_fontsize)
        
        if similarity_score is not None:
            # create text with white face color and black edge color
            ax.text(0.025, 0.925, f'S: {similarity_score:.2f}', transform=ax.transAxes, fontsize=14,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', 
                            edgecolor='black', alpha=0.7, linewidth=1.))

        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'{variable}.png'), dpi=300, transparent=False, bbox_inches='tight')
        plt.close()

    def pop_to_pop_weights_distribution(self, indices, variable='', voltage_scale=None):
        source_pop_names = other_v1_utils.pop_names(self.network)
        source_cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in source_pop_names]
        target_pop_names = other_v1_utils.pop_names(self.network)
        target_cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in target_pop_names]
        post_cell_types = [target_cell_types[i] for i in indices[:, 0]]
        pre_cell_types = [source_cell_types[i] for i in indices[:, 1]]

        if voltage_scale is not None:
            # Normalize the weights by the voltage scale
            initial_weights = self.model_variables_dict['Initial'][variable] * voltage_scale
            final_weights = self.model_variables_dict['Best'][variable] * voltage_scale
        else:
            initial_weights = self.model_variables_dict['Initial'][variable]
            final_weights = self.model_variables_dict['Best'][variable]

        df = pd.DataFrame({
            'Post_names': post_cell_types,
            'Pre_names': pre_cell_types,
            'Initial weight': initial_weights,
            'Final weight': final_weights,
        })

        # Melt DataFrame to long format
        df_melted = df.melt(id_vars=['Post_names', 'Pre_names'], value_vars=['Initial weight', 'Final weight'], 
                            var_name='Weight Type', value_name='Weight')
        df_melted['Weight'] = np.abs(df_melted['Weight'])
        # Create directory for saving plots
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}_distribution')
        os.makedirs(boxplots_dir, exist_ok=True)
        # Establish the order of the neuron types in the boxplots
        cell_type_order = np.sort(df['Pre_names'].unique())
        target_type_order = np.sort(df['Post_names'].unique())

        # Define the palette
        palette = {"Initial weight": "#87CEEB", "Final weight": "#FFA500"}
        # Create subplots
        num_pre_names = len(cell_type_order)
        num_columns = 4
        num_rows = (num_pre_names + num_columns - 1) // num_columns

        # Increase font sizes
        title_fontsize = 26
        axis_label_fontsize = 26
        xtick_fontsize = 20
        ytick_fontsize = 24
        legend_fontsize = 24

        fig, axes = plt.subplots(num_rows, num_columns, figsize=(24, 5 * num_rows), sharey=True)
        # Reduce the vertical spacing between subplots
        plt.subplots_adjust(hspace=0.1)  # Reduce vertical space between plots
        # Flatten the axes array and handle the first row separately
        axes = axes.flatten()
        for i, pre_name in enumerate(cell_type_order):            
            ax = axes[i] 
            subset_df = df_melted[df_melted['Pre_names'] == pre_name]
            similarity_score = compute_ks_statistics(subset_df, metric='Weight', min_n_sample=15)
            # subset_cell_type_order = np.sort(subset_df['Post_names'].unique())
            # Create boxplot for Initial and Final weights
            sns.boxplot(data=subset_df, x='Post_names', y='Weight', hue='Weight Type', 
                        order=target_type_order, ax=ax, palette=palette, 
                        width=0.7, fliersize=1.)
            # sns.violinplot(data=subset_df, x='Post_names', y='Weight', hue='Weight Type', order=subset_cell_type_order, ax=ax, palette=palette, width=0.7,
            #                split=True, inner="quart", gap=0.2)
            ax.set_title(f'Source Cell Type: {pre_name}', fontsize=title_fontsize)
            # ax.set_ylabel(r'$\vert$ Synaptic Weight (pA)$\vert$', fontsize=axis_label_fontsize)
            # ax.set_yscale('symlog', linthresh=0.001)
            ax.set_yscale('log')

            if i % num_columns == 0:  # First column
                ax.set_ylabel(r'$\vert$ Synaptic Weight (pA)$\vert$', fontsize=axis_label_fontsize)
            else:
                ax.set_ylabel('')

            # Show x-label only for the last row
            row_position = i // num_columns
            is_last_row = row_position == num_rows - 1 #or i >= num_pre_names - num_columns
            
            if is_last_row:
                ax.set_xlabel('Target Cell Type', fontsize=axis_label_fontsize)
            else:
                ax.set_xlabel('')

            # Make subplot borders thicker
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
                spine.set_color('black')

            bottom_limit = 0.01
            upper_limit = 100
            ax.set_ylim(bottom=bottom_limit, top=upper_limit)

            # Apply shadings to each layer
            xticklabel = ax.get_xticklabels()
            borders = get_borders(xticklabel)
            # get the current ylim
            ylim = ax.get_ylim()
            # draw the borders
            draw_borders(ax, borders, ylim)

            if not is_last_row:
                # remove the xticks
                ax.set_xticklabels([])	
            else:
                ax.tick_params(axis="x", labelsize=xtick_fontsize, rotation=90)

            ax.tick_params(axis="y", labelsize=ytick_fontsize)

            if i == 0:
                ax.legend(loc='lower left', fontsize=legend_fontsize)
            else:
                ax.get_legend().remove()

            if similarity_score is not None:
                ax.text(0.82, 0.95, f'S: {similarity_score:.2f}', transform=ax.transAxes, fontsize=21,
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='black',
                                  linewidth=1.))
        
        # Remove any unused subplots if the last grid position is empty
        for j in range(num_pre_names, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'{variable}.png'), dpi=300, transparent=False)
        plt.close()

    def pop_to_pop_weights_distribution_by_layer(self, indices, variable='', voltage_scale=None):
        source_pop_names = other_v1_utils.pop_names(self.network)
        source_cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in source_pop_names]
        target_pop_names = other_v1_utils.pop_names(self.network)
        target_cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in target_pop_names]
        post_cell_types = [target_cell_types[i] for i in indices[:, 0]]
        pre_cell_types = [source_cell_types[i] for i in indices[:, 1]]

        if voltage_scale is not None:
            # Normalize the weights by the voltage scale
            initial_weights = self.model_variables_dict['Initial'][variable] * voltage_scale
            final_weights = self.model_variables_dict['Best'][variable] * voltage_scale
        else:
            initial_weights = self.model_variables_dict['Initial'][variable]
            final_weights = self.model_variables_dict['Best'][variable]

        df = pd.DataFrame({
            'Post_names': post_cell_types,
            'Pre_names': pre_cell_types,
            'Initial weight': initial_weights,
            'Final weight': final_weights,
        })

        # Melt DataFrame to long format
        df_melted = df.melt(id_vars=['Post_names', 'Pre_names'], value_vars=['Initial weight', 'Final weight'], 
                            var_name='Weight Type', value_name='Weight')
        df_melted['Weight'] = np.abs(df_melted['Weight'])
        
        # Create directory for saving plots
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}_distribution')
        os.makedirs(boxplots_dir, exist_ok=True)
        
        # Establish the order of the neuron types in the boxplots
        cell_type_order = np.sort(df['Pre_names'].unique())
        target_type_order = np.sort(df['Post_names'].unique())
        
        # Define the palette
        palette = {"Initial weight": "#87CEEB", "Final weight": "#FFA500"}
        
        # Map variable names to descriptive titles
        title_mapping = {
            'sparse_recurrent_weights:0': 'Recurrent Synaptic Weights',
            'rest_of_brain_weights:0': 'Background Input Weights',
            'sparse_input_weights:0': 'LGN Input Weights'
        }
        # Get descriptive title or use original variable name if not in mapping
        descriptive_title = title_mapping.get(variable, variable)
        
        # Group cell types by layer
        layer_mapping = {}
        for cell_type in cell_type_order:
            if cell_type.startswith('L1'):
                layer = 'L1'
            elif cell_type.startswith('L2/3'):
                layer = 'L23'
            elif cell_type.startswith('L4'):
                layer = 'L4'
            elif cell_type.startswith('L5'):
                layer = 'L5'
            elif cell_type.startswith('L6'):
                layer = 'L6'
            else:
                layer = 'Other'
            
            if layer not in layer_mapping:
                layer_mapping[layer] = []
            layer_mapping[layer].append(cell_type)
        
        # Organize the plots by layer - one figure per layer
        layer_order = ['L1', 'L23', 'L4', 'L5', 'L6', 'Other']
        
        for layer in layer_order:
            if layer in layer_mapping and layer_mapping[layer]:
                cell_types_in_layer = layer_mapping[layer]
                num_cell_types = len(cell_types_in_layer)
                
                # Calculate grid dimensions - try to make it as square as possible
                num_cols = min(3, num_cell_types)  # Maximum 3 columns
                num_rows = (num_cell_types + num_cols - 1) // num_cols
                
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 5*num_rows), squeeze=False)
                axes = axes.flatten()
                
                # Create subplot for each cell type in this layer
                for i, cell_type in enumerate(cell_types_in_layer):
                    ax = axes[i]
                    subset_df = df_melted[df_melted['Pre_names'] == cell_type]
                    similarity_score = compute_ks_statistics(subset_df, metric='Weight', min_n_sample=15)
                    
                    # Create boxplot for Initial and Final weights
                    sns.boxplot(data=subset_df, x='Post_names', y='Weight', hue='Weight Type', 
                            order=target_type_order, ax=ax, palette=palette, width=0.7, fliersize=1.)
                    
                    ax.set_title(f'Source: {cell_type}', fontsize=14)
                    ax.set_yscale('log')
                    ax.set_ylim(bottom=0.01, top=100)
                    ax.set_ylabel(r'$\vert$ Weight (pA)$\vert$', fontsize=16)
                    ax.set_xlabel('Target Cell Type', fontsize=16)
                    ax.tick_params(axis="x", labelrotation=90, labelsize=14)
                    ax.tick_params(axis="y", labelsize=14)
                    
                    # Apply shadings to each layer
                    xticklabel = ax.get_xticklabels()
                    borders = get_borders(xticklabel)
                    ylim = ax.get_ylim()
                    draw_borders(ax, borders, ylim)
                    
                    # Add similarity score
                    if similarity_score is not None:
                        ax.text(0.85, 0.95, f'S: {similarity_score:.2f}', transform=ax.transAxes, 
                                fontsize=16,
                                verticalalignment='top', 
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, 
                                        edgecolor='black', linewidth=1))
                    
                    # Only show legend on first plot
                    if i == 0:
                        ax.legend(loc='lower left', fontsize=14)
                    else:
                        ax.get_legend().remove()
                
                # Remove any unused subplots
                for j in range(num_cell_types, len(axes)):
                    fig.delaxes(axes[j])
                
                # Add super title
                fig.suptitle(f'{descriptive_title} - {layer} Layer', fontsize=18, y=1.02)
                
                # Adjust layout
                plt.tight_layout()
                plt.savefig(os.path.join(boxplots_dir, f'{variable}_{layer}.png'), dpi=300, transparent=False, bbox_inches='tight')
                plt.close()
        
    def pop_to_pop_weights_analysis(self, indices, variable='', voltage_scale=None):
        source_pop_names = other_v1_utils.pop_names(self.network)
        source_cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in source_pop_names]
        target_pop_names = other_v1_utils.pop_names(self.network)
        target_cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in target_pop_names]
        post_cell_types = [target_cell_types[i] for i in indices[:, 0]]
        pre_cell_types = [source_cell_types[i] for i in indices[:, 1]]

        ### Initial Weight ###
        if voltage_scale is not None:
            # Normalize the weights by the voltage scale
            initial_weights = self.model_variables_dict['Initial'][variable] * voltage_scale
            final_weights = self.model_variables_dict['Best'][variable] * voltage_scale
        else:
            initial_weights = self.model_variables_dict['Initial'][variable]
            final_weights = self.model_variables_dict['Best'][variable]

        weight_changes = final_weights - initial_weights
        df = pd.DataFrame({'Post_names': post_cell_types, 
                            'Pre_names':pre_cell_types, 
                            'Initial weight': initial_weights, 
                            'Final weight': final_weights, 
                            'Weight Change': weight_changes})
        
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}')
        os.makedirs(boxplots_dir, exist_ok=True)
        
        # Increased fontsize for labels and ticks
        label_fontsize = 16
        tick_fontsize = 12
        title_fontsize = 18

        # Map variable names to descriptive titles
        title_mapping = {
            'sparse_recurrent_weights:0': 'Recurrent Weights',
            'rest_of_brain_weights:0': 'Background Input Weights',
            'sparse_input_weights:0': 'LGN Input Weights'
        }
        # Get descriptive title or use original variable name if not in mapping
        descriptive_title = title_mapping.get(variable, variable)

        # Plot for Initial Weight
        if not os.path.exists(os.path.join(self.logdir, 'Boxplots', variable, 'Initial_weight.png')):
            grouped_df = df.groupby(['Pre_names', 'Post_names'])['Initial weight'].mean().reset_index()
            # Create a pivot table to reshape the data for the heatmap
            pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Initial weight')
            # Plot heatmap
            fig = plt.figure(figsize=(12, 6))
            heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-20, vmax=20)
            plt.xticks(rotation=90, fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            plt.xlabel('Target Cell Type', fontsize=label_fontsize)
            plt.ylabel('Source Cell Type', fontsize=label_fontsize)
            plt.gca().set_aspect('equal')
            plt.title(f'Initial {descriptive_title}', fontsize=title_fontsize, fontweight='bold')
            # Create a separate color bar axis closer to the heatmap
            cbar = plt.colorbar(heatmap.collections[0], ax=plt.gca(), pad=0.025, aspect=20)
            cbar.ax.tick_params(labelsize=tick_fontsize)
            # Position the label above the colorbar
            cbar.ax.set_title('Weight (pA)', fontsize=label_fontsize, pad=10)

            # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
            plt.tight_layout()
            plt.savefig(os.path.join(boxplots_dir, f'Initial_weight.png'), dpi=300, transparent=True, bbox_inches='tight')
            plt.close()

        ### Final Weight ###
        grouped_df = df.groupby(['Pre_names', 'Post_names'])['Final weight'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Final weight')
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-20, vmax=20)
        plt.xticks(rotation=90, fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.xlabel('Target Cell Type', fontsize=label_fontsize)
        plt.ylabel('Source Cell Type', fontsize=label_fontsize)
        plt.gca().set_aspect('equal')
        plt.title(f'Final {descriptive_title}', fontsize=title_fontsize, fontweight='bold')
        # Plot color bar
        cbar = plt.colorbar(heatmap.collections[0], ax=plt.gca(), pad=0.025, aspect=20)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        # Position the label above the colorbar
        cbar.ax.set_title('Weight (pA)', fontsize=label_fontsize, pad=10)
        # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
        plt.tight_layout()  
        plt.savefig(os.path.join(boxplots_dir, f'Final_weight.png'), dpi=300, transparent=False, bbox_inches='tight')
        plt.close()

        ### Weight change ###
        grouped_df = df.groupby(['Pre_names', 'Post_names'])['Weight Change'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        try:
            pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Weight Change')
            # Plot heatmap
            plt.figure(figsize=(12, 6))
            heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0)
            plt.xticks(rotation=90, fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            plt.xlabel('Target Cell Type', fontsize=label_fontsize)
            plt.ylabel('Source Cell Type', fontsize=label_fontsize)
            plt.gca().set_aspect('equal')
            plt.title(f'Changes in {descriptive_title}', fontsize=title_fontsize, fontweight='bold')
            # Create a separate color bar axis closer to the heatmap
            cbar = plt.colorbar(heatmap.collections[0], ax=plt.gca(), pad=0.025, aspect=20)
            cbar.ax.tick_params(labelsize=tick_fontsize)
            # Position the label above the colorbar
            cbar.ax.set_title('Weight (pA)', fontsize=label_fontsize, pad=10)
            # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
            plt.tight_layout()
            plt.savefig(os.path.join(boxplots_dir, f'Weight_change.png'), dpi=300, transparent=False, bbox_inches='tight')
            plt.close()
        except:
            print('Skipping the plot for the weight change heatmap...')
            # raise the actual error
            print(grouped_df)
      
    def fano_factor(self, spikes, t_start=0.7, t_end=2.5, n_samples=100, analyze_core_only=True):
        
        # Trim spikes to the desired time window
        t_start_idx = int(t_start * 1000)
        t_end_idx = int(t_end * 1000)
        spikes = spikes[:, :, t_start_idx:t_end_idx, :]

        if analyze_core_only:
            # Isolate the core neurons
            # pop_names = other_v1_utils.pop_names(self.network, core_radius=self.flags.loss_core_radius, data_dir=self.flags.data_dir)
            core_mask = other_v1_utils.isolate_core_neurons(self.network, radius=self.flags.loss_core_radius, data_dir=self.flags.data_dir)
            n_core_neurons = np.sum(core_mask)
            # spikes = spikes[:, :, :, core_mask]
        else:
            n_core_neurons = spikes.shape[-1]
            core_mask = np.ones(n_core_neurons, dtype=bool)
            # pop_names = other_v1_utils.pop_names(self.network, data_dir=self.flags.data_dir)

        # Calculate the Fano Factor for the spikes
        pop_names = other_v1_utils.pop_names(self.network, data_dir=self.flags.data_dir)
        node_ei = np.array([pop_name[0] for pop_name in pop_names])
        # Get the IDs for excitatory neurons
        total_mask = np.logical_and(node_ei == 'e', core_mask)
        n_e_neurons = np.sum(total_mask)
        spikes = spikes[:, :, :, total_mask]
        node_id_e = np.arange(0, n_e_neurons)
        # Reshape the spikes tensor to have the shape (n_trials, seq_len, n_neurons)
        spikes = np.reshape(spikes, [spikes.shape[0]*spikes.shape[1], spikes.shape[2], spikes.shape[3]])
        n_trials = spikes.shape[0]
        # Pre-define bin sizes
        bin_sizes = np.logspace(-3, 0, 20)
        # using the simulation length, limit bin_sizes to define at least 5 bins
        bin_sizes_mask = bin_sizes < (t_end - t_start)/5
        bin_sizes = bin_sizes[bin_sizes_mask]
        # Vectorize the sampling process
        sample_size = 70
        sample_std = 30
        sample_counts = np.random.normal(sample_size, sample_std, n_samples).astype(int)
        # ensure that the sample counts are at least 15 and less than the number of neurons
        sample_counts = np.clip(sample_counts, 15, n_e_neurons)
        # trial_ids =np.random.choice(np.arange(n_trials), n_samples, replace=False)
        trial_ids = np.random.randint(n_trials, size=n_samples)
        
        # Generate Fano factors across random samples
        fanos = []
        for i in range(n_samples):
            random_trial_id = trial_ids[i]
            sample_num = sample_counts[i]
            sample_ids = np.random.choice(node_id_e, sample_num, replace=False)
            # selected_spikes = np.concatenate([spikes_timestamps[random_trial_id][np.isin(node_id, sample_ids), :]])
            # selected_spikes = selected_spikes[~np.isnan(selected_spikes)]
            # selected_spikes = new_spikes[random_trial_id][:, np.isin(node_id, sample_ids)]
            selected_spikes = spikes[random_trial_id][:, sample_ids]
            selected_spikes = np.sum(selected_spikes, axis=1)
            # if there are spikes use pop_fano
            if np.sum(selected_spikes) > 0:
                fano = pop_fano(selected_spikes, bin_sizes)
                fanos.append(fano)

        fanos = np.array(fanos)
        # mean_fano = np.mean(fanos, axis=0)
        return fanos, bin_sizes
        
    def fanos_figure(self, spikes, n_samples=100, spont_fano_duration=300, evoked_fano_duration=300, analyze_core_only=True, data_dir='Synchronization_data'):
        """
        Generate publication-quality Fano Factor figure comparing model and experimental data,
        with larger font sizes for better readability.
        """
        # Calculate fano factors for both sessions
        evoked_t_start_seconds = self.pre_delay / 1000 + 0.2
        evoked_t_end_seconds = evoked_t_start_seconds + evoked_fano_duration / 1000
        evoked_fanos, evoked_bin_sizes = self.fano_factor(spikes, t_start=evoked_t_start_seconds, t_end=evoked_t_end_seconds, n_samples=n_samples, analyze_core_only=analyze_core_only)
        
        spont_t_start_seconds = 0.2
        spont_t_end_seconds = spont_t_start_seconds + spont_fano_duration / 1000
        spontaneous_fanos, spont_bin_sizes = self.fano_factor(spikes, t_start=spont_t_start_seconds, t_end=spont_t_end_seconds, n_samples=n_samples, analyze_core_only=analyze_core_only)
    
        # Calculate mean, standard deviation, and SEM of the Fano factors
        evoked_fanos_mean = np.nanmean(evoked_fanos, axis=0)
        evoked_fanos_std = np.nanstd(evoked_fanos, axis=0)
        evoked_fanos_sem = evoked_fanos_std / np.sqrt(n_samples)
        spontaneous_fanos_mean = np.nanmean(spontaneous_fanos, axis=0)
        spontaneous_fanos_std = np.nanstd(spontaneous_fanos, axis=0)
        spontaneous_fanos_sem = spontaneous_fanos_std / np.sqrt(n_samples)
        
        # Find the frequency of maximum Fano factor
        evoked_max_fano = np.nanmax(evoked_fanos_mean)
        evoked_max_fano_freq = 1/(2*evoked_bin_sizes[np.nanargmax(evoked_fanos_mean)])
        spontaneous_max_fano = np.nanmax(spontaneous_fanos_mean)
        spontaneous_max_fano_freq = 1/(2*spont_bin_sizes[np.nanargmax(spontaneous_fanos_mean)])
    
        # Load experimental data
        evoked_exp_data_path = os.path.join(data_dir, 'Fano_factor_v1', f'v1_fano_running_{evoked_fano_duration}ms_evoked.npy')
        evoked_exp_fanos = np.load(evoked_exp_data_path, allow_pickle=True)
        spont_exp_data_path = os.path.join(data_dir, 'Fano_factor_v1', f'v1_fano_running_{spont_fano_duration}ms_spont.npy')
        spont_exp_fanos = np.load(spont_exp_data_path, allow_pickle=True)
        
        # Calculate statistics for experimental data
        evoked_exp_fanos_mean = np.nanmean(evoked_exp_fanos, axis=0)[:len(evoked_bin_sizes)]
        evoked_exp_fanos_std = np.nanstd(evoked_exp_fanos, axis=0)[:len(evoked_bin_sizes)]
        evoked_exp_fanos_sem = evoked_exp_fanos_std / np.sqrt(evoked_exp_fanos.shape[0])
        spont_exp_fanos_mean = np.nanmean(spont_exp_fanos, axis=0)[:len(spont_bin_sizes)]
        spont_exp_fanos_std = np.nanstd(spont_exp_fanos, axis=0)[:len(spont_bin_sizes)]
        spont_exp_fanos_sem = spont_exp_fanos_std / np.sqrt(spont_exp_fanos.shape[0])
        
        # Set color scheme
        model_evoked_color = '#2980b9'  # Blue
        model_spont_color = '#2980b9'   # Blue
        exp_evoked_color = '#2c3e50'          # Black
        exp_spont_color = '#e74c3c'            # Red
        poisson_color = '#7f7f7f'       # Gray
        
        # Font size settings - INCREASED
        TITLE_SIZE = 10      # Was 16
        AXIS_LABEL_SIZE = 18 # Was 14
        TICK_SIZE = 14       # Was 12
        LEGEND_SIZE = 16     # Was 12
        TEXT_SIZE = 16       # Was 12
        PANEL_LABEL_SIZE = 18 # Was 18
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(12, 5))  # Increased from (12, 5) for better visibility
        gs = plt.GridSpec(1, 2, wspace=0.15)
        
        # --- SUBPLOT A: EVOKED CONDITION ---
        ax1 = fig.add_subplot(gs[0])
        
        # Plot individual experimental trajectories with low opacity
        for row in range(evoked_exp_fanos.shape[0]):
            ax1.plot(evoked_bin_sizes, evoked_exp_fanos[row, :len(evoked_bin_sizes)], 
                        color='gray', alpha=0.15, linewidth=1.0, zorder=1)
        
        # Plot mean with error bars
        ax1.errorbar(evoked_bin_sizes, evoked_fanos_mean, yerr=evoked_fanos_sem, 
                    fmt='o-', color=model_evoked_color, linewidth=2, markersize=5, 
                    capsize=3, elinewidth=1.5, capthick=1.5, zorder=5,
                    label='Evoked Model')
        
        ax1.errorbar(evoked_bin_sizes, evoked_exp_fanos_mean, yerr=evoked_exp_fanos_sem, 
                    fmt='s-', color=exp_evoked_color, linewidth=2, markersize=5, 
                    capsize=3, elinewidth=1.5, capthick=1.5, zorder=4,
                    label='Evoked Experimental')
        
        # Add horizontal line at Fano=1 for Poisson reference
        ax1.axhline(y=1.0, color=poisson_color, linestyle='--', alpha=0.8, linewidth=2.0, zorder=2)
        ax1.text(0.75*evoked_bin_sizes[-1], 1.05, 'Poisson', color=poisson_color, fontsize=TEXT_SIZE)
        
        # Configure subplot
        ax1.set_xscale("log")
        ax1.set_xlabel('Bin Size (s)', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        ax1.set_ylabel('Fano Factor', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(True, linestyle='--', alpha=0.3, which='major')
        ax1.set_axisbelow(True)
        ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=LEGEND_SIZE)
        
        # Add panel label
        ax1.text(-0.12, 1.05, 'A', transform=ax1.transAxes, 
                    fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='bottom')
        
        # Set y-limits based on data
        upper_y_limit = max([2, max(np.max(evoked_fanos_mean), np.max(evoked_exp_fanos_mean)) * 1.2])
        ax1.set_ylim(0, upper_y_limit)
    
        ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        ax1.tick_params(axis='both', which='minor', labelsize=TICK_SIZE-2)
        
        # --- SUBPLOT B: SPONTANEOUS CONDITION ---
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        
        # Plot individual experimental trajectories with low opacity
        for row in range(spont_exp_fanos.shape[0]):
            ax2.plot(spont_bin_sizes, spont_exp_fanos[row, :len(spont_bin_sizes)], 
                        color='gray', alpha=0.15, linewidth=1.0, zorder=1)
        
        # Plot mean with error bars
        ax2.errorbar(spont_bin_sizes, spontaneous_fanos_mean, yerr=spontaneous_fanos_sem, 
                    fmt='o-', color=model_spont_color, linewidth=2, markersize=5, 
                    capsize=3, elinewidth=1.5, capthick=1.5, zorder=5,
                    label='Spontaneous Model')
        
        ax2.errorbar(spont_bin_sizes, spont_exp_fanos_mean, yerr=spont_exp_fanos_sem, 
                    fmt='s-', color=exp_spont_color, linewidth=2, markersize=5, 
                    capsize=3, elinewidth=1.5, capthick=1.5, zorder=4,
                    label='Spontaneous Experimental')
        
        # Add horizontal line at Fano=1 for Poisson reference
        ax2.axhline(y=1.0, color=poisson_color, linestyle='--', alpha=0.8, linewidth=2.0, zorder=2)
        ax2.text(0.75*spont_bin_sizes[-1], 1.05, 'Poisson', color=poisson_color, fontsize=TEXT_SIZE)
        
        # Configure subplot
        ax2.set_xscale("log")
        ax2.set_xlabel('Bin Size (s)', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(True, linestyle='--', alpha=0.3, which='major')
        ax2.set_axisbelow(True)
        ax2.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=LEGEND_SIZE)
        
        # Hide y-ticks for second subplot (since it shares y-axis with first)
        plt.setp(ax2.get_yticklabels(), visible=False)
        
        # Add panel label
        ax2.text(-0.12, 1.05, 'B', transform=ax2.transAxes, 
                    fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='bottom')
        
        # Set tick size for the second subplot
        ax2.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        ax2.tick_params(axis='both', which='minor', labelsize=TICK_SIZE-2)
        
        # Add suptitle with duration information if needed
        # fig.suptitle(f"Fano Factor Analysis ({evoked_fano_duration}ms windows)", 
        #              fontsize=TITLE_SIZE, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.join(self.images_dir, 'Fano_Factor'), exist_ok=True)
        output_file = os.path.join(self.images_dir, 'Fano_Factor', 
                                    f'V1_epoch_{self.current_epoch}_spont_{spont_fano_duration}ms_evoked_{evoked_fano_duration}ms.png')
        plt.savefig(output_file, dpi=300, transparent=False, bbox_inches='tight')
        plt.close()    

    def plot_populations_activity(self, v1_spikes):
        # Plot the mean firing rate of the population of neurons
        filename = f'Epoch_{self.current_epoch}'
        seq_len = v1_spikes.shape[1]
        Population_activity = PopulationActivity(n_neurons=self.n_neurons, network=self.network, 
                                                stimuli_init_time=self.pre_delay, stimuli_end_time=seq_len-self.post_delay, 
                                                image_path=self.images_dir, filename=filename, data_dir=self.flags.data_dir,
                                                core_radius=self.flags.plot_core_radius)
        Population_activity(v1_spikes, plot_core_only=True, bin_size=10)

    def plot_raster(self, x, v1_spikes, angle=0):
        seq_len = v1_spikes.shape[1]
        images_dir = os.path.join(self.images_dir, 'Raster_plots_OSI_DSI')
        os.makedirs(images_dir, exist_ok=True)
        graph = InputActivityFigure(
                                    self.network,
                                    self.flags.data_dir,
                                    images_dir,
                                    filename=f'Epoch_{self.current_epoch}_orientation_{angle}_degrees',
                                    frequency=self.flags.temporal_f,
                                    stimuli_init_time=self.pre_delay,
                                    stimuli_end_time=seq_len-self.post_delay,
                                    reverse=False,
                                    plot_core_only=True,
                                    core_radius=self.flags.plot_core_radius,
                                    )
        graph(x, v1_spikes)

    def plot_population_firing_rates_vs_tuning_angle(self, spikes, DG_angles, core_radius=400):
        # Save the spikes
        spikes_dir = os.path.join(self.images_dir, 'Spikes_OSI_DSI')
        os.makedirs(spikes_dir, exist_ok=True)

        # Isolate the core neurons
        core_mask = other_v1_utils.isolate_core_neurons(self.network, radius=self.flags.plot_core_radius, data_dir=self.flags.data_dir)
        spikes = spikes[:, :, :, core_mask]
        spikes = np.sum(spikes, axis=0).astype(np.float32)/self.flags.n_trials_per_angle
        seq_len = spikes.shape[1]

        tuning_angles = self.network['tuning_angle'][core_mask]
        for angle_id, angle in enumerate(DG_angles):
            firingRates = calculate_Firing_Rate(spikes[angle_id, :, :], stimulus_init=self.pre_delay, stimulus_end=seq_len-self.post_delay, temporal_axis=0)
            x = tuning_angles
            y = firingRates
            # Define bins for delta_angle
            bins = np.linspace(np.min(x), np.max(x), 50)
            # Compute average rates for each bin
            average_rates = []
            for i in range(len(bins)-1):
                mask = (x >= bins[i]) & (x < bins[i+1])
                average_rates.append(np.mean(y[mask]))
            # Create bar plot
            plt.figure(figsize=(10, 6))
            plt.bar(bins[:-1], average_rates, width=np.diff(bins))
            plt.xlabel('Tuning Angle')
            plt.ylabel('Average Rates')
            plt.title(f'Gratings Angle: {angle}')
            plt.savefig(os.path.join(spikes_dir, f'v1_spikes_angle_{angle}.png'))
            plt.close()

    def single_trial_callbacks(self, x, v1_spikes, y, bkg_noise=None):
        # Plot the population activity
        self.plot_populations_activity(v1_spikes)
        # Plot the raster plot
        self.plot_raster(x, v1_spikes, angle=y)

    def osi_dsi_analysis(self, v1_spikes, DG_angles):
        ### Power spectral analysis
        psd_analyzer = PSDAnalyzer(self.network, fs=1.0, analyze_core_only=True, population_average=True, normalize=False, normalize_by_n2=True, 
                                   normalize_by_rate=False, save_path=os.path.join(self.images_dir, 'Power_Spectrum'), data_dir=self.flags.data_dir)
        stimulus_duration = v1_spikes.shape[2]
        psd_analyzer(v1_spikes, t_spont_start=0, t_spont_end=self.pre_delay, t_evoked_start=self.pre_delay+0, t_evoked_end=stimulus_duration-self.post_delay)

        # Do the OSI/DSI analysis       
        boxplots_dir = os.path.join(self.images_dir, 'Boxplots_OSI_DSI')
        os.makedirs(boxplots_dir, exist_ok=True)
        fr_boxplots_dir = os.path.join(self.images_dir, f'Boxplots_OSI_DSI/Evoked_Rate(Hz)')
        os.makedirs(fr_boxplots_dir, exist_ok=True)
        spontaneous_boxplots_dir = os.path.join(self.images_dir, 'Boxplots_OSI_DSI/Spontaneous rate (Hz)')
        os.makedirs(spontaneous_boxplots_dir, exist_ok=True)
         
        # Fano factor analysis
        print('Fano factor analysis...')
        t_fano0 = time()
        self.fanos_figure(v1_spikes, n_samples=1000, spont_fano_duration=300, evoked_fano_duration=300, analyze_core_only=True)
        self.fanos_figure(v1_spikes, n_samples=1000, spont_fano_duration=1800, evoked_fano_duration=1800, analyze_core_only=True)
        print('Fanos figure saved in ', time() - t_fano0, ' seconds!\n')
        # Plot the tuning angle analysis
        self.plot_population_firing_rates_vs_tuning_angle(v1_spikes, DG_angles)
        # Estimate tuning parameters from the model neurons
        metrics_analysis = ModelMetricsAnalysis(v1_spikes, DG_angles, self.network, data_dir=self.flags.data_dir,
                                                drifting_gratings_init=self.pre_delay, drifting_gratings_end=self.pre_delay+self.flags.evoked_duration,
                                                spontaneous_init=0, spontaneous_end=self.pre_delay,
                                                core_radius=self.flags.plot_core_radius, df_directory=self.images_dir, save_df=True,
                                                neuropixels_df=self.flags.neuropixels_df)
        # Figure for OSI/DSI boxplots
        metrics_analysis(metrics=["Rate at preferred direction (Hz)", "OSI", "DSI"], directory=boxplots_dir, filename=f'Epoch_{self.current_epoch}')
        # Figure for Average firing rate boxplots
        metrics_analysis(metrics=["Evoked rate (Hz)"], directory=fr_boxplots_dir, filename=f'Epoch_{self.current_epoch}')   
        # Spontaneous rates figure
        metrics_analysis(metrics=['Spontaneous rate (Hz)'], directory=spontaneous_boxplots_dir, filename=f'Epoch_{self.current_epoch}') 


class Callbacks:
    def __init__(self, network, lgn_input, bkg_input, model, optimizer, flags, logdir, strategy, 
                metrics_keys, pre_delay=50, post_delay=50, checkpoint=None, model_variables_init=None, 
                save_optimizer=True, spontaneous_training=False):
        
        self.n_neurons = flags.neurons
        self.network = network
        self.lgn_input = lgn_input
        self.bkg_input = bkg_input
        if spontaneous_training:
            self.neuropixels_feature = 'Spontaneous rate (Hz)'
        else:
            self.neuropixels_feature = 'Evoked rate (Hz)'  
        self.model = model
        self.optimizer = optimizer
        self.flags = flags
        self.logdir = logdir
        self.strategy = strategy
        self.metrics_keys = metrics_keys
        self.pre_delay = pre_delay
        self.post_delay = post_delay
        self.step = 0
        self.step_running_time = []
        self.model_variables_dict = model_variables_init
        self.initial_metric_values = None
        self.summary_writer = tf.summary.create_file_writer(self.logdir)
        with open(os.path.join(self.logdir, 'config.json'), 'w') as f:
            json.dump(flags.flag_values_dict(), f, indent=4)

        if checkpoint is None:
            if save_optimizer:
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            else:
                checkpoint = tf.train.Checkpoint(model=model)
            self.min_val_loss = float('inf')
            self.no_improve_epochs = 0
            self.checkpoint_epochs = 0
            # create a dictionary to save the values of the metric keys after each epoch
            self.epoch_metric_values = {key: [] for key in self.metrics_keys}
            self.epoch_metric_values['sync'] = []
        else:
            # Load epoch_metric_values and min_val_loss from the file
            if os.path.exists(os.path.join(self.logdir, 'train_end_data.pkl')):
                with open(os.path.join(self.logdir, 'train_end_data.pkl'), 'rb') as f:
                    data_loaded = pkl.load(f)
                self.epoch_metric_values = data_loaded['epoch_metric_values']
                self.min_val_loss = data_loaded['min_val_loss']
                self.no_improve_epochs = data_loaded['no_improve_epochs']
                self.checkpoint_epochs = len(data_loaded['epoch_metric_values']['train_loss'])
            elif os.path.exists(os.path.join(os.path.dirname(flags.restore_from), 'train_end_data.pkl')):
                with open(os.path.join(os.path.dirname(flags.restore_from), 'train_end_data.pkl'), 'rb') as f:
                    data_loaded = pkl.load(f)
                self.epoch_metric_values = data_loaded['epoch_metric_values']
                self.min_val_loss = data_loaded['min_val_loss']
                self.no_improve_epochs = data_loaded['no_improve_epochs']
                self.checkpoint_epochs = len(data_loaded['epoch_metric_values']['train_loss'])
            else:
                print('No train_end_data.pkl file found. Initializing...')
                self.epoch_metric_values = {key: [] for key in self.metrics_keys}
                self.epoch_metric_values['sync'] = []
                self.min_val_loss = float('inf')
                self.no_improve_epochs = 0
                self.checkpoint_epochs = 0

        self.total_epochs = flags.n_runs * flags.n_epochs + self.checkpoint_epochs
        # Manager for the best model
        self.best_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir + '/Best_model', max_to_keep=1
        )
        self.latest_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir + '/latest', max_to_keep=1
        )
        # Manager for osi/dsi checkpoints 
        self.epoch_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir + '/Intermediate_checkpoints', max_to_keep=5
        )

    def on_train_begin(self):
        print("---------- Training started at ", dt.datetime.now().strftime('%d-%m-%Y %H:%M'), ' ----------\n')
        self.train_start_time = time()
        # self.epoch = self.flags.run_session * self.flags.n_epochs
        self.epoch = self.checkpoint_epochs

    def on_train_end(self, metric_values, normalizers=None):
        self.train_end_time = time()
        self.final_metric_values = metric_values
        print("\n ---------- Training ended at ", dt.datetime.now().strftime('%d-%m-%Y %H:%M'), ' ----------\n')
        print(f"Total time spent: {self.train_end_time - self.train_start_time:.2f} seconds")
        print(f"Average step time: {np.mean(self.step_running_time):.2f} seconds\n")
        # Determine the maximum key length for formatting the table
        max_key_length = max(len(key) for key in self.metrics_keys)

        # Start of the Markdown table
        print(f"| {'Metric':<{max_key_length}} | {'Initial Value':<{max_key_length}} | {'Final Value':<{max_key_length}} |")
        print(f"|{'-' * (max_key_length + 2)}|{'-' * (max_key_length + 2)}|{'-' * (max_key_length + 2)}|")

        n_metrics = len(self.initial_metric_values)//2
        for initial, final, key in zip(self.initial_metric_values[n_metrics:], self.final_metric_values[n_metrics:], self.metrics_keys[n_metrics:]):
            print(f"| {key:<{max_key_length}} | {initial:<{max_key_length}.3f} | {final:<{max_key_length}.3f} |")

        # Save epoch_metric_values and min_val_loss to a file
        data_to_save = {
            'epoch_metric_values': self.epoch_metric_values,
            'min_val_loss': self.min_val_loss,
            'no_improve_epochs': self.no_improve_epochs
        }
        if normalizers is not None:
            data_to_save['v1_ema'] = normalizers['v1_ema']

        with open(os.path.join(self.logdir, 'train_end_data.pkl'), 'wb') as f:
            pkl.dump(data_to_save, f)

        if self.flags.n_runs > 1:
            self.save_intermediate_checkpoint()

    def on_epoch_start(self):
        self.epoch += 1
        # self.step_counter.assign_add(1)
        self.epoch_init_time = time()
        date_str = dt.datetime.now().strftime('%d-%m-%Y %H:%M')
        print(f'Epoch {self.epoch:2d}/{self.total_epochs} @ {date_str}')
        tf.print(f'\nEpoch {self.epoch:2d}/{self.total_epochs} @ {date_str}')

    def on_epoch_end(self, x, v1_spikes, y, metric_values, bkg_noise=None, verbose=True,
                    x_spont=None, v1_spikes_spont=None):
        
        if self.flags.dtype != 'float32':
            v1_spikes = v1_spikes.numpy().astype(np.float32)
            x = x.numpy().astype(np.float32)
            y = y.numpy().astype(np.float32)
            if x_spont is not None:
                x_spont = x_spont.numpy().astype(np.float32)
                v1_spikes_spont = v1_spikes_spont.numpy().astype(np.float32)
        else:
            v1_spikes = v1_spikes.numpy()
            x = x.numpy()
            y = y.numpy()
            if x_spont is not None:
                x_spont = x_spont.numpy()
                v1_spikes_spont = v1_spikes_spont.numpy()

        self.step = 0
        if self.initial_metric_values is None:
            self.initial_metric_values = metric_values
        
        if verbose:
            print_str = f'  Validation:\n' 
            val_values = metric_values[len(metric_values)//2:]
            print_str += '    ' + compose_str(val_values) 
            print(print_str)
            for gpu_id in range(len(self.strategy.extended.worker_devices)):
                printgpu(gpu_id=gpu_id)

        # val_classification_loss = metric_values[6] - metric_values[8] - metric_values[9] 
        # metric_values.append(val_classification_loss)
        # self.epoch_metric_values = {key: value + [metric_values[i]] for i, (key, value) in enumerate(self.epoch_metric_values.items())}
        for i, (key, value) in enumerate(self.epoch_metric_values.items()):
            if key not in ['sync']:
                self.epoch_metric_values[key] = value + [metric_values[i]]

        if 'val_loss' in self.metrics_keys:
            val_loss_index = self.metrics_keys.index('val_loss')
            val_loss_value = metric_values[val_loss_index]
        else:
            val_loss_index = self.metrics_keys.index('train_loss')
            val_loss_value = metric_values[val_loss_index]

        self.plot_losses_curves()
        
        # # save latest model every 10 epochs
        # if self.epoch % 10 == 0:
        #     self.save_latest_model()    

        if val_loss_value < self.min_val_loss:
            self.min_val_loss = val_loss_value
            # if self.no_improve_epochs > 50: # plot the best model results if there has been at least 50 epochs from the last best model
            self.no_improve_epochs = 0
            self.save_best_model()
            self.plot_mean_firing_rate_boxplot(v1_spikes, y)

            if v1_spikes_spont is not None:
                self.plot_spontaneous_boxplot(v1_spikes_spont, y)
                self.composed_raster(x, v1_spikes, x_spont, v1_spikes_spont, y)
                self.composed_raster(x, v1_spikes, x_spont, v1_spikes_spont, y, plot_core_only=False)
                # self.plot_lgn_activity(x, x_spont)
                # self.plot_populations_activity(v1_spikes, v1_spikes_spont)
            else:
                self.plot_raster(x, v1_spikes, y)
            
            self.model_variables_dict['Best'] = {
                var.name: var.numpy() if len(var.shape) == 1 else var[:, 0].numpy()
                for var in self.model.trainable_variables
            }

        else:
            self.no_improve_epochs += 1
           
        with self.summary_writer.as_default():
            for k, v in zip(self.metrics_keys, metric_values):
                tf.summary.scalar(k, v, step=self.epoch)

        # EARLY STOPPING CONDITIONS
        if (0 < self.flags.max_time < (time() - self.epoch_init_time) / 3600):
            print(f'[ Maximum optimization time of {self.flags.max_time:.2f}h reached ]')
            stop = True
        elif self.no_improve_epochs >= 500:
            print("Early stopping: Validation loss has not improved for 500 epochs.")
            stop = True  
        else:
            stop = False

        return stop

    def on_step_start(self):
        self.step += 1
        self.step_init_time = time()
        # # reset the gpu memory stat
        # tf.config.experimental.reset_memory_stats('GPU:0')

    def on_step_end(self, train_values, y, verbose=True):
        self.step_running_time.append(time() - self.step_init_time)
        if verbose:
            print_str = f'  Step {self.step:2d}/{self.flags.steps_per_epoch}\n'
            print_str += '    ' + compose_str(train_values)
            print(print_str)
            tf.print(print_str)
            print(f'    Step running time: {time() - self.step_init_time:.2f}s')
            for gpu_id in range(len(self.strategy.extended.worker_devices)):
                printgpu(gpu_id=gpu_id)
         
    def save_intermediate_checkpoint(self):
        # Save the checkpoint to reload weights in the osi_dsi_estimator
        p = self.epoch_manager.save(checkpoint_number=self.epoch)
        print(f'Checkpoint model saved in {p}\n')
        
    def save_latest_model(self):
        try:
            p = self.latest_manager.save(checkpoint_number=self.epoch)
            print(f'Latest model saved in {p}\n')    
        except:
            print("Saving failed. Maybe next time?")    

    def save_best_model(self):
        # self.step_counter.assign_add(1)
        print(f'[ Saving the model at epoch {self.epoch} ]')
        try:
            p = self.best_manager.save(checkpoint_number=self.epoch)
            print(f'Model saved in {p}\n')
        except:
            print("Saving failed. Maybe next time?")

    def plot_losses_curves(self):        
        # Define labels and components
        labels = ['val_loss', 'val_rate_loss', 'val_voltage_loss', 'val_regularizer_loss', 'val_osi_dsi_loss', 'val_sync_loss']
        component_labels = [l for l in labels if l != 'val_loss']
        
        # Create descriptive names for the labels
        label_display_names = {
            'val_rate_loss': 'Rate Loss',
            'val_voltage_loss': 'Voltage Reg.',
            'val_regularizer_loss': 'Weight Reg.',
            'val_osi_dsi_loss': 'OSI/DSI Loss',
            'val_sync_loss': 'Sync. Loss',
            'val_loss': 'Total Loss'
        }
        
        # Create directory for loss curves
        images_dir = os.path.join(self.logdir, 'Loss_curves')
        os.makedirs(images_dir, exist_ok=True)
        
        # Normalize loss components
        normalized_data = {}
        for label in labels:
            if label != 'val_loss' and label in self.epoch_metric_values:
                values = np.array(self.epoch_metric_values[label])
                if len(values) > 0 and values[0] != 0:  # Avoid division by zero
                    normalized_data[label] = values / values[0]
        
        if not normalized_data:  # If no valid data yet
            return
        
        # Create epoch indices
        epochs = np.arange(1, len(next(iter(normalized_data.values()))) + 1)
                
        # Create figure with side-by-side layout
        fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.2)
        
        # Use a professional color palette (colorblind-friendly)
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(component_labels)))
        
        # --- SUBPLOT A: NORMALIZED LOSS COMPONENTS ---
        ax1 = fig.add_subplot(gs[0])
        
        # Plot normalized loss components
        for i, label in enumerate([l for l in component_labels if l in normalized_data]):
            display_name = label_display_names.get(label, label)
            ax1.plot(epochs, normalized_data[label], label=display_name, color=colors[i], linewidth=2.5, alpha=0.8)
        
        # Configure first subplot
        ax1.set_xlabel('Training Epoch', fontweight='bold', fontsize=16)
        ax1.set_ylabel('Relative Loss (w.r.t. Initial Value)', fontweight='bold', fontsize=16)
        ax1.set_yscale('log')
        ax1.grid(True, linestyle='--', alpha=0.3, which='both')
        ax1.set_axisbelow(True)  # Place grid behind data
        
        # Add minor grid lines for log scale
        ax1.grid(which='minor', linestyle=':', alpha=0.2)
        
        # Create legend with columns
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
                ncol=3, frameon=True, fancybox=True, shadow=True, fontsize=13)
        
        # Add panel label
        ax1.text(-0.12, 1.05, 'A', transform=ax1.transAxes, 
                fontsize=20, fontweight='bold', va='bottom')
        
        # Increase tick label size
        ax1.tick_params(axis='both', which='major', labelsize=14)
        
        # --- SUBPLOT B: STACKED AREA CHART ---
        ax2 = fig.add_subplot(gs[1])
        
        # Get data for stacked area plot
        valid_component_labels = [l for l in component_labels if l in self.epoch_metric_values]
        data = np.array([self.epoch_metric_values[label] for label in valid_component_labels])
        
        # Create stacked area plot
        areas = ax2.stackplot(epochs, data, labels=[label_display_names[lbl] for lbl in valid_component_labels], 
                            colors=colors[:len(valid_component_labels)], alpha=0.8, edgecolor='white', linewidth=0.3)
        
        # Add total loss line
        if 'val_loss' in self.epoch_metric_values:
            ax2.plot(epochs, self.epoch_metric_values['val_loss'], 
                    color='black', linewidth=3, linestyle='-', 
                    label=label_display_names['val_loss'])
        
        # Configure second subplot
        ax2.set_xlabel('Training Epoch', fontweight='bold', fontsize=16)
        ax2.set_ylabel('Loss Components', fontweight='bold', fontsize=16)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_axisbelow(True)
        
        # Format y-axis tick labels to be more readable
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        # Set upper limit based on data
        max_y = np.sum(data, axis=0).max() * 1.1 if data.size > 0 else 1
        ylimit = min(max_y, 5)  # Adjust as needed, capped at 5
        ax2.set_ylim(0, ylimit)
        
        # More elegant break symbol if needed
        if max_y > ylimit:
            break_size = 0.015
            kwargs = dict(transform=ax2.transAxes, color='black', clip_on=False, linewidth=1.5)
            ax2.plot((-break_size, +break_size), (1 - break_size, 1 + break_size), **kwargs)
            ax2.plot((-break_size, +break_size), (1 - 2*break_size, 1), **kwargs)
            ax2.plot((1 - break_size, 1 + break_size), (1 - break_size, 1 + break_size), **kwargs)
            ax2.plot((1 - break_size, 1 + break_size), (1 - 2*break_size, 1), **kwargs)
        
        # Increase tick label size
        ax2.tick_params(axis='both', which='major', labelsize=14)
        
        # Add annotations for important components in the stack
        if epochs.size > 0:  # Check if we have epochs to plot
            middle_epoch = len(epochs) // 2
            for i, (label, area) in enumerate(zip(valid_component_labels, areas)):
                # Find y-coordinate in the middle of each area at the middle epoch
                if i == 0:
                    y_pos = data[i][middle_epoch] / 2
                else:
                    y_pos = sum(data[:i, middle_epoch]) + data[i][middle_epoch] / 2
                
                # Only label areas that are visually significant
                if max(data[i]) > 0.1: #* ylimit:  # Threshold for labeling
                    # Add a white background to ensure text visibility
                    bbox_props = dict(
                        boxstyle="round,pad=0.3", 
                        fc="white", 
                        ec="none", 
                        alpha=0.6
                    )
                    
                    # Create annotation with high z-order and background
                    ax2.annotate(
                        label_display_names[label].split()[0],  # Use first word only
                        xy=(middle_epoch, y_pos),
                        ha='center', va='center',
                        color='black',  # Use black for contrast against white background
                        fontweight='bold', fontsize=14,  # Increased from 12 to 14
                        bbox=bbox_props,  # Add white background
                        zorder=1000  # Very high z-order to ensure it's on top
                    )
        
        # Add panel label
        ax2.text(-0.12, 1.05, 'B', transform=ax2.transAxes, 
                fontsize=20, fontweight='bold', va='bottom')
        
        # Legend for total loss
        if 'val_loss' in self.epoch_metric_values:
            total_loss_legend = ax2.legend([ax2.get_lines()[0]], [label_display_names['val_loss']], 
                                        loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=13)
            ax2.add_artist(total_loss_legend)
        
        # Tight layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(images_dir, f'losses_curves.png'), dpi=300, transparent=False, bbox_inches='tight')
        plt.close()
    
    def plot_raster(self, x, v1_spikes, y):
        seq_len = v1_spikes.shape[1]
        images_dir = os.path.join(self.logdir, 'Raster_plots')
        os.makedirs(images_dir, exist_ok=True)
        graph = InputActivityFigure(
                                    self.network,
                                    self.flags.data_dir,
                                    images_dir,
                                    filename=f'Epoch_{self.epoch}',
                                    frequency=self.flags.temporal_f,
                                    stimuli_init_time=self.pre_delay,
                                    stimuli_end_time=seq_len-self.post_delay,
                                    reverse=False,
                                    plot_core_only=True,
                                    core_radius=self.flags.plot_core_radius,
                                    )
        graph(x, v1_spikes)

    def composed_raster(self, x, v1_spikes, x_spont, v1_spikes_spont, y, plot_core_only=True):
        # concatenate the normal and spontaneous arrays
        x = np.concatenate((x_spont, x), axis=1)
        v1_spikes = np.concatenate((v1_spikes_spont, v1_spikes), axis=1)
        seq_len = v1_spikes.shape[1]
        images_dir = os.path.join(self.logdir, 'Raster_plots')
        if plot_core_only:
            images_dir = os.path.join(images_dir, 'Core_only')
        else:
            images_dir = os.path.join(images_dir, 'Full')
        os.makedirs(images_dir, exist_ok=True)
        graph = InputActivityFigure(
                                    self.network,
                                    self.flags.data_dir,
                                    images_dir,
                                    filename=f'Epoch_{self.epoch}_complete',
                                    frequency=self.flags.temporal_f,
                                    stimuli_init_time=int(seq_len/2) + self.pre_delay,
                                    stimuli_end_time=seq_len - self.post_delay,
                                    reverse=False,
                                    plot_core_only=plot_core_only,
                                    core_radius=self.flags.plot_core_radius,
                                    )
        graph(x, v1_spikes)

    def plot_lgn_activity(self, x, x_spont):
        x = x[0, :, :]
        x_spont = x_spont[0, :, :]
        x = np.concatenate((x_spont, x), axis=0)
        x_mean = np.mean(x, axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(x_mean)
        plt.title('Mean input activity')
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean input activity')
        os.makedirs(os.path.join(self.logdir, 'Populations activity'), exist_ok=True)
        plt.savefig(os.path.join(self.logdir, 'Populations activity', f'LGN_population_activity_epoch_{self.epoch}.png'))
        plt.close()

    def plot_populations_activity(self, v1_spikes, v1_spikes_spont):
        v1_spikes = np.concatenate((v1_spikes_spont, v1_spikes), axis=1)
        seq_len = v1_spikes.shape[1]
        # Plot the mean firing rate of the population of neurons
        filename = f'Epoch_{self.epoch}'
        Population_activity = PopulationActivity(n_neurons=self.n_neurons, network=self.network, 
                                                stimuli_init_time=self.pre_delay, stimuli_end_time=seq_len-self.post_delay, 
                                                image_path=self.logdir, filename=filename, data_dir=self.flags.data_dir,
                                                core_radius=self.flags.plot_core_radius)
        Population_activity(v1_spikes, plot_core_only=True, bin_size=10)

    def plot_mean_firing_rate_boxplot(self, v1_spikes, y):
        seq_len = v1_spikes.shape[1]
        DG_angles = y
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{self.neuropixels_feature}')
        os.makedirs(boxplots_dir, exist_ok=True)        
        if self.neuropixels_feature == "Evoked rate (Hz)":
            metrics_analysis = ModelMetricsAnalysis(v1_spikes, DG_angles, self.network, data_dir=self.flags.data_dir, 
                                                    drifting_gratings_init=self.pre_delay, drifting_gratings_end=seq_len-self.post_delay,
                                                    core_radius=self.flags.plot_core_radius, df_directory=self.logdir, save_df=False,
                                                    neuropixels_df=self.flags.neuropixels_df) 
        elif self.neuropixels_feature == 'Spontaneous rate (Hz)':
            metrics_analysis = ModelMetricsAnalysis(v1_spikes, DG_angles, self.network, data_dir=self.flags.data_dir, 
                                                    spontaneous_init=self.pre_delay, spontaneous_end=seq_len-self.post_delay,
                                                    core_radius=self.flags.plot_core_radius, df_directory=self.logdir, save_df=False,
                                                    neuropixels_df=self.flags.neuropixels_df) 
        # Figure for Average firing rate boxplots      
        metrics_analysis(metrics=[self.neuropixels_feature], directory=boxplots_dir, filename=f'Epoch_{self.epoch}')    
                
    def plot_spontaneous_boxplot(self, v1_spikes, y):
        DG_angles = y
        seq_len = v1_spikes.shape[1]
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/Spontaneous')
        os.makedirs(boxplots_dir, exist_ok=True)
        metrics_analysis = ModelMetricsAnalysis(v1_spikes, DG_angles, self.network, data_dir=self.flags.data_dir, 
                                                spontaneous_init=self.pre_delay, spontaneous_end=seq_len-self.post_delay,
                                                core_radius=self.flags.plot_core_radius, df_directory=self.logdir, save_df=False,
                                                neuropixels_df=self.flags.neuropixels_df)
        # Figure for Average firing rate boxplots
        metrics_analysis(metrics=['Spontaneous rate (Hz)'], directory=boxplots_dir, filename=f'Epoch_{self.epoch}')
