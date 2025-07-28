# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:28:39 2022

@author: UX325
"""

import os
import sys
import pandas as pd
import numpy as np
# import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
sys.path.append(os.path.join(os.getcwd(), "v1_model_utils"))
import other_v1_utils
from scipy.stats import ks_2samp
from scipy.stats import f as f_distribution
from scipy.optimize import curve_fit
from numba import njit
import shutil

mpl.style.use('default')
# rd = np.random.RandomState(seed=42)

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

def calculate_Firing_Rate(z, stimulus_init=500, stimulus_end=2500, temporal_axis=2):
    # Select the relevant portion of the data along the temporal axis
    dg_spikes = np.take(z, range(stimulus_init, stimulus_end), axis=temporal_axis)   
    # Calculate the mean along the temporal axis
    mean_dg_spikes = np.mean(dg_spikes, axis=temporal_axis)
    # Sum along the temporal axis and divide by the duration in seconds to get the firing rate
    mean_firing_rates = mean_dg_spikes * 1000
    
    return mean_firing_rates

def calculate_OSI_DSI(firingRates, network, session='drifting_gratings', DG_angles=range(0,360, 45), 
                      n_selected_neurons=None, core_radius=None, remove_zero_rate_neurons=False, 
                      directory='', save_df=False):
    
    # Get the pop names of the neurons
    if n_selected_neurons is not None:
        pop_names = other_v1_utils.pop_names(network, n_selected_neurons=n_selected_neurons) 
    elif core_radius is not None and core_radius > 0:
        pop_names = other_v1_utils.pop_names(network, core_radius=core_radius)
    else:
        pop_names = other_v1_utils.pop_names(network)

    # Get the number of neurons and DG angles
    n_neurons = len(pop_names)
    node_ids = np.arange(n_neurons)
    # Get the firing rates for every neuron and DG angle
    # all_rates = np.array([g["Ave_Rate(Hz)"] for _, g in rates_df.groupby("DG_angle")]).T
    n_trials, n_angles, n_neurons = firingRates.shape
    all_direction_rates = np.mean(firingRates, axis=0)
    average_all_direction_rates = np.mean(all_direction_rates, axis=0)
    
    # Save the results in a dataframe
    if os.path.exists(os.path.join(directory, f"v1_features_df.csv")):
        osi_dsi_df = pd.read_csv(os.path.join(directory, f"v1_features_df.csv"), sep=" ")
    else:
        osi_dsi_df = pd.DataFrame()
        osi_dsi_df["node_id"] = node_ids
        osi_dsi_df["pop_name"] = pop_names

    if session == 'drifting_gratings':
        # Find the preferred DG angle for each neuron
        if n_trials >= 8:
            TuningAngleEstimation = PreferredTuningAngleAnalysis(firing_rates=firingRates, orientations=DG_angles, 
                                                                preferred_orientations=network['tuning_angle'])
            new_tuning_angles, preferred_angle_rates = TuningAngleEstimation.calculate_tuning_angle()
            osi_dsi_df["max_mean_rate(Hz)"] = preferred_angle_rates
            osi_dsi_df["preferred_angle"] = new_tuning_angles
        else:
            osi_dsi_df["preferred_angle"] = DG_angles[np.argmax(all_direction_rates, axis=0)]
            osi_dsi_df["max_mean_rate(Hz)"] = np.max(all_direction_rates, axis=0)

        # Calculate the DSI and OSI
        if n_angles >= 8:
            phase_rad = np.deg2rad(DG_angles)
            # Ensure phase_rad is a 2D array with shape (8, 1) for broadcasting
            phase_rad = phase_rad[:, np.newaxis]
            denominator = np.sum(all_direction_rates, axis=0)
            dsi = np.where(denominator != 0, 
                    np.abs((all_direction_rates * np.exp(1j * phase_rad)).sum(axis=0)) / denominator, 
                    np.nan)
            osi = np.where(denominator != 0,
                        np.abs((all_direction_rates * np.exp(2j * phase_rad)).sum(axis=0)) / denominator,
                        np.nan)        
            osi_dsi_df['OSI'] = osi
            osi_dsi_df['DSI'] = dsi
        else:
            osi_dsi_df['OSI'] = np.nan
            osi_dsi_df['DSI'] = np.nan

        # Calculate the average firing rate
        osi_dsi_df['Ave_Rate(Hz)'] = average_all_direction_rates
        if remove_zero_rate_neurons:
            osi_dsi_df = osi_dsi_df[osi_dsi_df["Ave_Rate(Hz)"] != 0]

    elif session == 'spontaneous':
        osi_dsi_df['firing_rate_sp'] = average_all_direction_rates
        if remove_zero_rate_neurons:
            osi_dsi_df = osi_dsi_df[osi_dsi_df["firing_rate_sp"] != 0]

    if save_df:
        os.makedirs(directory, exist_ok=True)
        osi_dsi_df.to_csv(os.path.join(directory, f"v1_features_df.csv"), sep=" ", index=False)

    return osi_dsi_df

def compute_ks_statistics(df, metric='Ave_Rate(Hz)', data_source1='V1 GLIF model', data_source2='Neuropixels', min_n_sample=15):
    """
    Compute the Kolmogorov-Smirnov statistic and similarity scores for each cell type in the dataframe.
    Parameters:
    - df: pd.DataFrame, contains data with columns 'data_type' and 'Ave_Rate(Hz)', and indexed by cell type.
    Returns:
    - mean_similarity_score: float, the mean of the similarity scores computed across all cell types.
    """
    # Get unique cell types
    # cell_types = df.index.unique()
    cell_types = df['cell_type'].unique()
    # Initialize a dictionary to store the results
    ks_results = {}
    similarity_scores = {}
    # Iterate over cell types
    for cell_type in cell_types:
        # Filter data for current cell type from two different data types
        # df1 = df.loc[(df.index == cell_type) & (df['data_type'] == 'V1/LM GLIF model'), metric]
        # df2 = df.loc[(df.index == cell_type) & (df['data_type'] == 'Neuropixels'), metric]
        df1 = df.loc[(df['cell_type'] == cell_type) & (df['data_type'] == data_source1), metric]
        df2 = df.loc[(df['cell_type'] == cell_type) & (df['data_type'] == data_source2), metric]
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

def get_borders(ticklabel):
    prev_layer = "1"
    borders = [-0.5]
    for i in ticklabel:
        x = i.get_position()[0]
        text = i.get_text()
        if text[1] != prev_layer:
            borders.append(x - 0.5)
            prev_layer = text[1]
    borders.append(x + 0.5)
    return borders

def draw_borders(ax, borders, ylim):
    for i in range(0, len(borders), 2):
        w = borders[i + 1] - borders[i]
        h = ylim[1] - ylim[0]
        ax.add_patch(
            Rectangle((borders[i], ylim[0]), w, h, alpha=0.08, color="k", zorder=-10)
        )
    return ax   

@njit
def ang_dir(angle1, angle2):
    """
    Calculate the minimum angular distance between two angles.
    Returns the smaller of the two possible angle differences (clockwise or counterclockwise).
    All angles are in degrees.
    
    Parameters:
    -----------
    angle1, angle2 : float or numpy.ndarray
        Angles in degrees
        
    Returns:
    --------
    float or numpy.ndarray
        Minimum angular distance between the angles (0-180 degrees)
    """
    # Ensure angles are in the range [0, 360)
    angle1 = np.mod(angle1, 360)
    angle2 = np.mod(angle2, 360)
    
    # Calculate absolute difference
    diff = np.abs(angle1 - angle2)
    
    # Compare with the complementary angle (360 - diff)
    # and take the minimum - this avoids the list construction and axis parameter
    return np.minimum(diff, 360 - diff)

@njit
def double_gaussian(theta, C, Rp, Rn, theta_pref, sigma):
    # According to Mazurek, M., Kager, M., & Van Hooser, S. D. (2014). 
    # Robust quantification of orientation selectivity and direction selectivity. 
    # Frontiers in neural circuits, 8, 92.
    # the best method to estimate tuning parameters from orientation and direction
    # responses if to fit them with a double gaussian function.
    delta_theta = ang_dir(theta, theta_pref) # restrict the angle to be between 0 and 180
    delta_theta2 = ang_dir(theta, theta_pref-180)
    denominator = 2 * sigma**2
    term1 = Rp * np.exp(-(delta_theta**2) / denominator)
    term2 = Rn * np.exp(-(delta_theta2**2) / denominator)
    return C + term1 + term2


class PreferredTuningAngleAnalysis:
    def __init__(self, firing_rates, orientations, preferred_orientations):
        """
        This class determines the preferred tuning angle of each neuron in the network according to their responses 
        to different orientations. The preferred tuning angle is determined by fitting the responses to a double
        gaussian function and finding the peak of the fit. Responsive neurons are identified using the Hotelling T2 test.
        For more details check:
        Mazurek, M., Kager, M., & Van Hooser, S. D. (2014). Robust quantification of orientation selectivity and
        direction selectivity. Frontiers in neural circuits, 8, 92.
        """
        self.firing_rates = firing_rates
        self.orientations = orientations
        self.alpha = self.orientations[1] - self.orientations[0]
        self.preferred_orientations = preferred_orientations
        self.new_tuning_angles = []
        self.max_firing_rates = []

    def hotelling_t2_test(self, orientation_vectors):
        """
        Perform Hotelling T2 test on orientation vectors.
        """
        n_trials, _ = orientation_vectors.shape
        # Compute the mean orientation vector
        mean_vector = np.mean(orientation_vectors, axis=0)
        # Compute the covariance matrix of the orientation vectors
        cov_matrix = np.cov(orientation_vectors, rowvar=False)
        diff_mean = mean_vector - np.mean(mean_vector)
        # Try to compute the inverse of the covariance matrix
        try:
            cov_matrix_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # If singular, use a small regularization or pseudo-inverse
            cov_matrix_inv = np.linalg.pinv(cov_matrix)  # Use pseudo-inverse as a fallback
            # Alternatively, you could use regularization:
            # cov_matrix_inv = np.linalg.inv(cov_matrix + 1e-10 * np.eye(cov_matrix.shape[0]))
        # Compute the T2 statistic
        t2_statistic = n_trials * diff_mean.T @ cov_matrix_inv @ diff_mean
        # Convert the T2 statistic to an F-statistic
        df1 = 2
        df2 = n_trials - 2
        f_statistic = (df2 * t2_statistic) / (df1 * (n_trials - 1))
        # Compute the p-value from the F-distribution
        p_value = 1 - f_distribution.cdf(f_statistic, df1, df2)
        # Significant if p-value < 0.01
        return p_value < 0.05

    def get_bounds(self, M):
        # Define contraints on the fit parameters from the double gaussian
        return (
            [-M, 0, 0, 0, self.alpha/2],  # Lower bounds
            [M, 3*M, 3*M, 360, np.inf]  # Upper bounds
        )
    
    def calculate_orientation_vectors(self, responses, orientations):
        # Calculate orientation vectors for each trial.
        theta_rad = np.deg2rad(2 * orientations)
        # Calculate the cosines and sines of the orientations
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        # Use broadcasting to multiply responses with cosines and sines
        orientation_vectors_x = np.dot(responses, cos_theta)
        orientation_vectors_y = np.dot(responses, sin_theta)
        # Stack the results into a single array
        orientation_vectors = np.stack((orientation_vectors_x, orientation_vectors_y), axis=-1)
        
        return orientation_vectors

    def calculate_tuning_angle(self):
        for neuron_idx in range(self.firing_rates.shape[2]):
            neuron_responses_all_trials = self.firing_rates[:, :, neuron_idx]
            neuron_responses = np.mean(neuron_responses_all_trials, axis=0)

            if np.sum(neuron_responses_all_trials) == 0:
                self.new_tuning_angles.append(np.nan)
                self.max_firing_rates.append(np.max(neuron_responses))
                continue
            
            orientation_vectors = self.calculate_orientation_vectors(neuron_responses_all_trials, self.orientations)
            if not self.hotelling_t2_test(orientation_vectors):
                self.new_tuning_angles.append(np.nan)
                self.max_firing_rates.append(np.max(neuron_responses))
                continue
            
            M = np.max(neuron_responses)
            M_orientation = self.orientations[np.argmax(neuron_responses)]
            # Initial guess for the fit parameters
            initial_guess = [
                0,  # C
                M,  # Rp
                M,  # Rn
                M_orientation,  # theta_pref
                self.alpha / 2,  # sigma
            ]
            # Explore several initial values for sigma (a1, a2) and theta_pref (b1, b2)
            initial_sigmas = [self.alpha/2, self.alpha, 2*self.alpha]
            # If the maximum is at 0, we also try 360
            if M_orientation == 0:
                initial_pref_orientations = [0, 360]
            else:
                initial_pref_orientations = [M_orientation]

            bounds = self.get_bounds(M)
            best_fit = None
            best_error = np.inf   

            for pref_orientation in initial_pref_orientations:
                initial_guess[3] = pref_orientation      
                for sigma in initial_sigmas:
                    initial_guess[4] = sigma
                    try:
                        popt, _ = curve_fit(double_gaussian, self.orientations, neuron_responses, 
                                            p0=initial_guess, bounds=bounds, ftol=1e-4, maxfev=1000)
                        fit_error = np.sum((double_gaussian(self.orientations, *popt) - neuron_responses) ** 2)
                        if fit_error < best_error:
                            best_error = fit_error
                            best_fit = popt
                    except RuntimeError as e:
                        continue

            if best_fit is not None:
                pref_angle = best_fit[3] if best_fit[2] < best_fit[1] else best_fit[3] - 180
                pref_angle = pref_angle % 360
                self.new_tuning_angles.append(pref_angle)
                # get the max response of the fit
                max_response = double_gaussian(pref_angle, *best_fit)
                self.max_firing_rates.append(max_response)
            else:
                self.new_tuning_angles.append(np.nan)
                self.max_firing_rates.append(np.max(neuron_responses))

        self.new_tuning_angles = np.array(self.new_tuning_angles) 
        self.max_firing_rates = np.array(self.max_firing_rates)

        return self.new_tuning_angles, self.max_firing_rates

class ModelMetricsAnalysis:    

    def __init__(self, spikes, DG_angles, network, data_dir='GLIF_network', 
                 drifting_gratings_init=None, drifting_gratings_end=None, spontaneous_init=None, spontaneous_end=None,
                 core_radius=400, save_df=False, df_directory='Metrics_analysis', neuropixels_df='v1_OSI_DSI_DF.csv'):
        self.n_neurons = network['n_nodes']
        self.network = network
        self.data_dir = data_dir 
        self.drifting_gratings_init = drifting_gratings_init
        self.drifting_gratings_end = drifting_gratings_end
        self.spontaneous_init = spontaneous_init
        self.spontaneous_end = spontaneous_end
        
        # Handle both full paths and just filenames for neuropixels_df
        if os.path.isabs(neuropixels_df) or os.path.exists(neuropixels_df):
            self.neuropixels_df = os.path.basename(neuropixels_df)  # Extract just the filename
        else:
            self.neuropixels_df = neuropixels_df
            
        # self.analyze_core_only = analyze_core_only
        self.core_radius = core_radius
        # Isolate the core neurons if necessary
        if self.core_radius > 0:
            self.core_mask = other_v1_utils.isolate_core_neurons(self.network, radius=self.core_radius, data_dir=self.data_dir)
            n_neurons_plot = np.sum(self.core_mask)
        else:
            self.core_mask = np.full(self.n_neurons, True)
            # core_radius = None
            n_neurons_plot = self.n_neurons

        # Calculate the firing rates
        if len(spikes.shape) == 3:
            spikes = np.expand_dims(spikes, axis=0)
        elif len(spikes.shape) == 2:
            spikes = np.expand_dims(np.expand_dims(spikes, axis=0), axis=0)

        spikes = spikes[:, :, :, self.core_mask]
        if self.drifting_gratings_init is not None:
            firingRates = calculate_Firing_Rate(spikes, stimulus_init=self.drifting_gratings_init, 
                                                stimulus_end=self.drifting_gratings_end, temporal_axis=2)
            self.metrics_df = calculate_OSI_DSI(firingRates, self.network, session='drifting_gratings', DG_angles=DG_angles, n_selected_neurons=n_neurons_plot,
                                                core_radius=self.core_radius, remove_zero_rate_neurons=False, directory=df_directory, save_df=save_df)
        # Calculate the spontaneous metrics
        if self.spontaneous_init is not None:
            spontaneous_firingRates = calculate_Firing_Rate(spikes, stimulus_init=self.spontaneous_init,
                                                            stimulus_end=self.spontaneous_end, temporal_axis=2)
            self.metrics_df = calculate_OSI_DSI(spontaneous_firingRates, self.network, session='spontaneous', DG_angles=DG_angles,
                                                n_selected_neurons=n_neurons_plot, core_radius=self.core_radius, remove_zero_rate_neurons=False, directory=df_directory, save_df=save_df)

    def __call__(self, metrics=["Rate at preferred direction (Hz)", "OSI", "DSI"], axis=None, 
                 directory='', filename=''):

        boxplot = MetricsBoxplot(save_dir=directory, filename=filename)
        # boxplot.plot(metrics=metrics, metrics_df=metrics_df, additional_dfs=[osi_approx_real_df], additional_dfs_labels=['Approximation'], axis=axis)
        boxplot.plot(metrics=metrics, metrics_df=self.metrics_df, neuropixels_df=self.neuropixels_df, axis=axis)

    
class MetricsBoxplot:
    def __init__(self, save_dir='Metrics_analysis', filename=''):
        self.save_dir = save_dir
        self.filename = filename
        self.osi_dsi_dfs = []

    @staticmethod
    def pop_name_to_cell_type(pop_name):
        # Convert pop_name in the old format to cell types. E.g., 'e4Rorb' -> 'L4 Exc', 'i4Pvalb' -> 'L4 PV', 'i23Sst' -> 'L2/3 SST'
        shift = 0  # letter shift for L23
        layer = pop_name[1]
        if layer == "2":
            layer = "2/3"
            shift = 1
        elif layer == "1":
            return "L1 Htr3a"  # special case

        class_name = pop_name[2 + shift :]
        if class_name == "Pvalb":
            subclass = "PV"
        elif class_name == "Sst":
            subclass = "SST"
        elif (class_name == "Vip") or (class_name == "Htr3a"):
            subclass = "VIP"
        else:  # excitatory
            subclass = "Exc"

        return f"L{layer} {subclass}"

    @staticmethod
    def neuropixels_cell_type_to_cell_type(pop_name):
        if ' ' in pop_name:
            return pop_name

        # Convert pop_name in the neuropixels cell type to cell types. E.g, 'EXC_L23' -> 'L2/3 Exc', 'PV_L5' -> 'L5 PV'
        layer = pop_name.split('_')[1]
        class_name = pop_name.split('_')[0]
        if "2" in layer:
            layer = "L2/3"
        elif layer == "L1":
            return "L1 Htr3a"  # special case
        if class_name == "EXC":
            class_name = "Exc"

        return f"{layer} {class_name}"

    def get_osi_dsi_df(self, metric_file, data_source_name="", feature='', data_dir=""):
        # Load the data csv file and remove rows with empty cell type
        # if metric_file is a dataframe, then do not load it
        if data_dir == "Neuropixels_data":
            features_to_load = ['ecephys_unit_id', 'cell_type', 'firing_rate_sp', 'Ave_Rate(Hz)', "max_mean_rate(Hz)", "OSI", "DSI"]
            df = pd.read_csv(f"{data_dir}/{metric_file}", index_col=0, sep=" ", usecols=features_to_load).dropna(how='all')
            df = df[df['cell_type'].notna()]
            df["cell_type"] = df["cell_type"].apply(self.neuropixels_cell_type_to_cell_type)
        elif data_dir == 'Billeh_column_metrics':
            df = pd.read_csv(f"{data_dir}/{metric_file}", sep=" ")
            df["cell_type"] = df["pop_name"].apply(self.pop_name_to_cell_type)
        elif data_dir == "NEST_metrics":
            df = pd.read_csv(f"{data_dir}/{metric_file}", sep=" ")
            df["cell_type"] = df["pop_name"].apply(self.pop_name_to_cell_type)
            # plot only neurons within 200 um.
            df = df[(df["x"] ** 2 + df["z"] ** 2) < (200 ** 2)]
        elif isinstance(metric_file, pd.DataFrame):
            df = metric_file
            df["cell_type"] = df["pop_name"].apply(self.pop_name_to_cell_type)

        # Rename the maximum rate column
        if 'max_mean_rate(Hz)' in df.columns:
            df.rename(columns={"max_mean_rate(Hz)": "Rate at preferred direction (Hz)"}, inplace=True)
            # Cut off neurons with low firing rate at the preferred direction
            nonresponding = df["Rate at preferred direction (Hz)"] < 0.5
            df.loc[nonresponding, "OSI"] = np.nan
            df.loc[nonresponding, "DSI"] = np.nan
        
        if 'firing_rate_sp' in df.columns:
            df.rename(columns={"firing_rate_sp": "Spontaneous rate (Hz)"}, inplace=True)  
        elif 'Spont_Rate(Hz)' in df.columns:
            df.rename(columns={"Spont_Rate(Hz)": "Spontaneous rate (Hz)"}, inplace=True)
        
        if 'Ave_Rate(Hz)' in df.columns:
            df.rename(columns={"Ave_Rate(Hz)": "Evoked rate (Hz)"}, inplace=True)

        # Sort the neurons by neuron types
        df = df.sort_values(by="cell_type")

        # Add a column for the data source name
        if len(data_source_name) > 0:
            df["data_type"] = data_source_name
        else:
            df["data_type"] = data_dir

        columns = ["cell_type", "data_type", "Rate at preferred direction (Hz)", "OSI", "DSI", 'Evoked rate (Hz)', 'Spontaneous rate (Hz)']
        # Ensure all required columns exist in the DataFrame, fill with NaN if not present
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan

        df = df[columns]

        return df

    def plot(self, metrics=["Rate at preferred direction (Hz)", "OSI", "DSI"], metrics_df=None, neuropixels_df="v1_OSI_DSI_DF.csv", axis=None):
        # Get the dataframes for the model and Neuropixels OSI and DSI 
        if metrics_df is None:
            metrics_df = f"v1_OSI_DSI_DF.csv"

        self.osi_dsi_dfs.append(self.get_osi_dsi_df(f"V1_OSI_DSI_DF_pop_name.csv", data_source_name="Untrained model", data_dir='NEST_metrics'))
        self.osi_dsi_dfs.append(self.get_osi_dsi_df(metrics_df, data_source_name="V1 GLIF model", data_dir=self.save_dir))
        # self.osi_dsi_dfs.append(self.get_osi_dsi_df(f"OSI_DSI_neuropixels_v4.csv", data_source_name="Neuropixels", data_dir='Neuropixels_data'))
        # self.osi_dsi_dfs.append(self.get_osi_dsi_df(f"V1_OSI_DSI_DF.csv", data_source_name="Billeh et al (2020)", data_dir='Billeh_column_metrics'))
        self.osi_dsi_dfs.append(self.get_osi_dsi_df(neuropixels_df, data_source_name="Neuropixels", data_dir='Neuropixels_data'))
        df = pd.concat(self.osi_dsi_dfs, ignore_index=True)

        # Create a figure to compare several model metrics against Neuropixels data
        n_metrics = len(metrics)
        height = int(7*n_metrics)

        if axis is None:
            fig, axs = plt.subplots(n_metrics, 1, figsize=(12, height))
            # fig, axs = plt.subplots(n_metrics, 1, figsize=(12, 20))
            if n_metrics == 1:
                axs = [axs]
        else:
            axs = [axis]

        color_pal = {
            "Untrained model": "#FFCC80",  # Light orange
            "V1 GLIF model": "#FF8C00",    # Dark orange
            "Billeh et al (2020)": "tab:blue",  # Steel blue
            "Neuropixels": "tab:gray"       # Gray
        }

        # color_pal = {
        #     "V1 GLIF model": "tab:orange",
        #     "Neuropixels": "tab:gray",
        #     "Billeh et al (2020)": "tab:blue",
        #     "Untrained model": "tab:pink"
        # }

        # Set the order for hue in boxplots by creating a custom hue_order list
        hue_order = ["Untrained model", "V1 GLIF model", "Billeh et al (2020)", "Neuropixels"]
        # Filter hue_order to only include data types that are present in the dataframe
        hue_order = [h for h in hue_order if h in df["data_type"].unique()]

        # Establish the order of the neuron types in the boxplots
        cell_type_order = np.sort(df['cell_type'].unique())

        for idx, metric in enumerate(metrics):
            if metric in ["Rate at preferred direction (Hz)", 'Evoked rate (Hz)', 'Spontaneous rate (Hz)']:
                ylims = [0, 100]
            else:
                ylims = [0, 1]

            initial_average_similarity_score = compute_ks_statistics(df, metric=metric, data_source1='Untrained model', data_source2='Neuropixels')
            final_average_similarity_score = compute_ks_statistics(df, metric=metric, data_source1='V1 GLIF model', data_source2='Neuropixels')
            plot_one_metric(axs[idx], df, metric, ylims, color_pal, cell_type_order=cell_type_order,
                            hue_order=hue_order, initial_similarity_score=initial_average_similarity_score, final_similarity_score=final_average_similarity_score)

        axs[0].legend(loc="upper right", fontsize=20)
        # axs[0].set_title(f"V1", fontsize=20)
        if len(axs) > 1:
            for i in range(len(axs)-1):
                axs[i].set_xticklabels([])

        xticklabel = axs[n_metrics-1].get_xticklabels()
        for label in xticklabel:
            label.set_fontsize(20)
            # label.set_weight("bold")

        if axis is None:
            plt.tight_layout()
            os.makedirs(self.save_dir, exist_ok=True)
            plt.savefig(os.path.join(self.save_dir, self.filename+'.png'), dpi=300, transparent=False)
            plt.close()


def plot_one_metric(ax, df, metric_name, ylim, cpal=None, cell_type_order=None, hue_order=None, initial_similarity_score=None, final_similarity_score=None):

    sns.boxplot(
        x="cell_type",
        y=metric_name,
        hue="data_type",
        order=cell_type_order,
        hue_order=hue_order,  # Add this parameter to control the order of the hue categories
        data=df,
        ax=ax,
        width=0.7,
        palette=cpal,
    )
    # # Add pointplot for averages
    # sns.pointplot(
    #     x="cell_type",
    #     y=metric_name,
    #     hue="data_type",
    #     order=hue_order,
    #     data=df,
    #     ax=ax,
    #     dodge=0.4,  # adjust depending on the width of the boxplots
    #     linestyle='none',  # don't join the points with a line
    #     palette='dark',  # make the points dark
    #     markers='d',  # diamond-shaped markers
    #     markersize=0.75,  # adjust size of the points
    #     errorbar=None,  # no error bars
    #     legend=False
    # )

    ax.tick_params(axis="x", labelrotation=90, labelsize=20)
    ax.set_ylim(ylim)
    ax.set_xlabel("")
    # Modify the label sizes
    ax.set_ylabel(metric_name, fontsize=22)
    yticklabel = ax.get_yticklabels()
    for label in yticklabel:
        label.set_fontsize(20)

    # Apply shadings to each layer
    xticklabel = ax.get_xticklabels()
    borders = get_borders(xticklabel)
    draw_borders(ax, borders, ylim)

    # Hide the legend
    ax.get_legend().remove()
    # Add the average similarity score to the legend
    if initial_similarity_score is not None and final_similarity_score is not None:
        ax.text(0.05, 0.9, r'$S_0$' + f': {initial_similarity_score:.2f}\n' + r'$S_f$' + f': {final_similarity_score:.2f}', 
                transform=ax.transAxes, fontsize=20, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='black', alpha=0.7, linewidth=1.))
                
    return ax

class OneShotTuningAnalysis:
    def __init__(self, network, data_dir='GLIF_network', directory='', drifting_gratings_init=50, 
                 drifting_gratings_end=550, core_radius=400):
        self.n_neurons = network['n_nodes']
        self.network = network
        self.data_dir = data_dir 
        self.drifting_gratings_init = drifting_gratings_init
        self.drifting_gratings_end = drifting_gratings_end
        # self.analyze_core_only = analyze_core_only
        self.core_radius = core_radius
        self.directory = os.path.join(directory)
        os.makedirs(self.directory, exist_ok=True)

    def __call__(self, spikes, current_orientation):
        self.current_orientation = current_orientation[0][0]
        
        # Isolate the core neurons if necessary
        if self.core_radius > 0:
            self.core_mask = other_v1_utils.isolate_core_neurons(self.network, radius=self.core_radius, data_dir=self.data_dir)
            n_neurons_plot = np.sum(self.core_mask)
        else:
            self.core_mask = np.full(self.n_neurons, True)
            n_neurons_plot = self.n_neurons

        spikes = spikes[:, :, :, self.core_mask]
        # Calculate the firing rates for each neuron in the given configuration
        self.firing_rate = calculate_Firing_Rate(spikes, stimulus_init=self.drifting_gratings_init, stimulus_end=self.drifting_gratings_end, 
                                                temporal_axis=2)
        # self firing_rate has shape (1, 1, n_neurons), we need to remove the first two dimensions
        self.firing_rate = self.firing_rate[0, 0, :]
        self.tuning_angles = self.network['tuning_angle'][self.core_mask]
        self.pop_names = other_v1_utils.pop_names(self.network, n_selected_neurons=n_neurons_plot)
        self.cell_types = np.array([MetricsBoxplot.pop_name_to_cell_type(x) for x in self.pop_names])

        # Get the orientation angles and orientation assignments using the 'assign_orientation' method
        self.orientation_angles, self.orientation_assignments = self.assign_orientation(self.tuning_angles, n_neurons_plot)

    def assign_orientation(self, tuning_angles, n_neurons):
        # Assign each neuron to an orientation based on its preferred angle 
        orientation_angles = np.arange(0, 360, 45)
        n_angles = len(orientation_angles)
        orientation_assignments = np.zeros((n_neurons, n_angles)).astype(np.bool_)
        for i, angle in enumerate(orientation_angles):
            circular_diff = (tuning_angles - angle) % 360
            circular_diff = np.abs(np.minimum(circular_diff, 360 - circular_diff))
            orientation_assignments[:, i] = (circular_diff < 10).astype(np.bool_)

        return orientation_angles, orientation_assignments

    def plot_tuning_curves(self, epoch, remove_zero_rate_neurons=True):
        # Define the grid size
        nrows, ncols = 5, 4
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 25))  # Share x-axis among all plots
        plt.subplots_adjust(hspace=0.4)  # Adjust space between plots

        # Flatten the array of axes for easy iteration
        axs = axs.flatten()
        unique_cell_types = sorted(set(self.cell_types))  # Ensure consistent order of cell types

        # Loop over each unique cell type to plot tuning curves separately for each type
        for cell_id, cell_type in enumerate(unique_cell_types):
            # Create a boolean mask to filter neurons of the current cell type
            cell_type_mask = self.cell_types == cell_type
            # Initialize dictionaries to store mean and standard deviation of firing rates for each orientation angle
            orientation_firing_rates_dict = {}
            orientation_firing_rates_std_dict = {}
            # Loop over the orientation angles to calculate mean and standard deviation of firing rates
            for i, angle in enumerate(self.orientation_angles):
                # Create a boolean mask to filter neurons of the current cell type and orientation
                angle_mask = np.logical_and(self.orientation_assignments[:, i], cell_type_mask)
                firing_rates_at_angle = self.firing_rate[angle_mask]
                if remove_zero_rate_neurons:
                    zero_rate_mask = firing_rates_at_angle != 0
                    firing_rates_at_angle = firing_rates_at_angle[zero_rate_mask]
                    # fr_std = fr_std[zero_rate_mask]
                    # filtered_orientation_angles = np.array(self.orientation_angles)[zero_rate_mask]   
                # else:
                #     filtered_orientation_angles = self.orientation_angles   
                # Calculate the mean and standard deviation of firing rates for the current orientation angle
                orientation_firing_rates_dict[angle] = np.mean(firing_rates_at_angle)
                orientation_firing_rates_std_dict[angle] = np.std(firing_rates_at_angle)

            # Extract the firing rates and standard deviations from dictionaries
            fr = np.array(list(orientation_firing_rates_dict.values()))
            fr_std = np.array(list(orientation_firing_rates_std_dict.values()))

            # if remove_zero_rate_neurons:
            #     zero_rate_mask = fr != 0
            #     fr = fr[zero_rate_mask]
            #     fr_std = fr_std[zero_rate_mask]
            #     filtered_orientation_angles = np.array(self.orientation_angles)[zero_rate_mask]   
            # else:
            #     filtered_orientation_angles = self.orientation_angles   

            # Plot the tuning curve as a line plot with error bars on the respective subplot
            ax = axs[cell_id]
            ax.errorbar(self.orientation_angles, fr, yerr=fr_std, color='black', fmt='-o', label=f'{cell_type} tuning')

            # Add vertical line for the current orientation
            ax.axvline(x=self.current_orientation, color='red', linestyle='--', label=f'Current orientation: {self.current_orientation:.2f}\u00b0')
            ax.legend()  # Show legend
            
            # Set the x-axis tick positions and labels to be the orientation angles
            ax.set_xticks(self.orientation_angles)
            ax.set_xlabel('Tuning angle')
            ax.set_ylim(bottom=0)
            
            # Set the title of the plot using the area and current cell type
            ax.set_title(f'{cell_type}')
            
            # Set the y-axis label
            ax.set_ylabel('Firing rate (Hz)')
            
        # Hide the unused subplots
        for i in range(len(unique_cell_types), nrows * ncols):
            fig.delaxes(axs[i])

        plt.tight_layout()
        # plt.suptitle(f'{self.area} Tuning Curves', fontsize=20, y=1.02)  # Add main title and adjust its position
        path = os.path.join(self.directory, f'Tuning_curves')
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'epoch_{epoch}.png'), dpi=300, transparent=False)
        plt.close()

    def plot_max_rate_boxplots(self, epoch, remove_zero_rate_neurons=False, axis=None):
        # Create a DataFrame to store firing rates, preferred angles, and cell types.
        firing_rates_df = pd.DataFrame({'pop_name': self.pop_names, 'Rate at preferred direction (Hz)': np.full(len(self.firing_rate), np.nan)})
        circular_diff = (self.tuning_angles - self.current_orientation) % 360
        circular_diff = np.abs(np.minimum(circular_diff, 360 - circular_diff))
        # Isolate neurons that prefer the current orientation
        preferred_mask = (np.abs(circular_diff) < 10).astype(np.bool_)
        
        # Iterate over unique cell types to calculate preferred firing rates and store them in the DataFrame.
        unique_cell_types = sorted(set(self.cell_types))
        for i, cell_type in enumerate(unique_cell_types):
            # Create a boolean mask to filter neurons of the current cell type
            cell_type_mask = self.cell_types == cell_type
            mask = np.logical_and(preferred_mask, cell_type_mask)
            preferred_firing_rate = self.firing_rate[mask]
            firing_rates_df.loc[mask, "Rate at preferred direction (Hz)"] = preferred_firing_rate

        if remove_zero_rate_neurons:
            firing_rates_df = firing_rates_df[firing_rates_df["Rate at preferred direction (Hz)"] != 0]

        # Add a column to the DataFrame indicating the data type.
        firing_rates_df['data_type'] = 'V1 GLIF model'
        firing_rates_df['OSI'] = np.nan
        firing_rates_df['DSI'] = np.nan
        firing_rates_df['Ave_Rate(Hz)'] = np.nan
        # firing_rates_df['firing_rate_sp'] = np.nan

        # Create an instance of MetricsBoxplot and get OSI and DSI data from a file.
        filename = f'Epoch_{epoch}'
        boxplot = MetricsBoxplot(save_dir=self.directory, filename=filename)
        metrics = ["Rate at preferred direction (Hz)"]
        boxplot.plot(metrics=metrics, metrics_df=firing_rates_df, axis=axis)

