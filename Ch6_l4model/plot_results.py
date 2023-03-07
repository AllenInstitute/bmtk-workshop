import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

from bmtk.utils import sonata
from bmtk.utils.reports import SpikeTrains


"""
def plot_tunning_angle_fr(spikes_path, fr_window=(500.0, 3000.0), convolv_window=10):
    dur_secs = (fr_window[1] - fr_window[0]) / 1000.0
    print(dur_secs)

    net = sonata.File(
        data_files='network/l4_nodes.h5',
        data_type_files='network/l4_node_types.csv'
    )
    nodes_df = net.nodes['l4'].to_dataframe(index_by_id=False)
    nodes_df = nodes_df[['node_id', 'node_type_id', 'model_name', 'tuning_angle', 'model_type', 'layer', 'ei']]

    spikes = SpikeTrains.load('output/spikes.h5')
    spikes_df = spikes.to_dataframe(population='v1')

    fr_df = spikes_df['node_ids'].value_counts().rename_axis('node_id').to_frame('spike_counts').reset_index()
    fr_df['firing_rates'] = fr_df['spike_counts'].values / dur_secs
    fr_df['node_id'] = fr_df['node_id'].astype(np.uint64)
    fr_df = fr_df.merge(nodes_df, how='right', on='node_id')
    fr_df['spike_counts'] = fr_df['spike_counts'].fillna(0.0)
    fr_df['firing_rates'] = fr_df['firing_rates'].fillna(0.0)

    def create_subplot(ax, grp_df, label):
        ax.scatter(grp_df['tuning_angle'], grp_df['firing_rates'], s=2)
        grp_df['tuning_rounded'] = np.floor(grp_df['tuning_angle'])
        fr_avgs = grp_df[['tuning_rounded', 'firing_rates']].groupby('tuning_rounded').agg(np.mean)
        # max_frs[r] = np.max([max_frs[r], np.max(grp_frs_df['firing_rates'])])

        if convolv_window and len(fr_avgs['firing_rates']) > convolv_window:
            filter = np.array([1.0] * int(convolv_window)) / float(convolv_window)
            fr_avgs['firing_rates'] = np.convolve(fr_avgs['firing_rates'].values, filter, mode='same')

        ax.plot(fr_avgs['firing_rates'], c='r', linewidth=3, label=label)
        ax.legend(fontsize=10, loc='upper right')

    # plot excitatory cells by layer
    fig, axes = plt.subplots(5, 1)
    for r, layer in enumerate(['VisL23', 'VisL4', 'VisL5', 'VisL6']):
        exc_df = fr_df[(fr_df['ei'] == 'e') & (fr_df['layer'] == layer)]
        create_subplot(axes[r], exc_df, 'exc; {}'.format(layer))

    # plot inhibitory cells
    inh_df = fr_df[fr_df['ei'] == 'i']
    create_subplot(axes[r+1], inh_df, 'inh; ALL')

    for r in range(5):
        if r != 4:
            axes[r].set_xticklabels([])

    plt.show()
"""

def plot_tunning_angle_fr(spikes_path, fr_window=(500.0, 3000.0), convolv_window=10):
    dur_secs = (fr_window[1] - fr_window[0]) / 1000.0
    print(dur_secs)

    net = sonata.File(
        data_files='network/l4_nodes.h5',
        data_type_files='network/l4_node_types.csv'
    )
    nodes_df = net.nodes['l4'].to_dataframe(index_by_id=False)
    nodes_df = nodes_df[['node_id', 'node_type_id', 'model_name', 'tuning_angle', 'model_type', 'layer', 'ei']]

    spikes = SpikeTrains.load('output/spikes.h5')
    spikes_df = spikes.to_dataframe(population='v1')

    fr_df = spikes_df['node_ids'].value_counts().rename_axis('node_id').to_frame('spike_counts').reset_index()
    fr_df['firing_rates'] = fr_df['spike_counts'].values / dur_secs
    fr_df['node_id'] = fr_df['node_id'].astype(np.uint64)
    fr_df = fr_df.merge(nodes_df, how='right', on='node_id')
    fr_df['spike_counts'] = fr_df['spike_counts'].fillna(0.0)
    fr_df['firing_rates'] = fr_df['firing_rates'].fillna(0.0)
    fr_df['tuning_rounded'] = 0.0

    def create_subplot(ax, grp_df, label):
        ax.scatter(grp_df['tuning_angle'], grp_df['firing_rates'], s=2)

        # print(grp_df.index.values)
        fr_avgs = pd.DataFrame({
            'tuning_rounded': np.floor(grp_df['tuning_angle']),
            'firing_rates': grp_df['firing_rates']
        }).groupby('tuning_rounded').agg(np.mean)

        if convolv_window and len(fr_avgs['firing_rates']) > convolv_window:
            filter = np.array([1.0] * int(convolv_window)) / float(convolv_window)
            fr_avgs['firing_rates'] = np.convolve(fr_avgs['firing_rates'].values, filter, mode='same')

        ax.plot(fr_avgs['firing_rates'], c='r', linewidth=3, label=label)
        ax.legend(fontsize=10, loc='upper right')

    # plot excitatory cells by layer
    nrows = 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 5))
    for r, layer in enumerate(['VisL4']):
        exc_df = fr_df[(fr_df['ei'] == 'e') & (fr_df['layer'] == layer)]
        create_subplot(axes[r], exc_df, 'exc; {}'.format(layer))

    # plot inhibitory cells
    inh_df = fr_df[fr_df['ei'] == 'i']
    create_subplot(axes[r + 1], inh_df, 'inh; ALL')

    for r in range(nrows):
        if r != (nrows - 1):
            axes[r].set_xticklabels([])

    plt.show()


def plot_firing_rates():
    net = sonata.File(
        data_files='network/lgn_nodes.h5',
        data_type_files='network/lgn_node_types.csv'
    )
    lgn_nodes_df = net.nodes['lgn'].to_dataframe(index_by_id=False)
    lgn_nodes_df = lgn_nodes_df[['node_id', 'model_name']]

    rates_df = pd.read_csv('inputs/rates.gratings.90deg_4Hz.csv', sep=' ')
    rates_df = rates_df.merge(lgn_nodes_df, how='left', on='node_id')
    for model_name, model_df in rates_df.groupby('model_name'):
        rates_tally = None
        n_counts = 0.0
        for _, node_grp in model_df.groupby('node_id'):
            n_counts += 1.0
            rates_tally = node_grp['firing_rates'].values if rates_tally is None else rates_tally + node_grp['firing_rates'].values

        rates_tally /= n_counts
        plt.plot(rates_tally, label=model_name)

    plt.legend(fontsize='x-small')
    plt.show()


def plot_firing_rates_h5(rates_file):
    net = sonata.File(
        data_files='network/lgn_nodes.h5',
        data_type_files='network/lgn_node_types.csv'
    )
    lgn_nodes_df = net.nodes['lgn'].to_dataframe(index_by_id=False)
    lgn_nodes_df = lgn_nodes_df[['node_id', 'model_name']]

    fig, axes = plt.subplots(1, 1, figsize=(16, 8))
    with h5py.File(rates_file, 'r') as h5:
        node_ids_lu = h5['/firing_rates/lgn/node_id'][()]
        firing_rates_hz = h5['/firing_rates/lgn/firing_rates_Hz']

        for model_name, model_df in lgn_nodes_df.groupby('model_name'):
            node_ids = model_df['node_id'].values
            node_ids_idx = node_ids_lu[node_ids]
            model_frs = firing_rates_hz[:, node_ids_idx]
            model_frs_avg = np.mean(model_frs, axis=1)
            axes.plot(model_frs_avg, label=model_name)

    axes.set_xlabel('Time (ms)')
    axes.set_ylabel('Firing Rates (Hz)')
    plt.show()


if __name__ == '__main__':
    # plot_firing_rates_h5('inputs/rates.gratings.90deg_4Hz.h5')
    # plot_firing_rates()
    plot_tunning_angle_fr(spikes_path='output/spikes.h5')
