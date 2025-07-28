# functions for analyzing the optotagging data
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import lazyscience as lz


def optotagging_spike_counts(session, struct="VIS"):
    time_resolution = 0.0005  # 0.5 ms bins
    bin_edges = np.arange(-0.01, 0.025, time_resolution)

    trials = session.optogenetic_stimulation_epochs[
        (session.optogenetic_stimulation_epochs.duration > 0.009)
        & (session.optogenetic_stimulation_epochs.duration < 0.02)
    ]

    units = session.units[session.units.ecephys_structure_acronym.str.match(struct)]

    spike_matrix = np.zeros((len(trials), len(bin_edges), len(units)))
    for unit_idx, unit_id in enumerate(units.index.values):
        spike_times = session.spike_times[unit_id]
        for trial_idx, trial_start in enumerate(trials.start_time.values):
            in_range = (spike_times > (trial_start + bin_edges[0])) * (
                spike_times < (trial_start + bin_edges[-1])
            )
            binned_times = (
                (spike_times[in_range] - (trial_start + bin_edges[0])) / time_resolution
            ).astype("int")
            spike_matrix[trial_idx, binned_times, unit_idx] = 1

    return xr.DataArray(
        name="spike_counts",
        data=spike_matrix,
        coords={
            "trial_id": trials.index.values,
            "time_relative_to_stimulus_onset": bin_edges,
            "unit_id": units.index.values,
        },
        dims=["trial_id", "time_relative_to_stimulus_onset", "unit_id"],
    )


def plot_optotagging_response(da):
    bin_edges = da["time_relative_to_stimulus_onset"]
    time_resolution = bin_edges[1] - bin_edges[0]
    plt.figure(figsize=(5, 10))

    n_units = da.sizes["unit_id"]

    plt.imshow(
        da.mean(dim="trial_id").T / time_resolution,
        extent=[np.min(bin_edges), np.max(bin_edges), 0, n_units],
        aspect="auto",
        vmin=0,
        vmax=200,
    )

    for bound in [0.0005, 0.0095]:
        plt.plot([bound, bound], [0, n_units], ":", color="white", linewidth=1.0)

    plt.xlabel("Time (s)")
    plt.ylabel("Unit #")

    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label("Mean firing rate (Hz)")


def opto_positive_units(da, threshold=1e-5):
    duration = 0.008
    baseline = da.sel(time_relative_to_stimulus_onset=slice(-0.010, -0.002))
    baseline_rate = (
        baseline.sum(dim="time_relative_to_stimulus_onset").mean(dim="trial_id")
        / duration
    )
    baseline_var = (
        baseline.sum(dim="time_relative_to_stimulus_onset").var(dim="trial_id")
        / duration
    )

    evoked = da.sel(time_relative_to_stimulus_onset=slice(0.001, 0.009))
    evoked_rate = (
        evoked.sum(dim="time_relative_to_stimulus_onset").mean(dim="trial_id")
        / duration
    )
    numtrials = da.sizes["trial_id"]

    import lazyscience as lz

    opto_sig = lz.quasi_poisson_sig_test_frvar(
        evoked_rate, duration * numtrials, baseline_rate, baseline_var
    )

    return da["unit_id"][opto_sig < threshold]

