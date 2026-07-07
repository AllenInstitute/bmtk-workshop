from pathlib import Path
import shutil
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

REPO_ROOT = Path(__file__).resolve().parents[1]
BMTK_ROOT = REPO_ROOT / "bmtk-dpointnet"
if str(BMTK_ROOT) not in sys.path:
    sys.path.insert(0, str(BMTK_ROOT))

from bmtk.simulator.dpointnet.loss_functions import loss_utils

CELL_TYPE_COLORS = {
    "Exc": "red",
    "PV": "green",
    "SST": "blue",
    "VIP": "purple",
    "Htr3a": "purple",
}


def _pop_name_to_layer(pop_name):
    if pop_name.startswith(("e23", "i23")):
        return "L2/3"
    if pop_name.startswith(("e4", "i4")):
        return "L4"
    if pop_name.startswith(("e5", "i5")):
        return "L5"
    if pop_name.startswith(("e6", "i6")):
        return "L6"
    if pop_name.startswith("i1"):
        return "L1"
    return "other"


def _pop_name_to_cell_type(pop_name):
    return loss_utils.pop_name_to_cell_type(pop_name, ignore_l5e_subtypes=True)


def _cell_type_class(cell_type):
    return str(cell_type).split()[-1]


def _normalize_neuropixels_cell_type(cell_type):
    if not isinstance(cell_type, str):
        return cell_type
    if " " in cell_type or "_" not in cell_type:
        return cell_type
    return loss_utils.neuropixels_cell_type_to_cell_type(cell_type)


def load_v1_metadata(network_dir="GLIF_network/network"):
    network_dir = Path(network_dir)
    node_types = pd.read_csv(network_dir / "v1_node_types.csv", sep=r"\s+", engine="python")
    type_to_pop = node_types.set_index("node_type_id")["pop_name"].to_dict()

    with h5py.File(network_dir / "v1_nodes.h5", "r") as nodes_h5:
        node_ids = nodes_h5["nodes/v1/node_id"][:].astype(np.int64)
        node_type_ids = nodes_h5["nodes/v1/node_type_id"][:]
        x = nodes_h5["nodes/v1/0/x"][:].astype(np.float32)
        z = nodes_h5["nodes/v1/0/z"][:].astype(np.float32)

    pop_names = np.array([type_to_pop[node_type_id] for node_type_id in node_type_ids])
    layers = np.array([_pop_name_to_layer(pop_name) for pop_name in pop_names])
    cell_types = np.array([_pop_name_to_cell_type(pop_name) for pop_name in pop_names])
    return pd.DataFrame({
        "node_id": node_ids,
        "node_type_id": node_type_ids,
        "pop_name": pop_names,
        "layer": layers,
        "cell_type": cell_types,
        "x": x,
        "z": z,
    })


def _copy_attrs(source, target):
    for key, value in source.attrs.items():
        target.attrs[key] = value


def create_l4_cutout_network(source_dir="GLIF_network/network", target_dir="GLIF_network_l4_cutout/network", radius=200.0, overwrite=False):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if target_dir.exists():
        if not overwrite:
            return load_v1_metadata(target_dir)
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_v1_metadata(source_dir)
    distance_from_center = np.sqrt(metadata["x"].to_numpy() ** 2 + metadata["z"].to_numpy() ** 2)
    v1_mask = (metadata["layer"].to_numpy() == "L4") & (distance_from_center <= float(radius))
    selected_v1_ids = metadata.loc[v1_mask, "node_id"].astype(np.int64).to_numpy()
    if len(selected_v1_ids) == 0:
        raise ValueError(f"No L4 neurons found within radius {radius}")
    v1_id_map = {old_id: new_id for new_id, old_id in enumerate(selected_v1_ids)}

    def copy_dataset(source_group, target_group, name, data):
        dataset = target_group.create_dataset(name, data=data, dtype=source_group[name].dtype)
        _copy_attrs(source_group[name], dataset)

    with h5py.File(source_dir / "v1_nodes.h5", "r") as source_h5, h5py.File(target_dir / "v1_nodes.h5", "w") as target_h5:
        source_v1 = source_h5["nodes/v1"]
        source_props = source_v1["0"]
        v1_group = target_h5.create_group("nodes/v1")
        props_group = v1_group.create_group("0")
        copy_dataset(source_v1, v1_group, "node_id", np.arange(len(selected_v1_ids), dtype=source_v1["node_id"].dtype))
        for name in ["node_type_id", "node_group_id"]:
            copy_dataset(source_v1, v1_group, name, source_v1[name][:][v1_mask])
        copy_dataset(source_v1, v1_group, "node_group_index", np.arange(len(selected_v1_ids), dtype=source_v1["node_group_index"].dtype))
        for name in ["target_sizes", "tuning_angle", "x", "y", "z"]:
            copy_dataset(source_props, props_group, name, source_props[name][:][v1_mask])

    shutil.copy2(source_dir / "v1_node_types.csv", target_dir / "v1_node_types.csv")

    for population in ["lgn", "bkg"]:
        shutil.copy2(source_dir / f"{population}_nodes.h5", target_dir / f"{population}_nodes.h5")
        shutil.copy2(source_dir / f"{population}_node_types.csv", target_dir / f"{population}_node_types.csv")

    edge_specs = [
        ("v1_v1_edges.h5", "v1_v1_edge_types.csv", "v1_to_v1", True),
        ("lgn_v1_edges.h5", "lgn_v1_edge_types.csv", "lgn_to_v1", False),
        ("bkg_v1_edges.h5", "bkg_v1_edge_types.csv", "bkg_to_v1", False),
    ]
    for filename, edge_types_filename, edge_population, filter_source in edge_specs:
        with h5py.File(source_dir / filename, "r") as source_h5, h5py.File(target_dir / filename, "w") as target_h5:
            edge_group = source_h5[f"edges/{edge_population}"]
            source_ids = edge_group["source_node_id"][:].astype(np.int64)
            target_ids = edge_group["target_node_id"][:].astype(np.int64)
            edge_mask = np.isin(target_ids, selected_v1_ids)
            if filter_source:
                edge_mask &= np.isin(source_ids, selected_v1_ids)

            target_edges = target_h5.create_group(f"edges/{edge_population}")
            target_props = target_edges.create_group("0")
            remapped_targets = np.array([v1_id_map[node_id] for node_id in target_ids[edge_mask]], dtype=edge_group["target_node_id"].dtype)
            copy_dataset(edge_group, target_edges, "target_node_id", remapped_targets)
            if filter_source:
                remapped_sources = np.array([v1_id_map[node_id] for node_id in source_ids[edge_mask]], dtype=edge_group["source_node_id"].dtype)
                copy_dataset(edge_group, target_edges, "source_node_id", remapped_sources)
            else:
                copy_dataset(edge_group, target_edges, "source_node_id", edge_group["source_node_id"][:][edge_mask])
            for name in ["edge_type_id", "edge_group_id"]:
                copy_dataset(edge_group, target_edges, name, edge_group[name][:][edge_mask])
            copy_dataset(edge_group, target_edges, "edge_group_index", np.arange(int(edge_mask.sum()), dtype=edge_group["edge_group_index"].dtype))
            for name in edge_group["0"].keys():
                copy_dataset(edge_group["0"], target_props, name, edge_group["0"][name][:][edge_mask])
        shutil.copy2(source_dir / edge_types_filename, target_dir / edge_types_filename)

    return load_v1_metadata(target_dir)


def load_neuropixels_rates(neuropixels_df="Neuropixels_data/OSI_DSI_neuropixels_v4.csv.gz", stimulus_type="drifting_gratings"):
    if neuropixels_df is None:
        raise ValueError("neuropixels_df must be provided")
    neuropixels_df = Path(neuropixels_df)
    if stimulus_type in ["spontaneous", "gray"]:
        feature = "firing_rate_sp"
    elif stimulus_type in ["natural_stimuli", "natural_images"]:
        feature = "firing_rate_ns"
    elif stimulus_type == "drifting_gratings":
        feature = "Ave_Rate(Hz)"
    else:
        raise ValueError(f"Unknown stimulus_type: {stimulus_type}")

    try:
        rates = pd.read_csv(neuropixels_df, sep=" ")
    except ValueError:
        rates = pd.read_csv(neuropixels_df)
    if feature == "firing_rate_sp" and feature not in rates.columns and "Spont_Rate(Hz)" in rates.columns:
        feature = "Spont_Rate(Hz)"
    rates = rates[["cell_type", feature]].dropna().copy()
    rates["cell_type"] = rates["cell_type"].apply(_normalize_neuropixels_cell_type)
    return rates.rename(columns={feature: "firing_rate_hz"})


def _load_spike_table(spikes_file, population="v1", batch=0):
    with h5py.File(spikes_file, "r") as spikes_h5:
        if "spikes" not in spikes_h5 or population not in spikes_h5["spikes"]:
            return pd.DataFrame({"node_id": pd.Series(dtype=np.int64), "time_ms": pd.Series(dtype=np.float32)})
        spike_group = spikes_h5[f"spikes/{population}"]
        node_ids = spike_group["node_ids"][:].astype(np.int64)
        timestamps = spike_group["timestamps"][:].astype(np.float32)
        if "batch_num" in spike_group:
            batch_nums = spike_group["batch_num"][:]
            keep = batch_nums == batch
            node_ids = node_ids[keep]
            timestamps = timestamps[keep]
    return pd.DataFrame({"node_id": node_ids, "time_ms": timestamps})


def spike_firing_rates(spikes_file, network_dir="GLIF_network/network", batch=0, t_start=0.0, t_stop=500.0):
    metadata = load_v1_metadata(network_dir)
    spikes = _load_spike_table(spikes_file, batch=batch)
    spikes = spikes[(spikes["time_ms"] >= t_start) & (spikes["time_ms"] < t_stop)]
    duration_s = (t_stop - t_start) / 1000.0
    spike_counts = spikes.groupby("node_id").size().rename("spike_count")
    rates = metadata.join(spike_counts, on="node_id")
    rates["spike_count"] = rates["spike_count"].fillna(0.0)
    rates["firing_rate_hz"] = rates["spike_count"] / duration_s
    return rates


def summarize_firing_rates(rates, by=None):
    if rates is None:
        return pd.DataFrame()
    group = rates if by is None else rates.groupby(by)
    summary = group["firing_rate_hz"].agg(["count", "mean", "median", "std"])
    if by is None:
        return summary.to_frame().T
    return summary


def plot_raster(spikes_file, network_dir="GLIF_network/network", batch=0, t_start=0.0, t_stop=500.0, max_neurons=None):
    metadata = load_v1_metadata(network_dir).sort_values(["layer", "cell_type", "z", "x"]).reset_index(drop=True)
    if max_neurons is not None and len(metadata) > max_neurons:
        selected = metadata.iloc[np.linspace(0, len(metadata) - 1, max_neurons).astype(int)].copy()
    else:
        selected = metadata.copy()
    selected["row"] = np.arange(len(selected))
    selected["cell_class"] = selected["cell_type"].apply(_cell_type_class)

    spikes = _load_spike_table(spikes_file, batch=batch)
    spikes = spikes[
        (spikes["time_ms"] >= t_start)
        & (spikes["time_ms"] < t_stop)
        & spikes["node_id"].isin(selected["node_id"])
    ]
    spikes = spikes.merge(selected[["node_id", "row", "cell_class"]], on="node_id", how="left")

    fig, ax = plt.subplots(figsize=(6, 3), dpi=160)
    for cell_class, color in CELL_TYPE_COLORS.items():
        class_spikes = spikes[spikes["cell_class"] == cell_class]
        if class_spikes.empty:
            continue
        ax.scatter(class_spikes["time_ms"], class_spikes["row"], s=0.7, c=color, linewidths=0, label=cell_class)
    ax.set_xlim(t_start, t_stop)
    ax.set_ylim(-1, len(selected))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7, markerscale=4, loc="upper right")
    fig.tight_layout()
    return fig, ax


def plot_firing_rate_distribution(rates, neuropixels_rates=None, target_rate=None, by=None, bins=None, title=None):
    if rates is None:
        print("No model firing rates to plot yet.")
        return None, None
    if bins is None:
        bins = np.linspace(0.0, 30.0, 31)

    if by is None:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=160)
        ax.hist(rates["firing_rate_hz"], bins=bins, density=True, alpha=0.65, label="model")
        if neuropixels_rates is not None:
            ax.hist(neuropixels_rates["firing_rate_hz"], bins=bins, density=True, histtype="step", linewidth=1.5, label="Neuropixels")
        if target_rate is not None:
            ax.axvline(target_rate, color="black", linestyle="--", linewidth=1.0, label="target")
        axes = np.array([ax])
    else:
        groups = [name for name, group in rates.groupby(by) if not group.empty]
        n_cols = min(4, len(groups))
        n_rows = int(np.ceil(len(groups) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.0 * n_rows), dpi=160, sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, group_name in zip(axes, groups):
            group = rates[rates[by] == group_name]
            ax.hist(group["firing_rate_hz"], bins=bins, density=True, alpha=0.65, label="model")
            if neuropixels_rates is not None and by == "cell_type":
                target = neuropixels_rates[neuropixels_rates["cell_type"] == group_name]
                if not target.empty:
                    ax.hist(target["firing_rate_hz"], bins=bins, density=True, histtype="step", linewidth=1.2, label="Neuropixels")
            ax.set_title(group_name, fontsize=8)
        for ax in axes[len(groups):]:
            ax.set_visible(False)

    for ax in axes:
        if not ax.get_visible():
            continue
        ax.set_xlabel("FR (Hz)")
        ax.set_ylabel("Density")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig, axes


def plot_intro_result(spikes_file, network_dir, t_stop=1000.0, title="", neuropixels_rates=None, target_rate=None, by=None):
    if not Path(spikes_file).exists():
        print(f"Missing {spikes_file}. Run the corresponding simulation or training cell first.")
        return None
    rates = spike_firing_rates(spikes_file, network_dir=network_dir, t_stop=t_stop)
    print(f"Spike count: {len(_load_spike_table(spikes_file))}")
    display(summarize_firing_rates(rates, by=by) if by else summarize_firing_rates(rates))
    fig, ax = plot_raster(spikes_file, network_dir=network_dir, t_stop=t_stop)
    ax.set_title(f"{title} raster")
    plt.show()
    fig, axes = plot_firing_rate_distribution(
        rates,
        neuropixels_rates=neuropixels_rates,
        target_rate=target_rate,
        by=by,
        title=f"{title} firing-rate distribution",
    )
    plt.show()
    return rates


def plot_loss_table(losses_csv):
    losses = pd.read_csv(losses_csv)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=160)

    if {"loss_type", "epoch", "step", "loss_function", "loss_value"}.issubset(losses.columns):
        plot_rows = losses[
            (losses["loss_type"] == "step")
            & ~losses["loss_function"].isin(["__mean_rate", "__orientation_mean", "__orientation_first"])
        ].copy()
        max_step = max(int(plot_rows["step"].max()), 1)
        plot_rows["global_step"] = (plot_rows["epoch"].astype(int) - 1) * max_step + plot_rows["step"].astype(int)
        plot_rows["label"] = np.where(
            plot_rows["parameter"].fillna("") == "",
            plot_rows["loss_function"],
            plot_rows["parameter"] + ": " + plot_rows["loss_function"],
        )
        labels = ["__total_loss"] + [
            label for label in sorted(plot_rows["label"].unique())
            if label != "__total_loss" and "weight_regularizer" not in label
        ]
        for label in labels:
            series = plot_rows[plot_rows["label"] == label]
            if series.empty:
                continue
            display_label = "total loss" if label == "__total_loss" else label
            ax.plot(series["global_step"], series["loss_value"], label=display_label, linewidth=1.0)
        ax.set_xlabel("Training step")
    else:
        numeric_cols = [col for col in losses.columns if col != "datetime" and pd.api.types.is_numeric_dtype(losses[col])]
        if not numeric_cols:
            raise ValueError(f"No numeric loss columns found in {losses_csv}")
        x = np.arange(len(losses))
        for col in numeric_cols:
            if col.endswith("__mean_rate") or col.endswith("step") or col.endswith("epoch"):
                continue
            ax.plot(x, losses[col], label=col, linewidth=1.0)
        ax.set_xlabel("Logged step")

    ax.set_ylabel("Loss value")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    return fig, ax


def plot_loss_if_exists(losses_csv, title):
    losses_csv = Path(losses_csv)
    if not losses_csv.exists():
        print(f"Missing {losses_csv}. Run training to create it.")
        return None, None
    fig, ax = plot_loss_table(losses_csv)
    ax.set_title(title)
    plt.show()
    return fig, ax


def firing_rate_grid(metadata, spikes, layer, t_start, t_stop, xy_range=(-400.0, 400.0), bin_size=50.0):
    edges = np.arange(xy_range[0], xy_range[1] + bin_size, bin_size, dtype=np.float32)
    selected_cells = metadata[metadata["layer"] == layer].copy()
    selected_cells = selected_cells[
        selected_cells["x"].between(xy_range[0], xy_range[1], inclusive="left")
        & selected_cells["z"].between(xy_range[0], xy_range[1], inclusive="left")
    ]
    counts_per_bin, _, _ = np.histogram2d(selected_cells["z"], selected_cells["x"], bins=[edges, edges])

    selected_spikes = spikes[
        spikes["node_id"].isin(selected_cells["node_id"])
        & (spikes["time_ms"] >= t_start)
        & (spikes["time_ms"] < t_stop)
    ]
    spike_positions = selected_spikes.merge(selected_cells[["node_id", "x", "z"]], on="node_id", how="left")
    spike_counts, _, _ = np.histogram2d(spike_positions["z"], spike_positions["x"], bins=[edges, edges])

    duration_s = (t_stop - t_start) / 1000.0
    with np.errstate(invalid="ignore", divide="ignore"):
        fr_grid = spike_counts / counts_per_bin / duration_s
    fr_grid[counts_per_bin == 0] = np.nan
    return fr_grid, edges


def plot_spatial_fr_panels(spike_files_by_condition, network_dir="GLIF_network/network", batch=0, bin_size=50.0, t_windows=None, layers=("L4", "L5"), cmap="viridis", vmin=0.0, vmax=15.0):
    if t_windows is None:
        t_windows = [(start, start + 100) for start in range(0, 500, 100)]

    metadata = load_v1_metadata(network_dir)
    panel_data = []
    for condition, spikes_file in spike_files_by_condition.items():
        spikes = _load_spike_table(spikes_file, batch=batch)
        for layer in layers:
            row = []
            for t_start, t_stop in t_windows:
                grid, edges = firing_rate_grid(metadata, spikes, layer, t_start, t_stop, bin_size=bin_size)
                row.append(grid)
            panel_data.append((condition, layer, row, edges))

    if vmax is None:
        vmax = 0.0
        for _, _, grids, _ in panel_data:
            for grid in grids:
                if np.isfinite(grid).any():
                    vmax = max(vmax, float(np.nanmax(grid)))
        if vmax <= vmin:
            vmax = vmin + 1.0

    fig, axes = plt.subplots(len(panel_data), len(t_windows), figsize=(8, 6.5), dpi=180, sharex=True, sharey=True)
    if len(panel_data) == 1:
        axes = np.expand_dims(axes, axis=0)

    image = None
    for row_index, (condition, layer, grids, edges) in enumerate(panel_data):
        for col_index, ((t_start, t_stop), grid) in enumerate(zip(t_windows, grids)):
            ax = axes[row_index, col_index]
            image = ax.imshow(
                grid,
                origin="lower",
                extent=[edges[0], edges[-1], edges[0], edges[-1]],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                interpolation="nearest",
                aspect="equal",
            )
            if row_index == 0:
                ax.set_title(f"{t_start}-{t_stop} ms", fontsize=8)
            if col_index == 0:
                ax.set_ylabel(f"{condition}\n{layer}\nz (um)", fontsize=8)
            ax.tick_params(labelsize=7, length=2)

    for ax in axes[-1, :]:
        ax.set_xlabel("x (um)", fontsize=8)
    fig.subplots_adjust(right=0.88, wspace=0.08, hspace=0.18)
    cbar_ax = fig.add_axes([0.9, 0.18, 0.018, 0.64])
    fig.colorbar(image, cax=cbar_ax, label="Mean FR (Hz)")
    return fig, axes


def summarize_l5_side_bias(spike_files_by_condition, network_dir="GLIF_network/network", batch=0, t_start=0.0, t_stop=500.0):
    metadata = load_v1_metadata(network_dir)
    l5_cells = metadata[metadata["layer"] == "L5"].copy()
    rows = []
    duration_s = (t_stop - t_start) / 1000.0

    for condition, spikes_file in spike_files_by_condition.items():
        spikes = _load_spike_table(spikes_file, batch=batch)
        selected_spikes = spikes[(spikes["time_ms"] >= t_start) & (spikes["time_ms"] < t_stop)]
        spike_counts = selected_spikes.groupby("node_id").size().rename("spike_count")
        rates = l5_cells.join(spike_counts, on="node_id")
        rates["spike_count"] = rates["spike_count"].fillna(0.0)
        rates["firing_rate_hz"] = rates["spike_count"] / duration_s
        negative_z_hz = rates.loc[rates["z"] < 0, "firing_rate_hz"].mean()
        positive_z_hz = rates.loc[rates["z"] >= 0, "firing_rate_hz"].mean()
        rows.append({
            "condition": condition,
            "negative_z_hz": negative_z_hz,
            "positive_z_hz": positive_z_hz,
            "positive_minus_negative_hz": positive_z_hz - negative_z_hz,
        })

    return pd.DataFrame(rows).sort_values("condition")


def plot_spatial_results(spike_files_by_condition, network_dir="GLIF_network/network", title="Spatial firing-rate pattern", show_l5_bias=False):
    if not all(Path(path).exists() for path in spike_files_by_condition.values()):
        print("Spatial output files are not ready yet. Run the corresponding simulation or training cell first.")
        return None, None

    fig, axes = plot_spatial_fr_panels(spike_files_by_condition, network_dir=network_dir)
    fig.suptitle(title, y=0.98, fontsize=11)
    plt.show()
    if show_l5_bias:
        display(summarize_l5_side_bias(spike_files_by_condition, network_dir=network_dir))
    return fig, axes
