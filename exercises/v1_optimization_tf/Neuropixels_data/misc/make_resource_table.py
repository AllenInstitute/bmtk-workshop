# %%
# from quick start guide of the neuropixels tutorial

import os
import numpy as np
import matplotlib.pyplot as plt
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import pandas as pd

import analysisfuncs as af
from importlib import reload


# %%
manifest_path = os.path.join("/local1/data/ecephys_cache_dir/", "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
print(cache.get_all_session_types())
# %%
sessions = cache.get_session_table()
opto_sessions = sessions[sessions.full_genotype.str.contains("ChR2")]
len(sessions)
len(opto_sessions)
# %% loop and get all opto ids
cre_pos_all = []
for session_id in opto_sessions.index:
    print(session_id)
    session = cache.get_session_data(session_id)
    da = af.optotagging_spike_counts(session)
    cre_pos_units = af.opto_positive_units(da)
    cre_pos_all.append(cre_pos_units)
    # da.sel(unit_id=cre_pos_units)

# session_id = 791319847
# session = cache.get_session_data(session_id)

cre_pos_all_lin = np.hstack(cre_pos_all)
len(cre_pos_all_lin)
added_sessions = opto_sessions.assign(cre_positive_cells=[len(c) for c in cre_pos_all])
added_sessions[["full_genotype", "cre_positive_cells"]]
# %% save the result
# np.save("cre_positive_units", cre_pos_all_lin)


# %% browse specific session where there are few cells
# session_id = 791319847  # VIP
session_id = 829720705  # PV
session = cache.get_session_data(session_id)
da = af.optotagging_spike_counts(session)
cre_pos_units = af.opto_positive_units(da)
af.plot_optotagging_response(da)
# I admit this is one of the cleanest examples

plt.figure(figsize=(5, 5))
for unit_id in cre_pos_units:
    peak_channel = session.units.loc[int(unit_id)].peak_channel_id
    wv = session.mean_waveforms[int(unit_id)].sel(channel_id=peak_channel)
    plt.plot(wv.time * 1000, wv, "k", alpha=0.3)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (microvolts)")
_ = plt.plot([1.0, 1.0], [-160, 100], ":c")

# %% What do I need now?
# I need a big unit table, and define a cell-type column
# Also want to get fast-spiking vs broad-spiking cells

# step1 collect waveforms from all the recording
# units = cache.get_units()
# list(np.sort(units.columns))


def get_waveforms(session, struct="VIS"):
    units = session.units[session.units.ecephys_structure_acronym.str.match(struct)]
    waveforms = []
    for unit_id in units.index:
        peak_channel = session.units.loc[int(unit_id)].peak_channel_id
        wv = session.mean_waveforms[int(unit_id)].sel(channel_id=peak_channel)
        waveforms.append(wv)
    return (np.vstack(waveforms), np.array(units.index))


all_waves = []
all_unit_ids = []
for session_id in sessions.index:
    print(session_id)
    session = cache.get_session_data(session_id)
    waves, unit_ids = get_waveforms(session)
    all_waves.append(waves)
    all_unit_ids.append(unit_ids)
    # da.sel(unit_id=cre_pos_units)


# %%
# step2 perform pca to reduce dimensions
waveforms_stack = np.vstack(all_waves)
plt.plot(waveforms_stack[:20, :].transpose())
from sklearn.preprocessing import normalize

waveforms_stack_norm = -normalize(-waveforms_stack, axis=1, norm="max")
plt.plot(waveforms_stack_norm[:20, :].transpose())

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca_components = pca.fit_transform(waveforms_stack_norm)

# check results here
pca_components.shape
print(pca.explained_variance_ratio_)
plt.plot(pca_components[:, 0], pca_components[:, 1], ".")
plt.plot(pca_components[:, 1], pca_components[:, 2], ".")
plt.plot(pca_components[:, 2], pca_components[:, 0], ".")

# %%
# step3 cluster with a simplest possible method
from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(n_components=3, random_state=0)
labels = bgm.fit_predict(pca_components)


def binomial_prob(truth_table):
    prob = truth_table.sum() / truth_table.size
    err = np.sqrt(prob * (1 - prob) / truth_table.size)
    return (prob, err)


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for j in range(3):
    for i in range(3):
        print(binomial_prob(labels == i))
        ax[j].plot(
            pca_components[labels == i, j],
            pca_components[labels == i, (j + 1) % 3],
            ".",
        )

# %%
# plotting clustered waveforms
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    ax[i].plot(waveforms_stack_norm[labels == i, :].transpose(), "k", alpha=0.01)
    ax[i].set_ylim([-1.0, 1.0])


# %% saving the results
# I consider
# cluster 0: RS
# cluster 1: FS
# cluster 2: Other
names = np.array(["RS", "FS", "Other"])
namelist = names[labels]
all_unit_ids_lin = np.hstack(all_unit_ids)
wave_df = pd.DataFrame(namelist, index=all_unit_ids_lin, columns=["waveform_type"])
wave_df.index.name = "ecephys_unit_id"
wave_df.to_csv("waveform_type.csv")


# %% all the heavy-lifting is done. time to make a big table
# %load_ext autoreload
# %autoreload 0

# load up basic table
metrics = cache.get_unit_analysis_metrics_by_session_type(
    "brain_observatory_1.1",
    filter_by_validity=False,
    amplitude_cutoff_maximum=np.inf,
    presence_ratio_minimum=-np.inf,
    isi_violations_maximum=np.inf,
)

metrics2 = cache.get_unit_analysis_metrics_by_session_type(
    "functional_connectivity",
    filter_by_validity=False,
    amplitude_cutoff_maximum=np.inf,
    presence_ratio_minimum=-np.inf,
    isi_violations_maximum=np.inf,
)

all_metrics = pd.concat([metrics, metrics2], sort=True)


layer_info = pd.read_csv("layer_info.csv", index_col=0)

all_metrics = all_metrics.merge(layer_info, on="ecephys_unit_id", sort=False)


# %%
all_metrics

waveform_type = pd.read_csv("waveform_type.csv", index_col=0)

all_metrics_wt = all_metrics.join(waveform_type, on="ecephys_unit_id", sort=False)

all_metrics_wt  # yay! I got it!

# %% add a column for opto-tagging
all_metrics_wt["opto_tag"] = "No"

cre_positive_units = np.load("cre_positive_units.npy")

opto_name = {"P": "PV", "S": "SST", "V": "VIP"}

for u in cre_positive_units:
    opto_letter = all_metrics_wt["genotype"][u][0]
    all_metrics_wt["opto_tag"][u] = opto_name[opto_letter]


# %% let's save it
all_metrics_wt.to_csv("all_metrics_resourcetable.csv")

# %% load it

import pandas as pd
all_metrics_wt = pd.read_csv("all_metrics_resourcetable.csv", index_col=0)

# %% conditional probability analysis
# (all_metrics_wt["opto_tag"] == 'VIP').sum()  # working
import lazyscience as lz

# Non V1 cortical cells
# metric_v1 = all_metrics_wt[
#    (all_metrics_wt["ecephys_structure_acronym"].str.contains("VIS", na=False))
#    & (~all_metrics_wt["ecephys_structure_acronym"].str.contains("VISp", na=False))
# ]
# V1
# metric_v1 = all_metrics_wt[
#    (all_metrics_wt["ecephys_structure_acronym"].str.contains("VISp", na=False))
# ]
# All Visual cortical cells
metric_v1 = all_metrics_wt[
    (all_metrics_wt["ecephys_structure_acronym"].str.contains("VIS", na=False))
]

(metric_v1["opto_tag"] == "PV").sum()
(metric_v1["opto_tag"] == "SST").sum()
(metric_v1["opto_tag"] == "VIP").sum()
metric_v1[metric_v1["opto_tag"] == "PV"]["cortical_layer"].value_counts()
metric_v1[metric_v1["opto_tag"] == "SST"]["cortical_layer"].value_counts()
metric_v1[metric_v1["opto_tag"] == "VIP"]["cortical_layer"].value_counts()
metric_v1[metric_v1["opto_tag"] == "PV"]["waveform_type"].value_counts()
metric_v1[metric_v1["opto_tag"] == "SST"]["waveform_type"].value_counts()
metric_v1[metric_v1["opto_tag"] == "VIP"]["waveform_type"].value_counts()

all_metrics_wt[all_metrics_wt["opto_tag"] == "PV"]["waveform_type"].value_counts()
all_metrics_wt[all_metrics_wt["opto_tag"] == "SST"]["waveform_type"].value_counts()
all_metrics_wt[all_metrics_wt["opto_tag"] == "VIP"]["waveform_type"].value_counts()
all_metrics_wt["waveform_type"].value_counts()

lz.cond_prob(metric_v1["opto_tag"] == "PV", metric_v1["waveform_type"] == "FS")


# %% create cell categories
# E: waveform_type == 'RS' & opto_tag == 'No'
# PV: (waveform_type == 'FS' & opto_tag == 'No') | opto_tag == 'PV'
# SST: opto_tag == 'SST'
# VIP: opto_tag == 'VIP'


def cell_label(row):
    layer = row["cortical_layer"]
    if layer <= 1:
        return None
    layerlabel = "23" if layer == 2 else str(layer)
    if (row["waveform_type"] == "RS") & (row["opto_tag"] == "No"):
        return f"EXC_L{layerlabel}"
    if ((row["waveform_type"] == "FS") & (row["opto_tag"] == "No")) | (
        row["opto_tag"] == "PV"
    ):
        return f"PV_L{layerlabel}"
    if row["opto_tag"] == "SST":
        return f"SST_L{layerlabel}"
    if row["opto_tag"] == "VIP":
        return f"VIP_L{layerlabel}"
    return None


metric_v1["cell_type"] = metric_v1.apply(cell_label, axis=1)
# metric_v1["opto_tag"] == "VIP"

# %% here's our resource table
import numpy as np

list(metric_v1.columns)

bo_counts = metric_v1[metric_v1["session_type"] == "brain_observatory_1.1"][
    "cell_type"
].value_counts()
fc_counts = metric_v1[metric_v1["session_type"] == "functional_connectivity"][
    "cell_type"
].value_counts()

cellcounts = metric_v1["cell_type"].value_counts()

metric_v1.loc[(metric_v1["width_rf"] > 100), "width_rf"] = np.nan
metric_v1.loc[(metric_v1["height_rf"] > 100), "height_rf"] = np.nan

r_table_med = metric_v1.groupby("cell_type").median()
r_table_sem = metric_v1.groupby("cell_type").sem()
r_table = metric_v1.groupby("cell_type").mean()
list(metric_v1[metric_v1["cell_type"] == "PV_L6"]["c50_dg"])
list(metric_v1[metric_v1["cell_type"] == "PV_L6"]["g_osi_dg"])
list(metric_v1[metric_v1["cell_type"] == "PV_L6"]["firing_rate_dg"])
list(metric_v1[metric_v1["cell_type"] == "PV_L6"]["width_rf"])
r_table.insert(0, "n_cells_BO", bo_counts)
r_table.insert(1, "n_cells_FC", fc_counts)
r_table_med.insert(0, "n_cells_BO", bo_counts)
r_table_med.insert(1, "n_cells_FC", fc_counts)
r_table_sem.insert(0, "n_cells_BO", bo_counts)
r_table_sem.insert(1, "n_cells_FC", fc_counts)
# r_table

# %% limit the elements so that it is human-friendly
list(r_table.columns)
elems = [
    "n_cells_BO",
    "n_cells_FC",
    "g_osi_dg",
    "g_osi_sg",
    "g_dsi_dg",
    "f1_f0_dg",
    "pref_sf_sg",
    "pref_tf_dg",
    "c50_dg",
    "area_rf",
    "width_rf",
    "height_rf",
    "fano_dg",
    "fano_dm",
    "fano_fl",
    "fano_ns",
    "fano_rf",
    "fano_sg",
    "firing_rate",
    "firing_rate_dg",
    "firing_rate_sg",
    "firing_rate_dm",
    "firing_rate_fl",
    "firing_rate_ns",
    "firing_rate_rf",
    "image_selectivity_ns",
    "lifetime_sparseness_dg",
    "lifetime_sparseness_sg",
    "lifetime_sparseness_dm",
    "lifetime_sparseness_fl",
    "lifetime_sparseness_ns",
    "lifetime_sparseness_rf",
    "mod_idx_dg",
    "pref_phase_sg",
    "pref_speed_dm",
    "run_mod_dg",
    "run_mod_dm",
    "run_mod_fl",
    "run_mod_ns",
    "run_mod_rf",
    "run_mod_sg",
    "sustained_idx_fl",
    "cortical_depth",
    "cortical_layer",
]
minelems = [
    "n_cells_BO",
    "n_cells_FC",
    "g_osi_dg",
    "g_dsi_dg",
    "pref_sf_sg",
    "pref_tf_dg",
    "firing_rate_rf",
    "firing_rate_dg",
    "width_rf",
    "height_rf",
    "fano_dg",
    "c50_dg",
]


pd.set_option("display.max_columns", None)

r_table_med[elems]

pd.set_option("display.float_format", "{:0.2f}".format)
r_table_sem[minelems]
r_table[minelems]
r_table_med[minelems]

# %%  mixing in the contrast analysis
cont_table = pd.read_csv("contrast_analysis.csv", index_col=0)

merge_inner = pd.merge(
    left=metric_v1, right=cont_table, left_index=True, right_index=True
)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# filter out unresponsive cells

# calculate the fractoin for each cell type
# maybe filter out unnecessary cell types
# plot
fdf = merge_inner[merge_inner["sig0.01"]].sort_values("cell_type")
# fdf = merge_inner[
#    merge_inner["sig0.01"] & (merge_inner["ecephys_structure_acronym"] == "VISp")
# ].sort_values("cell_type")

# make groupby might be the most straightforward answer
fdf["is_hp"] = fdf["best_model"] == "high"
fdf["is_bp"] = fdf["best_model"] == "band"
fdf["is_lp"] = fdf["best_model"] == "low"
fdf["isV1"] = fdf["ecephys_structure_acronym"] == "VISp"
fgroup = fdf.groupby("cell_type").mean()

plt.style.use("seaborn")

fig, a = plt.subplots(2, 1, figsize=(14, 8))
conds = ["is_bp", "is_lp", "is_hp"]
colors = ["orange", "blue", "red"]
legends = ['bandpass', 'lowpass', 'highpass']
data = [fdf[fdf["isV1"]], fdf]
titles = ["V1 Cells", "Any Vis. Cortical cells"]

for j in range(2):
    plt.sca(a[j])
    for i in range(3):
        sns.pointplot(
            x="cell_type",
            y=conds[i],
            row="isV1",
            color=colors[i],
            data=data[j],
            linestyles="None",
        )
    a[j].set(xlabel="", ylabel="fraction", title=titles[j])
    a[j].set_xlim([-0.5, 13.5])

a[0].legend(a[0].lines[1::13], legends)

# %%

