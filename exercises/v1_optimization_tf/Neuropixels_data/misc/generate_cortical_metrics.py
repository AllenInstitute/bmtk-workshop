# %% combine what's already done, and make a further aggregated table

# ingredients
# all_metrics_resourcetable.csv
# contrast_analysis.csv
# layer_info.csv
# waveform_type.csv

import pandas as pd
from lazyscience import lazyscience as lz

all_metrics_wt = pd.read_csv("all_metrics_resourcetable.csv", index_col=0)

# %%
metric_cortex = all_metrics_wt[
    (all_metrics_wt["ecephys_structure_acronym"].str.contains("VIS", na=False))
]


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


metric_cortex["cell_type"] = metric_cortex.apply(cell_label, axis=1)

# %% load in contrast analysis

cont_table = pd.read_csv("contrast_analysis.csv", index_col=0)
# merged_table = pd.join(left=metric_cortex, right=cont_table)
merged_table = metric_cortex.join(cont_table)
merged_table.loc[merged_table["sig0.01"] == False, "best_model"] = None
merged_table["best_model"].value_counts()

merged_table.to_csv("cortical_metrics.csv")
