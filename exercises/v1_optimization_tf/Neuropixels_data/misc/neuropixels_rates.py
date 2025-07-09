# %%
# compute firing rates of each cell type in neuropixels data

import os
import numpy as np
import pandas as pd

# from lazyscience import lazyscience as lz  # delete if not necessary
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

manifest_path = os.path.join("/local1/data/ecephys_cache_dir/", "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

print(cache.get_all_session_types())

sessions = cache.get_session_table()
bo_sessions = sessions[sessions.session_type == "brain_observatory_1.1"]
# len(sessions)
print("# Brain Observatory 1.1 sessions: " + str(len(bo_sessions)))


# session = cache.get_session_data(bo_sessions.index.values[7])
# session.stimulus_presentations.stimulus_name.unique()
# session.get_stimulus_table(["spontaneous"])
# session.get_stimulus_table(["natural_scenes"])


# %%
# natural_scenes firing rates are pre-computed by the ecephys group, so use that value.
# here I just calculate spontaneous firing rates in a period that is sandwiched by
# natural scenes session


def get_spontaneous_fr(session, where=-1):
    units = session.units
    v1unit_id = units.index[units.ecephys_structure_acronym.str.contains("VIS")]
    spont_table = session.get_stimulus_table(["spontaneous"])
    long_spont = spont_table[spont_table.duration > 200]
    # the long spontaneous period is the last long one in the BO session
    last_long_id = long_spont.index[where]
    spont_table.loc[last_long_id]
    spont_table.index

    t_range = [10.0, 300.0]
    duration = t_range[1] - t_range[0]
    counts = session.presentationwise_spike_counts(
        t_range, stimulus_presentation_ids=[last_long_id], unit_ids=v1unit_id
    )

    fr = counts / duration  # to make it FR
    fr_df = fr.to_dataframe()
    fr_df = fr_df.reset_index(level=[0, 1])
    fr_df = fr_df.rename(columns={"spike_counts": "firing_rate_sp"})
    fr_df = fr_df[["firing_rate_sp"]]
    return fr_df


# %%
import multiprocessing as mp

pool = mp.Pool(mp.cpu_count())
alldf_mp = pool.map(
    get_spontaneous_fr, [cache.get_session_data(i) for i in bo_sessions.index.values]
)

df_all = pd.concat(alldf_mp)
df_all.to_csv("last_spontaneous_rate.csv")


# %% get the first one
pool = mp.Pool(mp.cpu_count())
alldf_mp = pool.map(
    get_spontaneous_fr, [cache.get_session_data(i) for i in sessions.index.values],
)

df_all = pd.concat(alldf_mp)
df_all.to_csv("last_spontaneous_rate2.csv")

