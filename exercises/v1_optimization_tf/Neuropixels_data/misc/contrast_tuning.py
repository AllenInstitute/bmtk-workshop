# %%
# from the database, choose some cells, and look at the contrast tuning properties.
# and determine whether the neuron is high-pass, low-pass or bandpass

import os
import numpy as np
import pandas as pd
from lazyscience import lazyscience as lz
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# from . import analysisfuncs as af  # this makes pylance happy
# import analysisfuncs as af  # this actually works
# from importlib import reload


# %% defining functions


def get_contrast_response(session):
    # take a session, and retrieve contrast response as a numpy array.
    # this also determines overall response significance to the stimulus
    session.stimulus_presentations.stimulus_name.unique()
    dgc_table = session.get_stimulus_table(["drifting_gratings_contrast"])

    dgc_table.contrast.value_counts()
    dgc_table.orientation.value_counts()

    dgc_simple = dgc_table[["orientation", "contrast"]]
    stimsort_index = dgc_simple.sort_values(["orientation", "contrast"]).index.values
    stimsort_ori = dgc_simple.sort_values(["orientation", "contrast"]).orientation

    units = session.units
    v1unit_id = units.index[units.ecephys_structure_acronym.str.contains("VIS")]
    v1_cont = session.presentationwise_spike_counts(
        [0, 0.5], dgc_table.index, v1unit_id
    )
    v1_cont_bg = session.presentationwise_spike_counts(
        [0.7, 1.0], dgc_table.index, v1unit_id
    )

    n_cells = len(v1unit_id)

    v1_cont_sorted = np.array(v1_cont.sel(stimulus_presentation_id=stimsort_index))
    v1_cont_bg_sorted = np.array(
        v1_cont_bg.sel(stimulus_presentation_id=stimsort_index)
    )
    # 4: number of directions
    # 9: number of orientations
    # 15: number of trials
    # perhaps these magic numbers should be determined from the data.
    v1_resp_reshaped = v1_cont_sorted.reshape([4, 9, 15, n_cells])
    v1_spont_reshaped = v1_cont_bg_sorted.reshape([4, 9, 15, n_cells])

    # trim off the first two elements (1% and 2% to match the Millman paper)
    v1_resp_reshaped = v1_resp_reshaped[:, 2:, :, :]
    v1_spont_reshaped = v1_spont_reshaped[:, 2:, :, :]
    contrasts = np.tile(
        np.array(np.sort(dgc_simple["contrast"].unique()[2:]), dtype=float), (15, 1)
    ).transpose()

    resp_sig = lz.quasi_poisson_sig_test_counts(
        response_counts=v1_resp_reshaped,
        window_size1=0.5,
        spont_counts=v1_spont_reshaped,
        window_size2=0.3,
        unit_axis=-1,
    )

    print("Responsive cells,", end=" ")
    lz.binomial_prob(resp_sig < 0.01, disp=True)

    ret_dict = dict(
        resp=v1_resp_reshaped,
        spont=v1_spont_reshaped,
        unit_id=v1unit_id,
        sig=resp_sig,
        contrasts=contrasts,
    )
    return ret_dict


def get_pref_ori(cell_response, cell_spont, debug=False):
    # get preferred orientation.
    # preferred orientation is determined by an orientation that evokes the most
    # significant respose (by quasi-poisson test)
    direction_sig = lz.quasi_poisson_sig_test_counts(
        response_counts=cell_response,
        window_size1=0.5,
        spont_counts=cell_spont,
        window_size2=0.3,
        unit_axis=0,  # to make it the axis of orientation
    )

    pref_ori = np.argmin(direction_sig)
    if debug:
        print("Pref_ori: " + str(pref_ori))
    return pref_ori


def sigmoid(c, s, c50):
    return 1.0 / (1.0 + np.exp(-s * (c - c50)))


def fit_contrast_models(data, contrasts):
    # fit 3 models (highpass, bandpass, lowpass) to the contrast response, and determine
    # the best fit model using AIC
    def highpass(h, b, c50):
        return b + h * sigmoid(contrasts, 10.0, c50)

    def lowpass(h, b, c50):
        return b + h * sigmoid(contrasts, -10.0, c50)

    def bandpass(h, b, c50r, c50f):
        return b + h * sigmoid(contrasts, 10, c50r) * sigmoid(contrasts, -10.0, c50f)

    half_mean = data.mean() / 2
    initp = (half_mean, half_mean, np.median(contrasts))
    limits = dict(b=(0, None), h=(0, None), c50=(0.0, 1.0))

    m_high = lz.spike_mle_nbinom(data, highpass, initp, limits)
    m_low = lz.spike_mle_nbinom(data, lowpass, initp, limits)

    initp_band = (half_mean, half_mean, np.median(contrasts), np.median(contrasts))
    limits_band = dict(b=(0, None), h=(0, None), c50r=(0.0, 1.0), c50f=(0.0, 1.0))

    m_band = lz.spike_mle_nbinom(data, bandpass, initp_band, limits_band)

    results = [m_high, m_band, m_low]
    aics = [2 * m.npar + 2 * m.fval for m in results]
    bestmodel = np.argmin(aics)
    diffs = np.diff(np.sort(aics))
    return (bestmodel, diffs[0], diffs[1])


def analyze_cell(id, resp, spont, contrasts):
    # analyze one neuron out of the session
    cell_response = resp[:, :, :, id]
    cell_spont = spont[:, :, :, id]
    pref_ori = get_pref_ori(cell_response, cell_spont)
    cell_resp_pref = cell_response[pref_ori, :, :]
    return fit_contrast_models(cell_resp_pref, contrasts)


def get_best_contrast_model(session):
    # do all the jobs for one session and make pandas dataframe
    contrast_responses = get_contrast_response(session)

    resp = contrast_responses["resp"]
    spont = contrast_responses["spont"]
    resp_sig = contrast_responses["sig"]
    cont = contrast_responses["contrasts"]
    n_cells = len(resp_sig)
    bestmodels = [analyze_cell(id, resp, spont, cont) for id in range(n_cells)]

    pd_clms = ["best_model", "aic_diff1", "aic_diff2"]

    df = pd.DataFrame(bestmodels, index=contrast_responses["unit_id"], columns=pd_clms)
    df["resp_sig"] = resp_sig
    df["sig0.01"] = resp_sig < 0.01
    return df


# %% the final big moment read sessions, parallel process the sessions, and store
#  the information
manifest_path = os.path.join("/local1/data/ecephys_cache_dir/", "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
print(cache.get_all_session_types())

sessions = cache.get_session_table()
func_sessions = sessions[sessions.session_type == "functional_connectivity"]
len(sessions)
print("# Functional connectivity sessions: " + str(len(func_sessions)))

import multiprocessing as mp

pool = mp.Pool(mp.cpu_count())
alldf_mp = pool.map(
    get_best_contrast_model,
    [cache.get_session_data(i) for i in func_sessions.index.values],
)
# took 7 mins to process all. not bad

df_all = pd.concat(alldf_mp)
model_desc = ["high", "band", "low"]
df_all["best_model"] = [model_desc[i] for i in df_all["best_model"]]

# df_all.to_csv("ResourceTable/NeuroPixels/contrast_analysis.csv")
# depending on where you run it...
df_all.to_csv("contrast_analysis.csv")


# %%


# s = cache.get_session_data(func_sessions.index[0])
# s.api.get_units()
