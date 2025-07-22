import sys
import quantities as pq
import pandas as pd
import numpy as np

sys.path.append("icsd_scripts/")

import h5py


def get_csd_input_dict(
    path2_lfp_h5: str = "output_allsynapses/cortical_electrode.h5",
) -> dict:
    """
    Get the input dictionary for the CSD calculation.
    """
    with h5py.File(path2_lfp_h5, "r") as h5:
        # channel_idx = h5["/ecp/channel_id"][()]
        # ts = np.arange(
        #     start=h5["/ecp/time"][0], stop=h5["/ecp/time"][1], step=h5["/ecp/time"][2]
        # )
        lfp = h5["/ecp/data"][()]

    coord_ele = pd.read_csv("components/electrodes/linear_electrode.csv", sep=" ")

    z_data = (
        np.array(coord_ele["y_pos"]) * (-1e-6) * pq.m
    )  # [um] --> [m] linear probe - electrodes' position in depth
    diam = 800e-6 * pq.m  # [m] source diameter
    h = 100e-6 * pq.m  # [m] distance between channels
    sigma = 0.3 * pq.S / pq.m  # [S/m] or [1/(ohm*m)] extracellular conductivity
    sigma_top = 0.3 * pq.S / pq.m  # [S/m] or [1/(ohm*m)] conductivity on top of cortex

    delta_input = {
        "lfp": lfp.T * 1e-3 * pq.V,  # [um] --> [mV]
        "coord_electrode": z_data,
        "diam": diam,
        "sigma": sigma,
        "sigma_top": sigma_top,
        "f_type": "gaussian",  # gaussian filter
        "f_order": (3, 1),  # 3-point filter, sigma = 1.
    }
    return delta_input
