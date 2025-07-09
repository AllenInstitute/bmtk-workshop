# %% [markdown]
# ## Exercise 5

# %% [markdown]
# ### General Instructions
#
# Simulate and plot the LFPs and CSD evoked a cortical column of 400 $\mu m$ radius consisting of L1 inhibitory population (Htr3a) and L2/3-L6 excitatory and inhibitory populations of PV, SST, and Htr3a cells. The network is designed to have a realistic distribution of synaptic connections and neuron types, with basal and apical dendritic synapses included. We provide the built network SONATA files (network*) for a nextwork with basal and apical synapses, only basal synapses, and only apical synapses, respectively. Use `BioNet` to run the simulations of the three networks and compared the evoked LFPs and CSDs.

# %% [markdown]
# - Total number of neurons: 100
# - Total number of external virtual neurons: 100
# - Simulation time: 3 s
# - Time step: 0.1 ms
# - Stimulations start at 0.5 s and last for 2.5 s

# %%
import json
import numpy as np
import matplotlib.pyplot as plt

from bmtk.builder.networks import NetworkBuilder


# %%
# L2/3 - 150 - 300
# L4 - 300 - 400
# L5 - 400 - 550
def get_coords(N, y_range, radius_min=0.0, radius_max=400.0):
    phi = 2.0 * np.pi * np.random.random([N])
    r = np.sqrt((radius_min**2 - radius_max**2) * np.random.random([N]) + radius_max**2)
    x = r * np.cos(phi)
    y = np.random.uniform(y_range[0], y_range[1], size=N)
    z = r * np.sin(phi)
    return x, y, z


# %%
cortcol_allsyns = NetworkBuilder("cortcol")
cortcol_basalsyns = NetworkBuilder("cortcol")
cortcol_apicalsyns = NetworkBuilder("cortcol")

# %%
for cortcol in [cortcol_allsyns, cortcol_basalsyns, cortcol_apicalsyns]:
    x, y, z = get_coords(20, y_range=[-300, -100])
    cortcol.add_nodes(
        N=20,
        # Reserved SONATA keywords used during simulation
        model_type="biophysical",
        model_template="ctdb:Biophys1.hoc",
        model_processing="aibs_perisomatic",
        dynamics_params="Scnn1a_485510712_params.json",
        morphology="Scnn1a_485510712_morphology.swc",
        # The x, y, z locations and orientations (in Euler angles) of each cell
        # Here, rotation around the pia-to-white-matter axis is randomized
        x=x,
        y=y,
        z=z,
        rotation_angle_xaxis=[0] * 20,
        rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=20),
        rotation_angle_zaxis=[3.646878266] * 20,
        tuning_angle=np.linspace(start=0.0, stop=360.0, num=20, endpoint=False),
        layer="L23",
        model_name="Scnn1a",
        ei="e",
    )

    x, y, z = get_coords(20, y_range=[-400, -300])
    cortcol.add_nodes(
        N=20,
        model_type="biophysical",
        model_template="ctdb:Biophys1.hoc",
        model_processing="aibs_perisomatic",
        dynamics_params="Scnn1a_485510712_params.json",
        morphology="Scnn1a_485510712_morphology.swc",
        # The x, y, z locations and orientations (in Euler angles) of each cell
        # Here, rotation around the pia-to-white-matter axis is randomized
        x=x,
        y=y,
        z=z,
        rotation_angle_xaxis=[0] * 20,
        rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=20),
        rotation_angle_zaxis=[3.646878266] * 20,
        tuning_angle=np.linspace(start=0.0, stop=360.0, num=20, endpoint=False),
        layer="L4",
        model_name="Scnn1a",
        ei="e",
    )

    x, y, z = get_coords(20, y_range=[-600, -400])
    cortcol.add_nodes(
        N=20,
        model_type="biophysical",
        model_template="ctdb:Biophys1.hoc",
        model_processing="aibs_perisomatic",
        dynamics_params="Scnn1a_485510712_params.json",
        morphology="Scnn1a_485510712_morphology.swc",
        x=x,
        y=y,
        z=z,
        rotation_angle_xaxis=[0] * 20,
        rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=20),
        rotation_angle_zaxis=[3.646878266] * 20,
        tuning_angle=np.linspace(start=0.0, stop=360.0, num=20, endpoint=False),
        layer="L5",
        model_name="Scnn1a",
        ei="e",
    )

    x, y, z = get_coords(20, y_range=[-600, -100])
    cortcol.add_nodes(
        N=20,
        model_type="biophysical",
        model_template="ctdb:Biophys1.hoc",
        model_processing="aibs_perisomatic",
        dynamics_params="Pvalb_473862421_params.json",
        morphology="Pvalb_473862421_morphology.swc",
        x=x,
        y=y,
        z=z,
        rotation_angle_xaxis=np.random.uniform(0.0, 2 * np.pi, size=20),
        rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=20),
        rotation_angle_zaxis=[2.539551891] * 20,
        model_name="Pvalb",
        ei="i",
    )

# Add synapses to the network
cortcol_allsyns.add_edges(
    source={"ei": "e"},
    target={"ei": "e"},
    connection_rule=lambda *_: np.random.randint(10, 15),
    dynamics_params="AMPA_ExcToExc.json",
    model_template="Exp2Syn",
    syn_weight=6.0e-03,
    delay=0.5,
    target_sections=["somatic", "basal", "apical"],
    distance_range=[2.0, 1.0e20],
)

cortcol_allsyns.add_edges(
    # Exc --> Inh connections
    source={"ei": "e"},
    target={"ei": "i"},
    connection_rule=lambda *_: np.random.randint(1, 10),
    dynamics_params="AMPA_ExcToInh.json",
    model_template="Exp2Syn",
    syn_weight=0.002,
    delay=3.0,
    target_sections=["somatic", "basal"],
    distance_range=[0.0, 1.0e20],
)

cortcol_allsyns.add_edges(
    # Inh --> Exc connections
    source={"ei": "i"},
    target={"ei": "e"},
    connection_rule=lambda *_: np.random.randint(0, 6),
    dynamics_params="GABA_InhToExc.json",
    model_template="Exp2Syn",
    syn_weight=0.0002,
    delay=2.0,
    target_sections=["somatic", "basal", "apical"],
    distance_range=[0.0, 50.0],
)

# ---- Basal synapses ----
cortcol_basalsyns.add_edges(
    source={"ei": "e"},
    target={"ei": "e"},
    connection_rule=lambda *_: np.random.randint(10, 15),
    dynamics_params="AMPA_ExcToExc.json",
    model_template="Exp2Syn",
    syn_weight=6.0e-03,
    delay=0.5,
    target_sections=["somatic", "basal"],
    distance_range=[2.0, 1.0e20],
)

cortcol_basalsyns.add_edges(
    # Exc --> Inh connections
    source={"ei": "e"},
    target={"ei": "i"},
    connection_rule=lambda *_: np.random.randint(1, 10),
    dynamics_params="AMPA_ExcToInh.json",
    model_template="Exp2Syn",
    syn_weight=0.002,
    delay=3.0,
    target_sections=["somatic", "basal"],
    distance_range=[0.0, 1.0e20],
)

cortcol_basalsyns.add_edges(
    # Inh --> Exc connections
    source={"ei": "i"},
    target={"ei": "e"},
    connection_rule=lambda *_: np.random.randint(0, 6),
    dynamics_params="GABA_InhToExc.json",
    model_template="Exp2Syn",
    syn_weight=0.0002,
    delay=2.0,
    target_sections=["somatic", "basal"],
    distance_range=[0.0, 50.0],
)

# ---- Apical synapses ----
cortcol_apicalsyns.add_edges(
    source={"ei": "e"},
    target={"ei": "e"},
    connection_rule=lambda *_: np.random.randint(10, 15),
    dynamics_params="AMPA_ExcToExc.json",
    model_template="Exp2Syn",
    syn_weight=6.0e-03,
    delay=0.5,
    target_sections=["apical"],
    distance_range=[2.0, 1.0e20],
)
cortcol_apicalsyns.add_edges(
    # Exc --> Inh connections
    source={"ei": "e"},
    target={"ei": "i"},
    connection_rule=lambda *_: np.random.randint(1, 10),
    dynamics_params="AMPA_ExcToInh.json",
    model_template="Exp2Syn",
    syn_weight=0.002,
    delay=3.0,
    target_sections=["apical"],
    distance_range=[0.0, 1.0e20],
)
cortcol_apicalsyns.add_edges(
    # Inh --> Exc connections
    source={"ei": "i"},
    target={"ei": "e"},
    connection_rule=lambda *_: np.random.randint(0, 6),
    dynamics_params="GABA_InhToExc.json",
    model_template="Exp2Syn",
    syn_weight=0.0002,
    delay=2.0,
    target_sections=["apical"],
    distance_range=[0.0, 50.0],
)

# # Save the networks to disk
# cortcol_allsyns.build()
# cortcol_allsyns.save(output_dir="network_allsynapses")

# cortcol_basalsyns.build()
# cortcol_basalsyns.save(output_dir="network_basalsynapses")

# cortcol_apicalsyns.build()
# cortcol_apicalsyns.save(output_dir="network_apicalsynapses")

# %% [markdown]
# 1.1.2. Create external nodes - virtual cells that provide inputs to the network.

# %%
np.random.seed(42)
external = NetworkBuilder("external")
# external.add_nodes(
#     N=100,
#     pop_name="external",
#     x=np.random.uniform(-40.0, 40.0, size=100),
#     y=np.random.uniform(-40.0, 40.0, size=100),
#     model_type="virtual",
#     ei="e",
# )
external.add_nodes(
    N=50,
    model_type="virtual",
    model_template="lgnmodel:tOFF_TF15",
    x=np.random.uniform(0.0, 240.0, 50),
    y=np.random.uniform(0.0, 120.0, 50),
    spatial_size=1.0,
    pop_name="tOFF",
    dynamics_params="tOFF_TF15.json",
)
external.add_nodes(
    N=50,
    model_type="virtual",
    model_template="lgnmodel:tON",
    x=np.random.uniform(0.0, 240.0, 50),
    y=np.random.uniform(0.0, 120.0, 50),
    spatial_size=1.0,
    pop_name="tON",
    dynamics_params="tON_TF8.json",
)

external_basalsyns = NetworkBuilder("external")
# external_basalsyns.add_nodes(
#     N=100,
#     pop_name="external",
#     x=np.random.uniform(-40.0, 40.0, size=100),
#     y=np.random.uniform(-40.0, 40.0, size=100),
#     model_type="virtual",
#     ei="e",
# )
external_basalsyns.add_nodes(
    N=50,
    model_type="virtual",
    model_template="lgnmodel:tOFF_TF15",
    x=np.random.uniform(0.0, 240.0, 50),
    y=np.random.uniform(0.0, 120.0, 50),
    spatial_size=1.0,
    pop_name="tOFF",
    dynamics_params="tOFF_TF15.json",
)
external_basalsyns.add_nodes(
    N=50,
    model_type="virtual",
    model_template="lgnmodel:tON",
    x=np.random.uniform(0.0, 240.0, 50),
    y=np.random.uniform(0.0, 120.0, 50),
    spatial_size=1.0,
    pop_name="tON",
    dynamics_params="tON_TF8.json",
)

external_apicalsyns = NetworkBuilder("external")
# external_apicalsyns.add_nodes(
#     N=100,
#     pop_name="external",
#     x=np.random.uniform(-40.0, 40.0, size=100),
#     y=np.random.uniform(-40.0, 40.0, size=100),
#     model_type="virtual",
#     ei="e",
# )
external_apicalsyns.add_nodes(
    N=50,
    model_type="virtual",
    model_template="lgnmodel:tOFF_TF15",
    x=np.random.uniform(0.0, 240.0, 50),
    y=np.random.uniform(0.0, 120.0, 50),
    spatial_size=1.0,
    pop_name="tOFF",
    dynamics_params="tOFF_TF15.json",
)
external_apicalsyns.add_nodes(
    N=50,
    model_type="virtual",
    model_template="lgnmodel:tON",
    x=np.random.uniform(0.0, 240.0, 50),
    y=np.random.uniform(0.0, 120.0, 50),
    spatial_size=1.0,
    pop_name="tON",
    dynamics_params="tON_TF8.json",
)

# %% [markdown]
# #### 1.2 Create the edges

# %% [markdown]
# 1.2.1. Internal edges - synaptic connections between neurons in the network.
#
# Cases:
# - All synapses
# - Basal dendritic synapses only
# - Apical dendritic synapses only

# %% [markdown]
# Basal & apical synapses

# %%
cortcol_allsyns.add_edges(
    source={"ei": "e"},
    target={"ei": "e"},
    connection_rule=lambda *_: np.random.randint(10, 15),
    dynamics_params="AMPA_ExcToExc.json",
    model_template="Exp2Syn",
    syn_weight=6.0e-03,
    delay=0.5,
    target_sections=["somatic", "basal", "apical"],
    distance_range=[2.0, 1.0e20],
)

cortcol_allsyns.add_edges(
    # Exc --> Inh connections
    source={"ei": "e"},
    target={"ei": "i"},
    connection_rule=lambda *_: np.random.randint(1, 10),
    dynamics_params="AMPA_ExcToInh.json",
    model_template="Exp2Syn",
    syn_weight=0.002,
    delay=3.0,
    target_sections=["somatic", "basal"],
    distance_range=[0.0, 1.0e20],
)

cortcol_allsyns.add_edges(
    # Inh --> Exc connections
    source={"ei": "i"},
    target={"ei": "e"},
    connection_rule=lambda *_: np.random.randint(0, 6),
    dynamics_params="GABA_InhToExc.json",
    model_template="Exp2Syn",
    syn_weight=0.0002,
    delay=2.0,
    target_sections=["somatic", "basal", "apical"],
    distance_range=[0.0, 50.0],
)

cortcol_allsyns.build()
cortcol_allsyns.save(output_dir="network_allsynapses")

# %% [markdown]
# Only basal synapses

# %%
cortcol_basalsyns.add_edges(
    source={"ei": "e"},
    target={"ei": "e"},
    connection_rule=lambda *_: np.random.randint(10, 15),
    dynamics_params="AMPA_ExcToExc.json",
    model_template="Exp2Syn",
    syn_weight=6.0e-03,
    delay=0.5,
    target_sections=["somatic", "basal"],
    distance_range=[2.0, 1.0e20],
)

cortcol_basalsyns.add_edges(
    # Exc --> Inh connections
    source={"ei": "e"},
    target={"ei": "i"},
    connection_rule=lambda *_: np.random.randint(1, 10),
    dynamics_params="AMPA_ExcToInh.json",
    model_template="Exp2Syn",
    syn_weight=0.002,
    delay=3.0,
    target_sections=["somatic", "basal"],
    distance_range=[0.0, 1.0e20],
)

cortcol_basalsyns.add_edges(
    # Inh --> Exc connections
    source={"ei": "i"},
    target={"ei": "e"},
    connection_rule=lambda *_: np.random.randint(0, 6),
    dynamics_params="GABA_InhToExc.json",
    model_template="Exp2Syn",
    syn_weight=0.0002,
    delay=2.0,
    target_sections=["somatic", "basal"],
    distance_range=[0.0, 50.0],
)

cortcol_basalsyns.build()
cortcol_basalsyns.save(output_dir="network_basalsynapses")

# %% [markdown]
# Only apical synapses

# %%
cortcol_apicalsyns.add_edges(
    source={"ei": "e"},
    target={"ei": "e"},
    connection_rule=lambda *_: np.random.randint(10, 15),
    dynamics_params="AMPA_ExcToExc.json",
    model_template="Exp2Syn",
    syn_weight=6.0e-03,
    delay=0.5,
    target_sections=["apical"],
    distance_range=[2.0, 1.0e20],
)

cortcol_apicalsyns.add_edges(
    # Inh --> Exc connections
    source={"ei": "i"},
    target={"ei": "e"},
    connection_rule=lambda *_: np.random.randint(0, 6),
    dynamics_params="GABA_InhToExc.json",
    model_template="Exp2Syn",
    syn_weight=0.0002,
    delay=2.0,
    target_sections=["apical"],
    distance_range=[0.0, 50.0],
)

cortcol_apicalsyns.build()
cortcol_apicalsyns.save(output_dir="network_apicalsynapses")

# %% [markdown]
# ##### 1.2.2 External edges - inputs to the network
# To stimulate the network, create 100 external (virtual) cells connecting to all cell populations in the network. Use the `SpikeGenerator` to create a Poisson spike train with a frequency of 10 Hz and a duration of 3 seconds. Store the generated spike trains in [/inputs](inputs). Plot the spike trains to visualize the input to the network.

# %% [markdown]
# Generate spike trains for virtual cells:

# %%
import os
import numpy as np

# from scipy import io
# from bmtk.builder.networks import NetworkBuilder
from bmtk.analyzer.spike_trains import plot_raster
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator


# %%
def generate_Poisson_inputs(path2inputs_dir, virt_pop_params, N=100):
    """Generate Poisson inputs to the network.

    Parameters:
    -----------
    path2inputs_dir : str
        Path to the directory where the inputs will be saved.
    virt_pop_params : dict
        Parameters of the virtual population.

    Returns:
    --------
    None
    """
    # generate pre-synaptic spikes
    psg = PoissonSpikeGenerator()

    for i in range(N):
        psg.add(
            node_ids=[i],
            firing_rate=virt_pop_params["firing_rate"],
            times=virt_pop_params["times"],
            population="external",
        )

    psg.to_sonata(os.path.join(path2inputs_dir, "spikes.h5"))
    psg.to_csv(os.path.join(path2inputs_dir, "spikes.csv"))


# %%
folder_name = "inputs"
os.makedirs(folder_name, exist_ok=True)

virt_pop_params = {"firing_rate": 10.0, "times": (0.5, 3.0)}

generate_Poisson_inputs(folder_name, virt_pop_params)

# plot the spike trains
_ = plot_raster(
    spikes_file=os.path.join(folder_name, "spikes.h5"),
    title=f"Rate: {virt_pop_params['firing_rate']} Hz",
)

# %% [markdown]
# Add external nodes to the network:

# %%
from bmtk.builder.bionet.swc_reader import get_swc


def set_synapses(
    src, trg, section_names=("soma", "apical", "basal"), distance_range=(0.0, 1.0e20)
):
    trg_swc = get_swc(trg, morphology_dir="components/morphologies/", use_cache=True)

    sec_ids, seg_xs = trg_swc.choose_sections(
        section_names, distance_range, n_sections=1
    )
    sec_id, seg_x = sec_ids[0], seg_xs[0]
    # swc_id, swc_dist = trg_swc.get_swc_id(sec_id, seg_x)
    # coords = trg_swc.get_coords(sec_id, seg_x)

    return [sec_id, seg_x]


# %%
# cm = external.add_edges(
#     target=cortcol_allsyns.nodes(ei="e", model_type="biophysical", layer="VisL4"),
#     source=external.nodes(),
#     connection_rule=20,
#     # iterator="one_to_all",
#     dynamics_params="AMPA_ExcToExc.json",
#     model_template="Exp2Syn",
#     delay=2.0,
#     syn_weight=0.00041,
#     target_sections=["basal", "apical", "somatic"],
#     distance_range=[0.0, 50.0],
# )

# cm = external.add_edges(
#     target=cortcol_allsyns.nodes(ei="i", model_type="biophysical"),
#     source=external.nodes(),
#     connection_rule=10,
#     # iterator="one_to_all",
#     dynamics_params="AMPA_ExcToInh.json",
#     model_template="Exp2Syn",
#     delay=2.0,
#     syn_weight=0.00095,
#     target_sections=["basal", "somatic"],
#     distance_range=[0.0, 1e20],
# )

cm = external.add_edges(
    source=external.nodes(),
    target=cortcol_allsyns.nodes(ei="e", layer="VisL4"),
    connection_rule=20,
    # iterator='one_to_all',
    dynamics_params="AMPA_ExcToExc.json",
    model_template="Exp2Syn",
    delay=2.0,
    syn_weight=0.00041,
    target_sections=["basal", "apical", "somatic"],
    distance_range=[0.0, 50.0],
)

external.save(output_dir="network_allsynapses")

del cm

# %%
# cm = external_basalsyns.add_edges(
#     target=cortcol_basalsyns.nodes(ei="e", model_type="biophysical", layer="VisL4"),
#     source=external_basalsyns.nodes(),
#     connection_rule=20,
#     # iterator="one_to_all",
#     dynamics_params="AMPA_ExcToExc.json",
#     model_template="Exp2Syn",
#     delay=2.0,
#     syn_weight=0.00041,
#     target_sections=["basal", "somatic"],
#     distance_range=[0.0, 50.0],
# )

# cm = external_basalsyns.add_edges(
#     target=cortcol_basalsyns.nodes(ei="i", model_type="biophysical"),
#     source=external_basalsyns.nodes(),
#     connection_rule=10,
#     # iterator="one_to_all",
#     dynamics_params="AMPA_ExcToInh.json",
#     model_template="Exp2Syn",
#     delay=2.0,
#     syn_weight=0.00095,
#     target_sections=["basal", "somatic"],
#     distance_range=[0.0, 1e20],
# )

cm = external_basalsyns.add_edges(
    source=external_basalsyns.nodes(),
    target=cortcol_basalsyns.nodes(ei="e", layer="VisL4"),
    connection_rule=20,
    # iterator='one_to_all',
    dynamics_params="AMPA_ExcToExc.json",
    model_template="Exp2Syn",
    delay=2.0,
    syn_weight=0.00041,
    target_sections=["basal", "apical", "somatic"],
    distance_range=[0.0, 50.0],
)

external_basalsyns.save(output_dir="network_basalsynapses")

del cm

# %%
# cm = external_apicalsyns.add_edges(
#     target=cortcol_apicalsyns.nodes(ei="e", model_type="biophysical", layer="VisL4"),
#     source=external_apicalsyns.nodes(),
#     connection_rule=20,
#     # iterator="one_to_all",
#     dynamics_params="AMPA_ExcToExc.json",
#     model_template="Exp2Syn",
#     delay=2.0,
#     syn_weight=0.00041,
#     target_sections=["apical"],  # , "somatic"
#     distance_range=[0.0, 50.0],
# )

# cm = external_apicalsyns.add_edges(
#     target=cortcol_apicalsyns.nodes(ei="i", model_type="biophysical"),
#     source=external_apicalsyns.nodes(),
#     connection_rule=10,
#     connection_params={
#         "dist_cutoff": 1e20,
#     },
#     iterator="one_to_all",
#     dynamics_params="AMPA_ExcToInh.json",
#     model_template="Exp2Syn",
#     delay=2.0,
#     syn_weight=0.00095,
#     target_sections=["apical"],  # , "somatic"
#     distance_range=[0.0, 1e20],
# )

cm = external_apicalsyns.add_edges(
    source=external_apicalsyns.nodes(),
    target=cortcol_apicalsyns.nodes(ei="e", layer="VisL4"),
    connection_rule=20,
    # iterator='one_to_all',
    dynamics_params="AMPA_ExcToExc.json",
    model_template="Exp2Syn",
    delay=2.0,
    syn_weight=0.00041,
    target_sections=["basal", "apical", "somatic"],
    distance_range=[0.0, 50.0],
)

external_apicalsyns.save(output_dir="network_apicalsynapses")

del cm

# %% [markdown]
# ### 2. Generate the simulation config.json file:
# - The config.json file should include the following parameters:
#   - duration: 3 seconds
#   - dt: 0.1 ms
#   - reports: add 'ecp' recordings to the `reports` section to record the LFPs.
#       - `electrode_positions`: path to the csv file containing the electrode positions. File provided in [components/electrodes](components/electrodes).
#   - input: use the external input created in step 1.1.2
#   - output: save the simulation results in a folder named [outputs](outputs)
#
# `Note`: You can use `config.lfp.json` file as a template. You can modify it to suit your simulation needs.

# %%
# Load the existing config file
with open("config.lfp.json", "r") as f:
    config = json.load(f)

# Create config for basal synapses
basal_config = config.copy()
basal_config["manifest"]["$NETWORK_DIR"] = "$BASE_DIR/network_basalsynapses"
basal_config["manifest"]["$OUTPUT_DIR"] = "$BASE_DIR/output_basalsynapses"

with open("config.lfp_basal.json", "w") as f:
    json.dump(basal_config, f, indent=2)


# Create config for apical synapses
apical_config = config.copy()
apical_config["manifest"]["$NETWORK_DIR"] = "$BASE_DIR/network_apicalsynapses"
apical_config["manifest"]["$OUTPUT_DIR"] = "$BASE_DIR/output_apicalsynapses"

with open("config.lfp_apical.json", "w") as f:
    json.dump(apical_config, f, indent=2)

print("Generated configuration files for basal and apical synapses simulations.")

# %% [markdown]
# ### 3. Run the simulation:
# - Use `BioNet` to run the simulation with the generated config.json file. The simulation should be run for 3 seconds with a time step of 0.1 ms. The results should be saved in the outputs folder.
# - Remember to compile the [./components/mechanisms](components/mechanisms) before running the simulation. You can do this by running the following command in the terminal:
# ```bash
# ! cd components/mechanisms && nrnivmodl modfiles
# ```

# %%
from bmtk.simulator import bionet

bionet.reset()
conf = bionet.Config.from_json("config.lfp.json")
conf.build_env()

net = bionet.BioNetwork.from_config(conf)
sim = bionet.BioSimulator.from_config(conf, network=net)
sim.run()

# %%
bionet.reset()
conf = bionet.Config.from_json("config.lfp_basal.json")
conf.build_env()

net = bionet.BioNetwork.from_config(conf)
sim = bionet.BioSimulator.from_config(conf, network=net)
sim.run()

# %%
bionet.reset()
conf = bionet.Config.from_json("config.lfp_apical.json")
conf.build_env()

net = bionet.BioNetwork.from_config(conf)
sim = bionet.BioSimulator.from_config(conf, network=net)
sim.run()

# %% [markdown]
# ### 4. Plot the results:
# - Plot the raster plot of the network activity. The x-axis should represent time and the y-axis should represent the neuron index. Use different colors for different populations.
# - Create a 2D plot of the LFPs recorded by the linear probe. The x-axis should represent time and the y-axis should represent the electrode number.

# %% [markdown]
# Plot raster of network activity:

# %%
_ = plot_raster(
    config_file="config.lfp.json",
    spikes_file="output_allsynapses/spikes.csv",
    title="Raster Plot for All Synapses",
    group_by="layer",
)

# %%
_ = plot_raster(
    config_file="config.lfp_basal.json",
    spikes_file="output_basalsynapses/spikes.csv",
    title="Raster Plot for Basal Synapses",
    group_by="layer",
)

# %%
_ = plot_raster(
    config_file="config.lfp_apical.json",
    spikes_file="output_apicalsynapses/spikes.csv",
    title="Raster Plot for Apical Synapses",
    group_by="layer",
)

# %% [markdown]
# Plot the LFPs:

# %%
from bmtk.analyzer.ecp import plot_ecp

_ = plot_ecp(config_file="config.lfp.json", report_name="cortical_electrode")

# %%
_ = plot_ecp(config_file="config.lfp_basal.json", report_name="cortical_electrode")

# %%
_ = plot_ecp(config_file="config.lfp_apical.json", report_name="cortical_electrode")

# %% [markdown]
# ### 5. Calculate the CSD:
# - Calculate the CSD from the simulated LFPs using the delta-iCSD method. You can find the implementation of the delta-iCSD method in the [icsd_scripts](icsd_scripts) folder.

# %%
import sys
import quantities as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("icsd_scripts/")

import icsd

from get_csd_input_dict import get_csd_input_dict

delta_icsd_allsynapses = get_csd_input_dict(
    "output_allsynapses/cortical_electrode.h5",
)
delta_icsd_basal = get_csd_input_dict(
    "output_basalsynapses/cortical_electrode.h5",
)
delta_icsd_apical = get_csd_input_dict(
    "output_apicalsynapses/cortical_electrode.h5",
)

# %%
csd_dict = dict(
    delta_icsd=icsd.DeltaiCSD(**delta_icsd_allsynapses),
    delta_icsd_basal=icsd.DeltaiCSD(**delta_icsd_basal),
    delta_icsd_apical=icsd.DeltaiCSD(**delta_icsd_apical),
)

csd_raw = {"delta_icsd": [], "delta_icsd_basal": [], "delta_icsd_apical": []}
csd_smooth = {"delta_icsd": [], "delta_icsd_basal": [], "delta_icsd_apical": []}
# Iterate through the csd_dict and compute raw and smoothed CSD

for method, csd_obj in list(csd_dict.items()):
    csd_raw[method] = csd_obj.get_csd()  # num_channels x trial_duration
    csd_smooth[method] = csd_obj.filter_csd(
        csd_raw[method]
    )  # num_channels x trial_duration

# %%
import matplotlib.pyplot as plt

delta_icsd_dict = dict(
    delta_icsd=delta_icsd_allsynapses,
    delta_icsd_basal=delta_icsd_basal,
    delta_icsd_apical=delta_icsd_apical,
)

for method, csd_smooth in list(csd_smooth.items()):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    lfp_data = delta_icsd_dict[method]["lfp"]

    # plot LFP signal
    ax = axes[0]
    im = ax.imshow(
        np.array(lfp_data),
        origin="upper",
        vmin=-abs(lfp_data).max(),
        vmax=abs(lfp_data).max(),
        cmap="bwr",
    )
    ax.axis(ax.axis("tight"))
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("LFP (%s)" % lfp_data.dimensionality.string)
    ax.set_xticklabels([])
    ax.set_title("LFP")
    ax.set_ylabel("ch #")

    # plot raw csd estimate
    ax = axes[1]
    im = ax.imshow(
        np.array(csd_raw[method]),
        origin="upper",
        vmin=-abs(csd_raw[method]).max(),
        vmax=abs(csd_raw[method]).max(),
        cmap="jet_r",
    )
    ax.axis(ax.axis("tight"))
    ax.set_title(csd_obj.name)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("CSD (%s)" % csd_raw[method].dimensionality.string)
    ax.set_xticklabels([])
    ax.set_ylabel("ch #")

    # plot spatially filtered csd estimate
    ax = axes[2]
    im = ax.imshow(
        np.array(csd_smooth),
        origin="upper",
        vmin=-abs(csd_smooth).max(),
        vmax=abs(csd_smooth).max(),
        cmap="jet_r",
    )
    ax.axis(ax.axis("tight"))
    ax.set_title(csd_obj.name + ", filtered")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("CSD (%s)" % csd_smooth.dimensionality.string)
    ax.set_ylabel("ch #")
    ax.set_xlabel("timestep")

# %% [markdown]
#
