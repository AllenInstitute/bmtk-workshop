# %% convert checkpoint file to a model file
from time import time
start_time = time()

import argparse
import logging


# debug = True
debug = False

if debug:
    args = argparse.Namespace(
        # checkpoint_name="../Simulation_results/v1_65871/b_o8l2/OSI_DSI_checkpoints/ckpt-725",
        # checkpoint_name="../core_1_result/v1_65871_data_dir_GLIF_network_1/b_5vwo/Best_model/ckpt-941",
        checkpoint_name="../core_result/v1_65871/b_czuz/Intermediate_checkpoints/ckpt-150",
        network_dir="../GLIF_network/network"
        )
else:
    # let's use the first argument as the checkpoint name.
    parser = argparse.ArgumentParser(description="Convert a TF checkpoint file to a BMTK edge file.")
    parser.add_argument("checkpoint_name", type=str, help="The name of the checkpoint file.")
    # as an optional argument we can specify the network directory.
    # with default of "GLIF_network/network
    parser.add_argument("--network_dir", type=str, default="GLIF_network/network", help="The directory of the network files.")
    parser.add_argument("--suffix", type=str, default="checkpoint", help="The suffix to be added to the edge file.")
    args = parser.parse_args()


print("Importing the necessary libraries...")
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


print("Reading the variables from checkpoint...")
checkpoint_name = args.checkpoint_name
# print checkpoint name.
print(f"Checkpoint name: {checkpoint_name}")
ckpt = tf.train.load_checkpoint(checkpoint_name)
all_variables = ckpt.get_variable_to_shape_map()
all_variables


# calculate the new weight values, which is the recurrent weight times
# the voltage scale.
# all_variables["model/layer_with_weights-2/cell/voltage_scale/.ATTRIBUTES/VARIABLE_VALUE"]
# all_variables["model/layer_with_weights-1/_bkg_weights/.ATTRIBUTES/VARIABLE_VALUE"]


# get the value.
voltage_scale = ckpt.get_tensor("model/layer_with_weights-0/cell/voltage_scale/.ATTRIBUTES/VARIABLE_VALUE")

# also, the recurrent weight values
recurrent_weights = ckpt.get_tensor("model/layer_with_weights-0/cell/recurrent_weight_values/.ATTRIBUTES/VARIABLE_VALUE")
recurrent_weights = recurrent_weights.flatten()


# connection indices
indices = ckpt.get_tensor("model/layer_with_weights-0/cell/recurrent_indices/.ATTRIBUTES/VARIABLE_VALUE")
target_indices = indices[:, 0]
del indices

voltage_scale_edges = voltage_scale[target_indices]
del target_indices
# del voltage_scale

checkpoint_weights_sorted = recurrent_weights * voltage_scale_edges
del recurrent_weights
del voltage_scale_edges

# convert it to 64-bit
checkpoint_weights_sorted = checkpoint_weights_sorted.astype(np.float64)


# next, duplicate the edges file to create the basis for copying.
# orig_net = "../GLIF_network/network/v1_v1_edges.h5"
# target_net = "../GLIF_network/network/v1_v1_edges_checkpoint.h5"
orig_net = f"{args.network_dir}/v1_v1_edges.h5"
target_net = f"{args.network_dir}/v1_v1_edges_{args.suffix}.h5"

# first copy the file
print("Copying the edge file...")
import shutil
shutil.copy(orig_net, target_net)

# next, open the file in read and write mode
import h5py
f = h5py.File(target_net, "r+")


# Read necessary variables to for sorting the weights
print("Reading the necessary variables...")
orig_weights = f["/edges/v1_to_v1/0/syn_weight"][:]
source_ids = f["/edges/v1_to_v1/source_node_id"][:]
target_ids = f["/edges/v1_to_v1/target_node_id"][:]
orig_indices = np.stack([target_ids, source_ids], axis=1)
del source_ids
del target_ids

# define the sort method
def sort_indices(indices):
    max_ind = np.max(indices) + 1
    if np.iinfo(indices.dtype).max < max_ind * (max_ind + 1):
        indices = indices.astype(np.int64)
    q = indices[:, 0] * max_ind + indices[:, 1]
    sorted_ind = np.argsort(q)
    # sorted_arrays = list(map(lambda arr: arr[sorted_ind], [indices, *arrays]))
    # return tuple(sorted_arrays), sorted_ind
    return sorted_ind

print("Sorting the weights...")
sorted_ind = sort_indices(orig_indices)
del orig_indices
revert_ind = np.argsort(sorted_ind)
del sorted_ind
checkpoint_weights_reverted = checkpoint_weights_sorted[revert_ind]
del revert_ind
del checkpoint_weights_sorted

zerofrac = np.sum(checkpoint_weights_reverted == 0) / len(checkpoint_weights_reverted)

# here, let's check if the sign of the weights mostly agree.
print("Verifying the weights...")

non_agree_frac = np.sum(((orig_weights * checkpoint_weights_reverted) < 0)) / len(orig_weights)
# non_agree_frac = np.sum(np.sign(orig_weights) != np.sign(checkpoint_weights_reverted)) / len(orig_weights)
print(f"*** Percentage of weights that do not agree in sign: {(non_agree_frac * 100):.2f}% ***")
print("This should be 0% if the weights are correctly sorted.")
# print("If this is less than 1%, then it is likely that the weights are correctly sorted.")
# print("A small fraction may disagree due to the weights being zero.")
print(f"By the way, {zerofrac * 100:.1f}% of the weights became zero.")

# write the weights
print("Writing the weights...")
f["/edges/v1_to_v1/0/syn_weight"][:] = checkpoint_weights_reverted

# close the file
f.close()

print(f"Done. Time for processing: {time() - start_time:.1f} s.")




# %% deal with the bkg as well

start_time = time()
print("Processing BKG weights...")

# The BKG weights names are changed at one point, so let's try both.
try:
    bkg_weights = ckpt.get_tensor("model/layer_with_weights-0/cell/bkg_input_weights/.ATTRIBUTES/VARIABLE_VALUE")
    bkg_indices = ckpt.get_tensor("model/layer_with_weights-0/cell/bkg_input_indices/.ATTRIBUTES/VARIABLE_VALUE")
except RuntimeError: # for the case of the old model
    logging.warning("Old model detected. Reading the old variable names...")
    bkg_weights = ckpt.get_tensor("model/layer_with_weights-1/_bkg_weights/.ATTRIBUTES/VARIABLE_VALUE")
    bkg_indices = ckpt.get_tensor("model/layer_with_weights-1/_bkg_indices/.ATTRIBUTES/VARIABLE_VALUE")
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")
    raise e

bkg_target_indices = bkg_indices[:, 0]
del bkg_indices

bkg_voltag_scale_edges = voltage_scale[bkg_target_indices]
del bkg_target_indices
bkg_weights_sorted = bkg_weights * bkg_voltag_scale_edges


orig_net = f"{args.network_dir}/bkg_v1_edges.h5"
target_net = f"{args.network_dir}/bkg_v1_edges_{args.suffix}.h5"
# first copy the file
print("Copying the BKG edge file...")
import shutil
shutil.copy(orig_net, target_net)

# next, open the file in read and write mode
import h5py
f = h5py.File(target_net, "r+")


print("Reading the necessary variables...")
# orig_weights = f["/edges/bkg_to_v1/0/syn_weight"][:]
source_ids = f["/edges/bkg_to_v1/source_node_id"][:]
target_ids = f["/edges/bkg_to_v1/target_node_id"][:]
orig_indices = np.stack([target_ids, source_ids], axis=1)
del source_ids
del target_ids


print("Sorting the weights...")
sorted_ind = sort_indices(orig_indices)
del orig_indices
revert_ind = np.argsort(sorted_ind)
del sorted_ind
bkg_weights_reverted = bkg_weights_sorted[revert_ind]
del revert_ind
del bkg_weights_sorted


f["/edges/bkg_to_v1/0/syn_weight"] = bkg_weights_reverted

f.close()

print(f"Done. Time for processing bkg: {time() - start_time:.1f} s.")