import os
import numpy as np
import json
import pickle
import pandas as pd
from datetime import datetime, timedelta
# from datetime import timedelta

from bmtk.builder import NetworkBuilder


lgn_models = json.load(open('lgn_models.json', 'r'))


def generate_column_positions(n_cells, depth_range, radial_range):
    radius_outer = radial_range[1]
    radius_inner = radial_range[0]

    phi = 2.0 * np.pi * np.random.random(n_cells)
    r = np.sqrt((radius_outer ** 2 - radius_inner ** 2) * np.random.random(n_cells) + radius_inner ** 2)
    x = r * np.cos(phi)
    z = r * np.sin(phi)
    y = (depth_range[1] - depth_range[0]) * np.random.random(n_cells) + depth_range[0]
    return x, y, z


def positions_grids(n_cells, x_grids=15, y_grids=10, size=(240.0, 120.0)):
    """Randomly generate x, y positions on a screen/rectangle of given "size" divided into evenly sized
    grids. The number of cells in each grid will be evenly distributed += 1 cell. But within each grid
    the distribution of coordinates are randomized.
    """

    n_grids = x_grids * y_grids
    tile_width = size[0] / x_grids
    tile_height = size[1] / y_grids

    # Keeps track of the total number of cells in each grid evenly divided, if n_cells is not divisible
    #  by n_grids then randomly distribute the remaining cells into different grids
    grid_cell_counts = np.full(y_grids * x_grids, int(n_cells / n_grids))
    grid_remainder = n_cells % n_grids
    if grid_remainder != 0:
        rand_idxs = np.random.choice(range(len(grid_cell_counts)), size=grid_remainder, replace=False)
        grid_cell_counts[rand_idxs] += 1
    grid_cell_counts = grid_cell_counts.reshape((x_grids, y_grids))

    # for each grid generate random coordinates
    xs = np.zeros(n_cells)
    ys = np.zeros(n_cells)
    x_indx, y_indx = 0, 0
    for i in range(x_grids):
        for j in range(y_grids):
            n = grid_cell_counts[i, j]
            xs[x_indx:(x_indx + n)] = np.random.uniform(i * tile_width, (i + 1) * tile_width, n)
            ys[y_indx:(y_indx + n)] = np.random.uniform(j * tile_height, (j + 1) * tile_height, n)
            x_indx += n
            y_indx += n

    return xs, ys


def convert_x_to_lindegs(xcoords):
    return np.tan(0.07 * np.array(xcoords) * np.pi / 180.) * 180.0 / np.pi


def convert_z_to_lindegs(zcoords):
    return np.tan(0.04 * np.array(zcoords) * np.pi / 180.) * 180.0 / np.pi


def get_selection_probability(src_type, lgn_models_subtypes_dictionary):
    current_model_subtypes = lgn_models_subtypes_dictionary[src_type[0:4]]['sub_types']
    current_model_probabilities = lgn_models_subtypes_dictionary[src_type[0:4]]['probabilities']
    lgn_model_idx = [i for i, model in enumerate(current_model_subtypes) if src_type == model][0]
    return current_model_probabilities[lgn_model_idx]



def select_lgn_sources(sources, target, lgn_mean, probability, poissonParameter, sON_ratio, centers_d_min,
                       centers_d_max, ON_OFF_w_min, ON_OFF_w_max, aspectRatio_min, aspectRatio_max, N_syn):
    # target_id = target.node_id
    # import math

    source_ids = [s.node_id for s in sources]

    # parametersDictionary = lgn_params
    # pop_name = [key for key in parametersDictionary if key in target['pop_name']][0]

    # Check if target supposed to get a connection and if not, then no need to keep calculating.
    if np.random.random() > probability:
        return [None]*len(source_ids)

    # if target_id % 250 == 0:
    #     logger.info("connection LGN cells to L4 cell #", target_id)

    # subfields_centers_distance_min = parametersDictionary[pop_name]['centers_d_min']
    # subfields_centers_distance_max = parametersDictionary[pop_name]['centers_d_max']
    subfields_centers_distance_L = centers_d_max - centers_d_min

    # subfields_ON_OFF_width_min = parametersDictionary[pop_name]['ON_OFF_w_min']  # 6.0 8.0 #10.0 #8.0 #8.0 #14.0 #15.0
    # subfields_ON_OFF_width_max = parametersDictionary[pop_name]['ON_OFF_w_max']  # 8.0 10.0 #12.0 #10.0 #15.0 #20.0
    subfields_ON_OFF_width_L = ON_OFF_w_max - ON_OFF_w_min

    # subfields_width_aspect_ratio_min = parametersDictionary[pop_name]['aspectRatio_min']  # 2.8  # 1.9 #1.4 #0.9 #1.0
    # subfields_width_aspect_ratio_max = parametersDictionary[pop_name]['aspectRatio_max']  # 3.0  # 2.0 #1.5 #1.1 #1.0
    subfields_width_aspect_ratio_L = aspectRatio_max - aspectRatio_min

    x_position_lin_degrees = convert_x_to_lindegs(target['x'])
    y_position_lin_degrees = convert_z_to_lindegs(target['z'])

    vis_x = lgn_mean[0] + ((x_position_lin_degrees))  # - l4_mean[0]) / l4_dim[0]) * lgn_dim[0]
    vis_y = lgn_mean[1] + ((y_position_lin_degrees))  # - l4_mean[2]) / l4_dim[2]) * lgn_dim[1]

    ellipse_center_x0 = vis_x  # tar_cells[tar_gid]['vis_x']
    ellipse_center_y0 = vis_y  # tar_cells[tar_gid]['vis_y']

    tuning_angle = float(target['tuning_angle'])
    tuning_angle = None if np.isnan(tuning_angle) else tuning_angle
    #tuning_angle = None if math.isnan(target['tuning_angle']) else target['tuning_angle']
    if tuning_angle is None:
        ellipse_b0 = (ON_OFF_w_min + np.random.uniform(0.0, 1.0) * subfields_ON_OFF_width_L) / 2.0  # Divide by 2 to convert from width to radius.
        ellipse_b0 = 2.5 * ellipse_b0  # 1.5 * ellipse_b0
        ellipse_a0 = ellipse_b0  # ellipse_b0
        top_N_src_cells_subfield = 15  # 20
        ellipses_centers_halfdistance = 0.0
        tuning_angle_value = 0.0
    else:
        tuning_angle_value = float(tuning_angle)
        ellipses_centers_halfdistance = (centers_d_min + np.random.uniform(0.0, 1.0)* subfields_centers_distance_L) / 2.0
        ellipse_b0 = (ON_OFF_w_min + np.random.uniform(0.0, 1.0) * subfields_ON_OFF_width_L) / 2.0  # Divide by 2 to convert from width to radius.
        ellipse_a0 = ellipse_b0 * (aspectRatio_min + np.random.uniform(0.0, 1.0) * subfields_width_aspect_ratio_L)
        ellipse_phi = tuning_angle_value + 180.0 + 90.0  # Angle, in degrees, describing the rotation of the canonical ellipse away from the x-axis.
        ellipse_cos_mphi = np.cos(-np.radians(ellipse_phi))
        ellipse_sin_mphi = np.sin(-np.radians(ellipse_phi))
        top_N_src_cells_subfield = 8  # 10 #9

        ###############################################################################################################
        # probability_sON = parametersDictionary[pop_name]['sON_ratio']
        if np.random.random() < sON_ratio:
            cell_sustained_unit = 'sON_'
        else:
            cell_sustained_unit = 'sOFF_'

    cell_TF = np.random.poisson(poissonParameter)
    while cell_TF <= 0:
        cell_TF = np.random.poisson(poissonParameter)

    sON_subunits = np.array([1., 2., 4., 8.])
    sON_sum = np.sum(abs(cell_TF - sON_subunits))
    p_sON = (1 - abs(cell_TF - sON_subunits) / sON_sum) / (len(sON_subunits) - 1)

    sOFF_subunits = np.array([1., 2., 4., 8., 15.])
    sOFF_sum = np.sum(abs(cell_TF - sOFF_subunits))
    p_sOFF = (1 - abs(cell_TF - sOFF_subunits) / sOFF_sum) / (len(sOFF_subunits) - 1)

    tOFF_subunits = np.array([4., 8., 15.])
    tOFF_sum = np.sum(abs(cell_TF - tOFF_subunits))
    p_tOFF = (1 - abs(cell_TF - tOFF_subunits) / tOFF_sum) / (len(tOFF_subunits) - 1)

    # to match previous algorithm reorganize source cells by type
    cell_type_dict = {}
    for lgn_model in lgn_models:
        # cell_type_dict[lgn_model] = [
        #     (src_id, src_dict) for src_id, src_dict in zip(source_ids, sources) if src_dict['model_name'] == lgn_model
        # ]
        cell_type_dict[lgn_model] = [(src.node_id, src) for src in sources if src['model_name'] == lgn_model]


    lgn_models_subtypes_dictionary = {
        'sON_': {'sub_types': ['sON_TF1', 'sON_TF2', 'sON_TF4', 'sON_TF8'], 'probabilities': p_sON},
        'sOFF': {'sub_types': ['sOFF_TF1', 'sOFF_TF2', 'sOFF_TF4', 'sOFF_TF8', 'sOFF_TF15'], 'probabilities': p_sOFF},
        'tOFF': {'sub_types': ['tOFF_TF4', 'tOFF_TF8', 'tOFF_TF15'], 'probabilities': p_tOFF},
    }

    ##################################################################################################################

    # For this target cell, if it has tuning, select the input cell types
    # Note these parameters will not matter if the cell does not have tuning but are calculated anyway
    # Putting it here instead of previous if-else statement for clarity
    # cumulativeP = np.cumsum(connectivityRatios[pop_name]['probabilities'])
    # lgn_model_idx = np.where((np.random.random() < np.array(cumulativeP)) == True)[0][0]
    # sustained_Subunit = connectivityRatios[pop_name]['lgn_models'][lgn_model_idx][0]
    # transient_Subunit = connectivityRatios[pop_name]['lgn_models'][lgn_model_idx][1]
    src_cells_selected = {}
    for src_type in cell_type_dict.keys():
        src_cells_selected[src_type] = []

        if tuning_angle is None:
            ellipse_center_x = ellipse_center_x0
            ellipse_center_y = ellipse_center_y0
            ellipse_a = ellipse_a0
            ellipse_b = ellipse_b0
        else:
            if ('tOFF_' in src_type[0:5]):
                ellipse_center_x = ellipse_center_x0 + ellipses_centers_halfdistance * ellipse_sin_mphi
                ellipse_center_y = ellipse_center_y0 + ellipses_centers_halfdistance * ellipse_cos_mphi
                ellipse_a = ellipse_a0
                ellipse_b = ellipse_b0
            elif ('sON_' in src_type[0:5] or 'sOFF_' in src_type[0:5]):
                ellipse_center_x = ellipse_center_x0 - ellipses_centers_halfdistance * ellipse_sin_mphi
                ellipse_center_y = ellipse_center_y0 - ellipses_centers_halfdistance * ellipse_cos_mphi
                ellipse_a = ellipse_a0
                ellipse_b = ellipse_b0
            else:
                # Make this a simple circle.
                ellipse_center_x = ellipse_center_x0
                ellipse_center_y = ellipse_center_y0
                # Make the region from which source cells are selected a bit smaller for the transient_ON_OFF cells,
                # since each source cell in this case produces both ON and OFF responses.
                ellipse_b = ellipses_centers_halfdistance/2.0 #0.01 #ellipses_centers_halfdistance + 1.0*ellipse_b0 #0.01 #0.5 * ellipse_b0 # 0.8 * ellipse_b0
                ellipse_a = ellipse_b0 #0.01 #ellipse_b0


        # Find those source cells of the appropriate type that have their visual space coordinates within the ellipse.
        for src_id, src_dict in cell_type_dict[src_type]:
            # print(src_dict)

            x, y = (src_dict['x'], src_dict['y'])

            x = x - ellipse_center_x
            y = y - ellipse_center_y

            x_new = x
            y_new = y
            if tuning_angle is not None:
                x_new = x * ellipse_cos_mphi - y * ellipse_sin_mphi
                y_new = x * ellipse_sin_mphi + y * ellipse_cos_mphi

            if ((x_new / ellipse_a) ** 2 + (y_new / ellipse_b) ** 2) <= 1.0:
                if tuning_angle is not None:
                    if src_type == 'sONsOFF_001' or src_type == 'sONtOFF_001':
                        src_tuning_angle = float(src_dict['tuning_angle'])
                        delta_tuning = abs(abs(abs(180.0 - abs(tuning_angle_value - src_tuning_angle) % 360.0) - 90.0) - 90.0)
                        if delta_tuning < 15.0:
                            src_cells_selected[src_type].append(src_id)

                    # elif src_type in ['sONtOFF_001']:
                    #     src_cells_selected[src_type].append(src_id)

                    elif cell_sustained_unit in src_type[:5]:
                        selection_probability = get_selection_probability(src_type, lgn_models_subtypes_dictionary)
                        if np.random.random() < selection_probability:
                            src_cells_selected[src_type].append(src_id)

                    elif 'tOFF_' in src_type[:5]:
                        selection_probability = get_selection_probability(src_type, lgn_models_subtypes_dictionary)
                        if np.random.random() < selection_probability:
                            src_cells_selected[src_type].append(src_id)

                else:
                    if (src_type == 'sONsOFF_001' or src_type == 'sONtOFF_001'):
                        src_cells_selected[src_type].append(src_id)
                    else:
                        selection_probability = get_selection_probability(src_type, lgn_models_subtypes_dictionary)
                        if np.random.random() < selection_probability:
                            src_cells_selected[src_type].append(src_id)

    select_cell_ids = [id for _, selected in src_cells_selected.items() for id in selected]

    # if len(select_cell_ids) > 30:
    #     select_cell_ids = np.random.choice(select_cell_ids, 30, replace=False)
    nsyns_ret = [N_syn if id in select_cell_ids else None for id in source_ids]
    return nsyns_ret


def build_v1(fraction=0.01):
    models = json.load(open('point_neuron_models.json', 'r'))

    v1 = NetworkBuilder('v1')
    radial_range = (1.0, 845.0)
    for layer, layer_dict in models.items():
        depth_range = layer_dict['depth_range']
        for model_props in layer_dict['models']:
            n_cells = int(model_props['N']*fraction)
            x, y, z = generate_column_positions(n_cells, depth_range=depth_range, radial_range=radial_range)
            tuning_angle = np.linspace(0.0, 360.0, n_cells, endpoint=False)

            v1.add_nodes(
                N=n_cells,
                layer=layer,
                x=x,
                y=y,
                z=z,
                tuning_angle=tuning_angle,
                model_name=model_props['model_name'],
                cell_line=model_props['cell_line'],
                ei=model_props['ei'],
                model_type=model_props['model_type'],
                model_template=model_props['model_template'],
                dynamics_params=model_props['dynamics_params']
            )

    v1.build()
    v1.save(output_dir='network')
    return v1


def build_lgn(v1, fraction=1.0):
    cell_models = json.load(open('point_neuron_models.json', 'r'))
    # lgn_models = json.load(open('lgn_models.json', 'r'))
    x_grids, y_grids = 15, 10
    field_size = (240.0, 120.0)

    lgn = NetworkBuilder('lgn')
    for model, params in lgn_models.items():
        n_cells = int(params['N']*fraction)
        x, y = positions_grids(n_cells, x_grids=x_grids, y_grids=y_grids, size=field_size)
        tuning_angle = [np.NaN]*n_cells if not params['tuning_angle'] else np.linspace(0.0, 360.0, n_cells, endpoint=False)
        size_range = params['size_range']

        # if model not in ['sONtOFF_001', 'sONsOFF_001']:
        # if model not in ['sONsOFF_001']:
        #     continue


        lgn.add_nodes(
            N=n_cells,
            model_name=params['model_name'],
            subtype=params['subtype'],
            model_type=params['model_type'],
            model_template=params['model_template'],
            dynamics_params=params['dynamics_params'],
            non_dom_params=params.get('non_dom_params', None),
            x=x,
            y=y,
            tuning_angle=tuning_angle,
            spatial_size=np.random.uniform(size_range[0], size_range[1], n_cells),
            sf_sep=params.get('sf_sep', None),
            jitter_lower=0.975,
            jitter_upper=1.025
        )

    connections = json.load(open('lgn_v1_connections.json', 'r'))
    for layer, layer_dict in cell_models.items():
        for model_props in layer_dict['models']:
            model_name = model_props['model_name']
            cell_line = model_props['cell_line']

            conn_props = connections[cell_line]['connection_params']
            conn_props['lgn_mean'] = (field_size[0]/2.0, field_size[1]/2.0)

            edge_props = connections[cell_line]['edge_types_params']
            lgn.add_edges(
                source=lgn.nodes(),
                target=v1.nodes(model_name=model_name),
                connection_rule=select_lgn_sources,
                connection_params=conn_props,
                iterator='all_to_one',
                **edge_props
            )

    lgn.build()
    lgn.save(output_dir='network')
    return lgn


if __name__ == '__main__':
    start = datetime.now()
    v1 = build_v1(fraction=0.05)
    lgn = build_lgn(v1, fraction=1.0)
    end = datetime.now()
    print('build time:', timedelta(seconds=(end - start).total_seconds()))