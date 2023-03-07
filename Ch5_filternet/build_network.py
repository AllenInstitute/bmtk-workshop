import numpy as np

from bmtk.builder.networks import NetworkBuilder


def get_coords_column(N, radius_min=0.0, radius_max=400.0):
    phi = 2.0 * np.pi * np.random.random([N])
    r = np.sqrt((radius_min ** 2 - radius_max ** 2) * np.random.random([N]) + radius_max ** 2)
    x = r * np.cos(phi)
    y = np.random.uniform(400.0, 500.0, size=N)
    z = r * np.sin(phi)
    return x, y, z


def get_coords_plane(ncells, size_x=240.0, size_y=120.0):
    xs = np.random.uniform(0.0, size_x, ncells)
    ys = np.random.uniform(0.0, size_y, ncells)
    return xs, ys


def exc_exc_rule(source, target, max_syns):
    """Connect rule for exc-->exc neurons, should return an integer 0 or greater. The number of
    connections will be weighted according to the difference between source and target cells'
    tuning_angle property"""
    if source['node_id'] == target['node_id']:
        # prevent a cell from synapsing with itself
        return 0

    # calculate the distance between tuning angles and use it to choose
    # number of connections using a binomial distribution.
    src_tuning = source['tuning_angle']
    trg_tuning = target['tuning_angle']
    tuning_dist = np.abs((src_tuning - trg_tuning + 180) % 360 - 180)
    probs = 1.0 - (np.max((tuning_dist, 10.0)) / 180.0)
    return np.random.binomial(n=max_syns, p=probs)


def others_conn_rule(source, target, max_syns, max_distance=300.0, sigma=60.0):
    """Connection rule for exc-->inh, inh-->exc, or inh-->inh connections. The number of connections
    will be based on the euclidian distance between source and target cell.
    """
    if source['node_id'] == target['node_id']:
        return 0

    dist = np.sqrt((source['x'] - target['x']) ** 2 + (source['z'] - target['z']) ** 2)
    if dist > max_distance:
        return 0

    prob = np.exp(-(dist / sigma) ** 2)
    return np.random.binomial(n=max_syns, p=prob)


def connect_lgn_cells(source, targets, max_targets, min_syns=1, max_syns=15, lgn_size=(240, 120),
                      l4_radius=400.0, ellipse=(100.0, 500.0)):
    # map the lgn cells from the plane to the circle
    x, y = source['x'], source['y']
    x = x / lgn_size[0] - 0.5
    y = y / lgn_size[1] - 0.5
    src_x = x * np.sqrt(1.0 - (y**2/2.0)) * l4_radius
    src_y = y * np.sqrt(1.0 - (x**2/2.0)) * l4_radius

    # Find (the indices) of all the target cells that are within the given ellipse, if there are more than max_targets
    # then randomly choose them
    a, b = ellipse[0]**2, ellipse[1]**2
    dists = [(src_x-t['x'])**2/a + (src_y-t['y'])**2/b for t in targets]
    valid_targets = np.argwhere(np.array(dists) <= 1.0).flatten()
    if len(valid_targets) > max_targets:
        valid_targets = np.random.choice(valid_targets, size=max_targets, replace=False)

    # create an array of all synapse count. Most targets with have 0 connection, except for the "valid_targets" which
    # will have between [min_syns, max_syns] number of connections.
    nsyns_arr = np.zeros(len(targets), dtype=np.int)
    for idx in valid_targets:
        nsyns_arr[idx] = np.random.randint(min_syns, max_syns)

    return nsyns_arr


def build_l4():
    l4 = NetworkBuilder('l4')

    # Add nodes
    x, y, z = get_coords_column(80)
    l4.add_nodes(
        N=80,
        model_type='point_neuron',
        model_template='nest:glif_lif_asc_psc',
        dynamics_params='Scnn1a_515806250_glif_lif_asc.json',
        x=x, y=y, z=z,
        tuning_angle=np.linspace(start=0.0, stop=360.0, num=80, endpoint=False),
        model_name='Scnn1a',
        ei_type='e'
    )

    x, y, z = get_coords_column(80)
    l4.add_nodes(
        N=80,
        model_type='point_neuron',
        model_template='nest:glif_lif_asc_psc',
        dynamics_params='Rorb_512332555_glif_lif_asc.json',
        x=x, y=y, z=z,
        model_name='Rorb',
        ei_type='e',
        tuning_angle=np.linspace(start=0.0, stop=360.0, num=80, endpoint=False),
    )

    x, y, z = get_coords_column(80)
    l4.add_nodes(
        N=80,
        model_type='point_neuron',
        model_template='nest:glif_lif_asc_psc',
        dynamics_params='Nr5a1_587862586_glif_lif_asc.json',
        x=x, y=y, z=z,
        model_name='Nr5a1',
        ei_type='e',
        tuning_angle=np.linspace(start=0.0, stop=360.0, num=80, endpoint=False),
    )

    x, y, z = get_coords_column(60)
    l4.add_nodes(
        N=60,
        model_type='point_neuron',
        model_template='nest:glif_lif_asc_psc',
        dynamics_params='Pvalb_574058595_glif_lif_asc.json',
        x=x, y=y, z=z,
        model_name='PValb',
        ei_type='i',
    )

    # Add recurrent edges
    l4.add_edges(
        source=l4.nodes(ei_type='e'),
        target=l4.nodes(ei_type='e'),
        connection_rule=exc_exc_rule,
        connection_params={'max_syns': 15},
        syn_weight=2.5,
        delay=2.0,
        dynamics_params='static_ExcToExc.json',
        model_template='static_synapse',
    )

    l4.add_edges(
        source=l4.nodes(ei_type='e'),
        target=l4.nodes(ei_type='i'),
        connection_rule=others_conn_rule,
        connection_params={'max_syns': 12},
        syn_weight=5.0,
        delay=2.0,
        dynamics_params='static_ExcToInh.json',
        model_template='static_synapse',
    )

    l4.add_edges(
        source=l4.nodes(ei_type='i'),
        target=l4.nodes(ei_type='e'),
        connection_rule=others_conn_rule,
        connection_params={'max_syns': 4},
        syn_weight=-6.5,
        delay=2.0,
        dynamics_params='static_InhToExc.json',
        model_template='static_synapse',
    )

    l4.add_edges(
        source=l4.nodes(ei_type='i'),
        target=l4.nodes(ei_type='i'),
        connection_rule=others_conn_rule,
        connection_params={'max_syns': 4},
        syn_weight=-3.0,
        delay=2.0,
        dynamics_params='static_InhToInh.json',
        model_template='static_synapse',
    )

    l4.build()
    l4.save(output_dir='network')
    return l4


def build_lgn(l4):
    lgn = NetworkBuilder('lgn')

    # Build Nodes
    x, y = get_coords_plane(50)
    lgn.add_nodes(
        N=50,
        x=x,
        y=y,
        model_type='virtual',
        model_template='lgnmodel:tON_TF8',
        dynamics_params='tON_TF8.json',
        ei_type='e'
    )

    x, y = get_coords_plane(50)
    lgn.add_nodes(
        N=50,
        x=x,
        y=y,
        model_type='virtual',
        model_template='lgnmodel:tOFF_TF8',
        dynamics_params='tOFF_TF8.json',
        ei_type='e'
    )

    # Build Edges
    lgn.add_edges(
        source=lgn.nodes(),
        target=l4.nodes(ei_type='e'),
        connection_rule=connect_lgn_cells,
        connection_params={'max_targets': 6},
        iterator='one_to_all',
        model_template='static_synapse',
        dynamics_params='static_ExcToExc.json',
        delay=2.0,
        syn_weight=11.0
    )

    lgn.add_edges(
        source=lgn.nodes(),
        target=l4.nodes(ei_type='i'),
        connection_rule=connect_lgn_cells,
        connection_params={'max_targets': 12, 'ellipse': (400.0, 400.0)},
        iterator='one_to_all',
        model_template='static_synapse',
        dynamics_params='static_ExcToInh.json',
        delay=2.0,
        syn_weight=13.0
    )

    lgn.build()
    lgn.save(output_dir='network')
    return lgn


if __name__ == '__main__':
    l4_net = build_l4()
    lgn_net = build_lgn(l4_net)