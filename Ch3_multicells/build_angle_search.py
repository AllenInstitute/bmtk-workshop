import numpy as np
from bmtk.builder import NetworkBuilder

# Scnn1a: z-axis=3.7
# Rorb: z-axis=4.4
# Nr5a1: z-axis=4.04

def search_morphology():
    N = 5
    net = NetworkBuilder('test')
    net.add_nodes(
        N=N,
        model_type='biophysical',
        # morphology='Scnn1a_485510712_morphology.swc',
        # morphology='Rorb_486509958_morphology.swc',
        # morphology='Nr5a1_485507735_morphology.swc',
        morphology='Pvalb_473862421_morphology.swc',
        y=np.zeros(N),
        z=np.zeros(N),
        x=np.linspace(0.0, 1000.0, num=N),
        rotation_angle_xaxis=0.0,
        rotation_angle_yaxis=0.0, # np.random.uniform(0.0, 2*np.pi, size=N),
        rotation_angle_zaxis=np.linspace(0.0, 2*np.pi, num=N)
    )
    net.build()
    net.save_nodes(output_dir='network_angle_search')


def get_coords(N, radius_min=0.0, radius_max=400.0):
    phi = 2.0 * np.pi * np.random.random([N])
    r = np.sqrt((radius_min**2 - radius_max**2) * np.random.random([N]) + radius_max**2)
    x = r * np.cos(phi)
    y = np.random.uniform(0.0, 150.0, size=N)
    z = r * np.sin(phi)
    return x, y, z

def build_network():
    N = 20

    net = NetworkBuilder('test')
    
    x, y, z = get_coords(N)
    net.add_nodes(
        N=N,
        model_type='biophysical',
        morphology='Scnn1a_485510712_morphology.swc',
        x=x,
        y=y,
        z=z,
        rotation_angle_xaxis=np.zeros(N),
        rotation_angle_yaxis=np.random.uniform(0.0, 2*np.pi, size=N),
        rotation_angle_zaxis=np.full(N, 3.7)
    )
    
    x, y, z = get_coords(N)
    net.add_nodes(
        N=N,
        model_type='biophysical',
        morphology='Rorb_486509958_morphology.swc',
        x=x,
        y=y,
        z=z,
        rotation_angle_xaxis=np.zeros(N),
        rotation_angle_yaxis=np.random.uniform(0.0, 2*np.pi, size=N),
        rotation_angle_zaxis=np.full(N, 4.4)
    )
    
    x, y, z = get_coords(N)
    net.add_nodes(
        N=N,
        model_type='biophysical',
        morphology='Nr5a1_485507735_morphology.swc',
        x=x,
        y=y,
        z=z,
        rotation_angle_xaxis=np.zeros(N),
        rotation_angle_yaxis=np.random.uniform(0.0, 2*np.pi, size=N),
        rotation_angle_zaxis=np.full(N, 4.04)
    )
    
    x, y, z = get_coords(N)
    net.add_nodes(
        N=N,
        model_type='biophysical',
        morphology='Pvalb_473862421_morphology.swc',
        x=x,
        y=y,
        z=z,
        rotation_angle_xaxis=np.zeros(N),
        rotation_angle_yaxis=np.random.uniform(0.0, 2*np.pi, size=N),
        rotation_angle_zaxis=np.random.uniform(0.0, 2*np.pi, size=N)
    )

    net.build()
    net.save_nodes(output_dir='network_angle_search')


if __name__ == '__main__':
    # search_morphology()
    build_network()

