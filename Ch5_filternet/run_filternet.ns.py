import os, sys
from bmtk.simulator import filternet


def run(config_file):
    config = filternet.Config.from_json(config_file)
    config.build_env()

    net = filternet.FilterNetwork.from_config(config)
    sim = filternet.FilterSimulator.from_config(config, net)
    sim.run()


if __name__ == '__main__':
    run('config.filternet_ns.json')
