GLIF (Point)_network
====================

Contains the files necessary to run a full simulation of the point-neuron V1 network with thalamacortical (LGN) and background (BKG) inputs. Is designed
to run using the Allen Institute [Brain Modeling Toolkit](https://github.com/AllenInstitute/bmtk); but the network, input and config files are in the 
[SONATA data format](https://github.com/AllenInstitute/sonata) for use with other tools that support SONATA.


Requirements
------------
Python 2.7 or 3.6+
NEST 12.0+
BMTK 0.0.8+


Running the simulation
----------------------
For most clusters the simulation can be started using the following command inside a SLURM/MOAB/etc. script (replace N with the number of cores to use)
$ mpirun -np N python run_pointnet.py config.json


Simulation output
-----------------
By default once the simulation has started running it will create an "output/" folder containing the simulation results. (WARNING: if an existing "output/" folder from
a previous simulation already exists bmtk will overwrite it). 

"output/log.txt" will keep a running tally of the simulation and can be good to check-on (eg $ tail -f output/log.txt) while the simulation is running. When completed
the spike trains for all the V1 cells will be stored in "output/spikes.h5". The spikes are stored according to the SONATA format 
(https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#spike-file), and can be read using tools like pysonata, libsonata, or any hdf5 
API (eg h5py).

You can change where and how the output is stored in "simulation_config.json" under the "outputs" section.


Modifying the simulation
------------------------
Circuit and simulation parameters to instantiate and run a given simulation is defined under "config.json", and can be edited under a standard text editor to modify 
the simulation.

Information about the configuration files can be found in the [SONATA documentation](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#tying-it-all-together---the-networkcircuit-config-file).
Also see [here](https://github.com/AllenInstitute/sonata/tree/master/examples) and [here](https://github.com/AllenInstitute/bmtk/tree/develop/docs/examples) for various
examples of different types of simulations.


Directory Structure
-------------------
* point_components/ - model files used to instantiate individual cells and synapses
* network/ - contains the SONATA formated network files for V1, LGN and BKG nodes plus their connectivity 
* inputs/ - spike train files used to drive the LGN and BKG inputs.
* output/ - results from an individual simulation.


Perturbation Simulations
------------------------
"config_activate_e6Nstr1.json" and "config_silence_e6Nstr1.json" are example simulations to show how subpopulations of cells can be inhibited or activated, respectively,
as described  in the BMTK Paper Figure 8 (Dai et al. 2020). To run these simulations

$ mpirun -np N python run_pointnet.py config_silence_e6Nstr1.json

or

$ mpirun -np N python run_pointnet.py config_activate_e6Nstr1.json



