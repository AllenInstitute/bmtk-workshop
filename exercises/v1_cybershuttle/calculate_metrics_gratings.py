import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as pylab
import os
import json

def calculateFiringRate(gids, ts, numNrns, gray_screen = 0):

    print ("Calculating Firing Rate")
    gids = gids[np.where(ts > gray_screen)[0]]

    gid_bins = np.arange(0 - 0.5, numNrns + 0.5, 1)

    hist, bins = np.histogram(gids, bins=gid_bins)

    if gray_screen > 0:
        mean_firing_rates = hist / (  (3000. - gray_screen) / 1000.)
    else:
        mean_firing_rates = hist / (3000. / 1000.)
    return mean_firing_rates


def calculate_Rates_DF(numNrns, trials = 10,  oris = np.arange(0, 360, 45)):

    Rates_DF = pd.DataFrame(index = range(numNrns * len(oris)), columns= ['DG_angle', 'node_id', 'Avg_rate(Hz)', 'SD_rate(Hz)'])

    for i, ori in enumerate(oris):
        firingRatesTrials = np.zeros((trials, numNrns))

        for trial in range(trials):

            spikes_file_name = 'output_all_directions/spikes_driftingGratings_ori' + str(float(ori)) + '_trial' + str(trial) + '.txt'

            spikes = np.loadtxt(spikes_file_name, unpack=True)
            ts, gids = spikes
            gids = gids.astype(int)


            firingRates = calculateFiringRate(gids, ts, numNrns, gray_screen = 500.)
            print (spikes_file_name)

            firingRatesTrials[trial, :] = firingRates


        Rates_DF.loc[i*numNrns:(i+1)*numNrns-1, 'DG_angle']     = ori
        Rates_DF.loc[i*numNrns:(i+1)*numNrns-1, 'node_id']      = np.arange(numNrns)
        Rates_DF.loc[i*numNrns:(i+1)*numNrns-1, 'Avg_rate(Hz)'] = np.mean(firingRatesTrials, axis= 0)
        Rates_DF.loc[i*numNrns:(i+1)*numNrns-1, 'SD_rate(Hz)']  = np.std(firingRatesTrials, axis= 0)


    Rates_DF.to_csv('output_all_directions/Rates_DF.csv', sep = ' ', index = False)


def calculate_OSI_DSI_from_DF(Rates_DF):

    numNrns = Rates_DF['node_id'].max() + 1
    OSI_DSI_DF = pd.DataFrame(index = range(numNrns), columns= ['node_id', 'DSI', 'OSI', 'preferred_angle', 'max_mean_rate(Hz)'])
    ori = np.arange(0, 360, 45)

    zz = 0
    for i in range(numNrns):
        rates = np.array(Rates_DF.loc[Rates_DF['node_id'] == i, 'Avg_rate(Hz)'])
        angles = np.array(Rates_DF.loc[Rates_DF['node_id'] == i, 'DG_angle'])
        if i%2000 == 0:
            print ('cell: ', i)

        max_rate_ind = np.argmax(rates)
        if np.size(max_rate_ind) > 1:
            print (i, rates)
        if max(rates) > 0.0:

            # DSI = (rates[max_rate_ind] - rates[ (max_rate_ind + 4) % 8]) / (rates[max_rate_ind] + rates[ (max_rate_ind + 4) % 8])
            numerator = [rates[k] * np.exp(1j* np.deg2rad(theta)) for k, theta in enumerate(angles)]
            denominator = rates
            DSI = np.abs(np.array(numerator).sum()) / denominator.sum()

            # OSI = abs(np.sum(rates*np.exp(2j*ori)) / sum(rates))
            numerator = [rates[k] * np.exp(2*1j* np.deg2rad(theta)) for k, theta in enumerate(angles)]
            denominator = rates
            OSI = np.abs(np.array(numerator).sum()) / denominator.sum()



            pref_ang = angles[max_rate_ind]
        else:
            DSI = np.nan
            OSI = np.nan
            pref_ang = np.nan

        OSI_DSI_DF.loc[i, 'node_id'] = i
        OSI_DSI_DF.loc[i, 'DSI'] = DSI
        OSI_DSI_DF.loc[i, 'OSI'] = OSI
        OSI_DSI_DF.loc[i, 'preferred_angle'] = pref_ang
        OSI_DSI_DF.loc[i, 'max_mean_rate(Hz)'] = np.max(rates)
        OSI_DSI_DF.loc[i, 'Avg_Rate(Hz)'] = np.mean(rates)


    OSI_DSI_DF.to_csv('output_all_directions/OSI_DSI_DF.csv', sep = ' ', index = False)



def calculate_metrics_gratings():

    trials = 10
    oris = np.arange(0, 360, 45)

    nodes_file_name = 'network/v1_nodes.h5'
    nodes_h5 = h5py.File(nodes_file_name, 'r')
    numNrns = len(nodes_h5['/nodes']['v1']['node_id'])

    #nodes_DF = pd.read_csv(nodes_file_name, sep=' ')
    #numNrns = len(nodes_DF)

    calculate_Rates_DF(numNrns, trials = trials, oris = oris)
    print ("Done_Rates_DF!")

    Rates_DF = pd.read_csv('output_all_directions/Rates_DF.csv', sep=' ', index_col=False)
    calculate_OSI_DSI_from_DF(Rates_DF)
    print ("Done with all!")
