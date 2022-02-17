#!/usr/bin/env python
# coding: utf-8

SIMULATED_ = False
PLOT_FIGURE = True
SAVE_FIGURE = True
SAVE_VIDEO = False
LOAD_CAM_DATA = False
PERF_FILENAME = 'stats_carrada_spike/carrada_stats_sigma_new_2.csv'

## Imports
import os
import cv2
import json
import time
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.feature import peak_local_max
import sys
from sklearn.cluster import KMeans
from collections import Counter

if SIMULATED_:
    from components.SimulatedRadar import SimulatedRadar
else:
    from components.RadarLoader import RadarLoader
    seq_name = sys.argv[1]
# get_ipython().run_line_magic('run', "carrada_utils/scripts/set_path.py '/Volumes/SAMSUNG/Carrada/'")


from brian2 import *

# Network params
INTENSITY_EXC = 1.5 #float(sys.argv[1])
INTENSITY_INH = -15.0 #float(sys.argv[2])
SIGMA_EXC = 1.0 # float(sys.argv[1]) # 1.0 # 5?
SIGMA_INH = 10.0 # float(sys.argv[2]) # 10.0

CONN_CUTOFF=120
TAU = 200 # ms
LOWER_VOLT_THRESH = -1.0
GRID_DISTANCE=100.0

# Input params
if SIMULATED_:
    INPUT_WEIGHT = 0.1
else:
    INPUT_WEIGHT = 1.0
DATA_FREQ_MULTIPLIER=2.0
TIME_BETWEEN_FRAMES = 100.0

# Simulation input params
NOISE_FACTOR = 0.1

# Input and network size
SIZE_X = 64
SIZE_Y = 64

# Evaluation params
CUTOFF_DIST_EVAL = 5.0
SIGMA_DIST_EVAL = 3.0

# In[4]:


LEN_DATA = 0
if SIMULATED_:
    sim_data_loader = SimulatedRadar(noise_factor=NOISE_FACTOR)
    spiking_data, spiking_indices = sim_data_loader.get_random_datastream_spiking_brian2(size_x=SIZE_X)
    ds = sim_data_loader.get_random_datastream()
    gt_paths = sim_data_loader.get_paths()
    LEN_DATA = len(ds)
    print(f"Simulation data is loaded. Total number of frames: {LEN_DATA}")
else:
    if LOAD_CAM_DATA:
        data_loader = RadarLoader(seq_name)
        raw_camera_data = data_loader.get_color_image_datastream(resize=(SIZE_X,SIZE_Y))
        diff_datastream = data_loader.get_range_angle_stream_data_differentiated_normalized(resize=(SIZE_X,SIZE_Y), clip_and_normalize=True)
    raw_data = np.load(f"carrada_processed/{seq_name}_raw_data.npy")
    spiking_data = np.load(f"carrada_processed/{seq_name}_spiking_data.npy")
    spiking_indices = np.load(f"carrada_processed/{seq_name}_spiking_indices.npy")
    dense_ = np.load(f"carrada_processed/{seq_name}_dense_.npy")
    LEN_DATA = len(raw_data-1)
    print(f"CARRADA data is loaded. Total number of frames: {LEN_DATA}")


# In[5]:


def cfar_implementation(frame):
    # Taken from https://qiita.com/harmegiddo/items/8a7e1b4b3a899a9e1f0c
    # fields
    if SIMULATED_:
        GUARD_CELLS = 5
        BG_CELLS = 10
        ALPHA = 50
    else:
        GUARD_CELLS = 5
        BG_CELLS = 10
        ALPHA = 5
    CFAR_UNITS = 1 + (GUARD_CELLS * 2) + (BG_CELLS * 2)
    HALF_CFAR_UNITS = int(CFAR_UNITS/2) + 1
    OUTPUT_IMG_DIR = "./"

    # preparing
    inputImg = frame
    estimateImg = np.zeros((inputImg.shape[0], inputImg.shape[1], 1), np.uint8)

    # search
    for i in range(inputImg.shape[0] - CFAR_UNITS):
        center_cell_x = i + BG_CELLS + GUARD_CELLS
        for j in range(inputImg.shape[1] - CFAR_UNITS):
            center_cell_y = j  + BG_CELLS + GUARD_CELLS
            average = 0
            for k in range(CFAR_UNITS):
                for l in range(CFAR_UNITS):
                    if (k >= BG_CELLS) and (k < (CFAR_UNITS - BG_CELLS)) and (l >= BG_CELLS) and (l < (CFAR_UNITS - BG_CELLS)):
                        continue
                    average += inputImg[i + k, j + l]
            average /= (CFAR_UNITS * CFAR_UNITS) - ( ((GUARD_CELLS * 2) + 1) * ((GUARD_CELLS * 2) + 1) )

            if inputImg[center_cell_x, center_cell_y] > (average * ALPHA):
                estimateImg[center_cell_x, center_cell_y] = 1# inputImg[center_cell_x, center_cell_y]

    # output
    tmpName = OUTPUT_IMG_DIR + "est.png"
    return estimateImg

def get_euclidian_dist(x_1,y_1,x_2,y_2):
    theta_1 = math.radians((x_1 - SIZE_X/2) / (SIZE_X/2) * 90.0)
    theta_2 = math.radians((x_2 - SIZE_X/2) / (SIZE_X/2) * 90.0)
    R_1 = (SIZE_Y-y_1)/SIZE_Y*50.0
    R_2 = (SIZE_Y-y_2)/SIZE_Y*50.0
    loc_x_1 = R_1 * math.sin(theta_1)
    loc_y_1 = R_1 * math.cos(theta_1)
    loc_x_2 = R_2 * math.sin(theta_2)
    loc_y_2 = R_2 * math.cos(theta_2)
    
    err_dist = np.sqrt((loc_x_1 - loc_x_2)**2 + (loc_y_1 - loc_y_2)**2)
    return err_dist



start_scope()
tau = TAU*ms

eqs = '''
dv/dt = -v/tau : 1
x : meter
y : meter
'''

N = SIZE_X*SIZE_Y
rows = SIZE_X
cols = SIZE_Y
# initialize the grid positions
grid_dist = GRID_DISTANCE*umeter
sigma_exc = SIGMA_EXC*grid_dist
sigma_inh = SIGMA_INH*grid_dist
intensity_exc = INTENSITY_EXC/(2*math.pi*SIGMA_EXC**2)
intensity_inh = INTENSITY_INH/(2*math.pi*SIGMA_INH**2)
conn_distance = CONN_CUTOFF*grid_dist
lower_volt_thresh = LOWER_VOLT_THRESH
G = NeuronGroup(N, eqs, threshold='v>1.0', reset='v = 0.0', method='exact')
G.x = '(i // rows) * grid_dist - rows/2.0 * grid_dist'
G.y = '(i % rows) * grid_dist - cols/2.0 * grid_dist'

# Synapses
S = Synapses(G, G, 'w : 1', on_pre='v_post += w')
S.connect(condition='(i!=j) and (sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) < conn_distance)')
# Weight varies with distance
S.w = '(intensity_exc*exprel(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigma_exc**2))))'

S_inhib = Synapses(G, G, 'w : 1', on_pre='v_post = clip((w + v_post), lower_volt_thresh, inf)') # TODO: inf oder eins??
S_inhib.connect(condition='(i!=j) and (sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) < conn_distance)')
# Weight varies with distance
S_inhib.w =  '(intensity_inh*exprel(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigma_inh**2))))'


G_input = SpikeGeneratorGroup(N, spiking_indices, spiking_data*ms)
weight=INPUT_WEIGHT# 0.1 works
S_input = Synapses(G_input, G, on_pre='v += weight')
S_input.connect(i='j')

M_spike = SpikeMonitor(G)
M_spike_input = SpikeMonitor(G_input)

print("Model built with:")
print(f"SIGMA_exc:{SIGMA_EXC} | SIGMA_inh:{SIGMA_INH}")
print(f"I_exc:{INTENSITY_EXC} | I_inh:{INTENSITY_INH}")
print(f"Tau: {TAU} | Lower_volt_thresh: {LOWER_VOLT_THRESH}")
#M_state = StateMonitor(G, 'v', record=True)


# In[13]:


run((LEN_DATA-1)*TIME_BETWEEN_FRAMES*ms, report="stdout", report_period=30000*ms)


# In[14]:



kalman_input_1 = []
kalman_times_1 = []
kalman_input_2 = []
kalman_times_2 = []
for qqq in range(0,LEN_DATA-1):
    if SIMULATED_:
        cfar_out = cfar_implementation(ds[qqq])
    else:
        cfar_out = cfar_implementation(raw_data[qqq])
    # https://stackoverflow.com/questions/58115511/determine-the-center-of-the-cluster-with-the-most-points?rq=1

    # https://stackoverflow.com/questions/58115511/determine-the-center-of-the-cluster-with-the-most-points?rq=1
    if len(np.argwhere(cfar_out > 0)) > 1:
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(np.argwhere(cfar_out > 0))

        counter = Counter(kmeans.labels_)
        largest_cluster_idx = np.argmax(counter.values())
        largest_cluster_center = kmeans.cluster_centers_[largest_cluster_idx ]

        kmean_points = kmeans.cluster_centers_.astype(int)[:,0:2]
        kmean_dist_ = get_euclidian_dist(kmean_points[0][0],kmean_points[0][1],kmean_points[1][0],kmean_points[1][1])
        kalman_input_1.append(kmean_points[0])
        kalman_times_1.append(qqq)
        if kmean_dist_ > 5:
            kalman_input_2.append(kmean_points[1])
            kalman_times_2.append(qqq)

    

measurements = np.asarray(kalman_input_1)

initial_state_mean = [measurements[0, 0],
                      0,
                      measurements[0, 1],
                      0]

transition_matrix = [[1, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]]

observation_matrix = [[1, 0, 0, 0],
                      [0, 0, 1, 0]]

kf1 = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean)

kf1 = kf1.em(measurements, n_iter=5)
(smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)



qq_x = np.asarray(smoothed_state_means[:, 0],uint8)
qq_y = np.asarray(smoothed_state_means[:, 2],uint8)


qq_x_2 = np.asarray(smoothed_state_means[:, 0],uint8)
qq_y_2 = np.asarray(smoothed_state_means[:, 2],uint8)


# In[16]:



kalman_locs = []
kalman_output_map = []
i = 0
for j in range(LEN_DATA-1):
    if kalman_times_1[i] == j:
        kalman_grid = np.zeros((SIZE_X,SIZE_X))
        kalman_locs.append([qq_x[i], qq_y[i]])
        kalman_grid[qq_x[i]][qq_y[i]] = 1
        kalman_output_map.append(kalman_grid.copy())
        if i+1 < len(kalman_times_1):
            i+=1
    else:
        kalman_grid = np.zeros((SIZE_X,SIZE_X))
        kalman_output_map.append(kalman_grid.copy())


# Prepare data
spike_times_plot = np.array(M_spike.t/ms)
spike_index_plot = np.array(M_spike.i)

input_spike_times_plot = np.array(M_spike_input.t/ms)
input_spike_index_plot = np.array(M_spike_input.i)

# Generate input and output spike maps, evaluate scores.
input_spike_map_data = []
output_spike_map_data = []
expected_response_map = []
euclidian_dists = []
if SIMULATED_:
    euclidian_dists_2 = []
    
euclidian_dists_plot = []
euclidian_times = []
kalman_euclidian_dists_plot = []
kalman_euclidian_times = []
expected_response_errs = []

kalman_euclidian_err = []

for i in range(0,LEN_DATA-1):
    grid_input = np.zeros((SIZE_X,SIZE_X))
    grid_output = np.zeros((SIZE_X,SIZE_X))
    for ind_ in spike_index_plot[(i*100.0 < spike_times_plot) & (spike_times_plot < i*100.0+100.0)]:
        grid_output[ind_//SIZE_X][ind_%SIZE_X]+=1
    for ind_ in input_spike_index_plot[(i*100.0< input_spike_times_plot) & (input_spike_times_plot < i*100.0+100.0)]:
        grid_input[ind_//SIZE_X][ind_%SIZE_X]+=1
    
    input_spike_map_data.append((grid_input).copy())
    output_spike_map_data.append((grid_output).copy())
    
    
    
    if SIMULATED_:

        gt_path_points = gt_paths[:,i]
        # Find first peak in the output
        

        # Euclidian distance error
        max_err = 100.0
        max_err_2 = 100.0
        max_err_2_ = 100.0
        max_err_2_2 = 100.0
        
        peak_coordinates = peak_local_max(grid_output*100, min_distance=10)
        
        if i in kalman_times_1:
            err_1 = get_euclidian_dist(gt_path_points[0][0],gt_path_points[0][1], kalman_locs[kalman_times_1.index(i)][0], kalman_locs[kalman_times_1.index(i)][1])
            #err_2 = get_euclidian_dist(gt_path_points[1][0],gt_path_points[0][1], kalman_locs[kalman_times_1.index(i)][0], kalman_locs[kalman_times_1.index(i)][1])


            kalman_euclidian_err.append(min(err_1, max_err))

            kalman_euclidian_dists_plot.append(min(err_1, max_err))
            kalman_euclidian_times.append(i)
        
        if len(peak_coordinates) > 0:
            peak_1 = peak_coordinates[0]
            if len(peak_coordinates) > 1:
                peak_2 = peak_coordinates[1]
            for it_ in gt_path_points:
                    dist = get_euclidian_dist(it_[0],it_[1], peak_1[0], peak_1[1])
                    
                    if dist < max_err:
                        max_err = dist
                    if len(peak_coordinates) > 1:
                        dist_2 = get_euclidian_dist(it_[0],it_[1], peak_2[0], peak_2[1])
                        if dist_2 < max_err:
                            max_err_2 = dist
        euclidian_dists.append(max_err)
        euclidian_dists_plot.append(max_err)
        euclidian_times.append(i)
        euclidian_dists_2.append(max_err_2)

        # Expected response error only first one, add second later easily by creating another map
        j = gt_path_points[0][0] * SIZE_Y + gt_path_points[0][1]
        j_2 = gt_path_points[1][0] * SIZE_Y + gt_path_points[1][1]
        grid = np.empty((SIZE_X,SIZE_Y))
        for i in range(0,SIZE_X*SIZE_Y):
            dist = np.sqrt(
                    (i // SIZE_X - j // SIZE_X) ** 2 + (i % SIZE_Y - j % SIZE_Y) ** 2
                )
            dist_2 = np.sqrt(
                    (i // SIZE_X - j_2 // SIZE_X) ** 2 + (i % SIZE_Y - j_2 % SIZE_Y) ** 2
                )
            if dist > CUTOFF_DIST_EVAL:
                transfer_func = 0
            else:
                transfer_func = ( (
                             np.exp(-dist**2 / (SIGMA_DIST_EVAL ** 2))
                        ) / (math.pi * 2 * (SIGMA_DIST_EVAL ** 2)))
                
            if dist_2 > CUTOFF_DIST_EVAL:
                transfer_func_2 = 0
            else:
                transfer_func_2 = ( (
                             np.exp(-dist_2**2 / (SIGMA_DIST_EVAL ** 2))
                        ) / (math.pi * 2 * (SIGMA_DIST_EVAL ** 2)))
            grid[i//SIZE_X][i%SIZE_Y] = transfer_func #+ transfer_func_2
        
        expected_response = np.array(grid / np.max(grid))
        expected_response_map.append(expected_response)
        if np.max(grid_output) != 0:
            actual_response = grid_output / np.max(grid_output)
            error_between_response = np.sum(np.absolute(expected_response - actual_response))
            expected_response_errs.append(error_between_response)
        else:
            expected_response_errs.append(np.sum(np.absolute(expected_response)))
    else:
        max_err = 100.0
        err_1 = 100.0
        #ex_resp_grid = np.empty((SIZE_X,SIZE_Y))
        annot_points = np.argwhere(dense_[i] == 0.5)
        if len(annot_points) > 0:
            mean_point = np.sum(np.array(annot_points), axis=0) // len(np.array(annot_points))
            
            if i in kalman_times_1:
                err_1 = get_euclidian_dist(mean_point[0],mean_point[1] ,kalman_locs[kalman_times_1.index(i)][0], kalman_locs[kalman_times_1.index(i)][1])
                #err_2 = get_euclidian_dist(gt_path_points[1][0],gt_path_points[0][1], kalman_locs[kalman_times_1.index(i)][0], kalman_locs[kalman_times_1.index(i)][1])


                kalman_euclidian_err.append(min(err_1, max_err))

                kalman_euclidian_dists_plot.append(min(err_1, max_err))
                kalman_euclidian_times.append(i)
            
            # Euclidian distance error
            peak_coordinates = peak_local_max(grid_output, min_distance=10)
            if len(peak_coordinates) > 0:
                peak_1 = peak_coordinates[0]
                dist = get_euclidian_dist(mean_point[0],mean_point[1], peak_1[0], peak_1[1])
                if dist < max_err:
                    max_err = dist
            euclidian_dists_plot.append(max_err)
            euclidian_times.append(i)
            grid = np.empty((SIZE_X,SIZE_Y))
            for i in range(0,SIZE_X*SIZE_Y):
                dist__ = np.sqrt(
                        (i // SIZE_X - mean_point[0]) ** 2 + (i % SIZE_Y - mean_point[1]) ** 2
                    )
                if dist__ > CUTOFF_DIST_EVAL:
                    transfer_func = 0
                else:
                    transfer_func = ( (
                                 np.exp(-dist__**2 / (SIGMA_DIST_EVAL ** 2))
                            ) / (math.pi * 2 * (SIGMA_DIST_EVAL ** 2)))

                grid[i//SIZE_X][i%SIZE_Y] = transfer_func
            
            
            expected_resp_grid = np.array(grid / np.max(grid))
        else:
            err_1 = 100.0
            max_err = 100.0
            expected_resp_grid = np.zeros((SIZE_X, SIZE_Y))
        
        if np.max(grid_output) != 0:
            actual_response = grid_output / np.max(grid_output)
            error_between_response = np.sum(np.absolute(expected_resp_grid - actual_response))
            expected_response_errs.append(error_between_response)
        else:
            if len(annot_points) == 0:
                max_err = 100.0
            expected_response_errs.append(np.sum(np.absolute(expected_resp_grid)))
        expected_response_map.append(expected_resp_grid)
        euclidian_dists.append(max_err)
        kalman_euclidian_err.append(err_1)
    




if PLOT_FIGURE:
    #210.0 in spikes_in.segments[0].spiketrains[37].times
    fig2,((ax_1), (ax_2), (ax_3))= plt.subplots(3, 1, figsize=(12,12), dpi=200)

    #ax_1.tick_params(axis='both',labelsize=8)
    ax_1.set_title("Spike Input", fontsize=10)
    input_spike_train_plot = ax_1.plot(M_spike_input.t/ms, M_spike_input.i, '.k', ms=1)
    ax_1.set_xlabel('Time (ms)', fontsize=8)
    ax_1.set_ylabel('Neuron Index', fontsize=8)
    ax_1.tick_params(axis='both',labelsize=4)

    ax_2.set_title("Network Output Spikes",fontsize=10)
    input_spike_train_plot = ax_2.plot(M_spike.t/ms, M_spike.i, '.k', ms=1)
    ax_2.set_xlabel('Time (ms)',fontsize=8)
    ax_2.set_ylabel('Neuron Index', fontsize=8)
    ax_2.tick_params(axis='both',labelsize=4)

    ax_3.set_title("Euclidian distance error",fontsize=10)
    input_spike_train_plot = ax_3.plot(euclidian_times, euclidian_dists_plot, '.r', ms=2.0, label="Network Performance")
    input_spike_train_plot = ax_3.plot(kalman_euclidian_times, kalman_euclidian_dists_plot, '.g', ms=2.0,  label="Kalman Performance")
    ax_3.set_ylim(0,120.0)
    ax_4 = ax_3.twinx()
    input_spike_train_plot = ax_4.plot(expected_response_errs, label="Expected response Difference",linewidth=0.6, color="blue")
    ax_3.set_xlabel('Frame number',fontsize=8)
    ax_3.set_ylabel('Kalman Score', fontsize=7, color="black")
    ax_4.set_ylabel('Expected response Difference', fontsize=7, color="black")
    ax_3.tick_params(axis='both',labelsize=4)
    ax_4.tick_params(axis='both',labelsize=4)
    ax_3.legend(loc="upper left", fontsize=8)
    ax_4.legend(loc="upper right", fontsize=8)

    # ax_3.set_title("Expected Response diff",fontsize=10)
    # input_spike_train_plot = ax_3.plot(expected_response_errs, label="Expected response error)
    # ax_3.set_xlabel('Time (ms)',fontsize=8)
    # ax_3.set_ylabel('Neuron Index', fontsize=8)
    # ax_3.tick_params(axis='both',labelsize=4)
    if SAVE_FIGURE and not SIMULATED_:
        fig2.savefig(f"plots_carrada_spike/thesis_carrada_{seq_name}_eval_tau_{int(TAU)}_sExc_{int(10*SIGMA_EXC)}_sInh_{int(10*SIGMA_INH)}_inExc_{int(10*INTENSITY_EXC)}_inInh_{int(INTENSITY_INH)}.png")


# In[20]:


network_failures = len(np.argwhere(np.array(euclidian_dists_plot) < 10.0)) / len(euclidian_dists_plot) * 100
kalman_failures = len(np.argwhere(np.array(kalman_euclidian_err) < 10.0)) / len(kalman_euclidian_err) * 100

network_tracked_perf = np.mean(np.array(euclidian_dists_plot)[np.argwhere(np.array(euclidian_dists_plot) < 10.0)])

kalman_tracked_perf = np.mean(np.array(kalman_euclidian_err)[np.argwhere(np.array(kalman_euclidian_err) < 10.0)])

if SIMULATED_:
    euclidian_error_mean = np.mean(euclidian_dists)
else:
    euclidian_error_mean = np.mean(euclidian_dists_plot)
    
if SIMULATED_:
    euclidian_error_mean_2 = np.mean(euclidian_dists_2)
expected_response_error_mean = np.mean(expected_response_errs)
kalman_euclidian_err_mean = np.mean(kalman_euclidian_err)

print(f"Euclidian error: {euclidian_error_mean}")
print(f"Kalman Euclidian error: {kalman_euclidian_err_mean}")
print(f"Expected resp error: {expected_response_error_mean}")
if SIMULATED_:
    with open(PERF_FILENAME, 'a') as file:
        file.write(f'{-1},{euclidian_error_mean},{euclidian_error_mean_2},{expected_response_error_mean},{network_failures},{kalman_failures},{kalman_tracked_perf},{network_tracked_perf},{SIGMA_EXC},{SIGMA_INH},{INTENSITY_EXC},{INTENSITY_INH}\n')
else:
    with open(PERF_FILENAME, 'a') as file:
        file.write(f'"{seq_name}",{kalman_euclidian_err_mean},{euclidian_error_mean},{expected_response_error_mean},{network_failures},{kalman_failures},{kalman_tracked_perf},{network_tracked_perf},{SIGMA_EXC},{SIGMA_INH},{INTENSITY_EXC},{INTENSITY_INH}\n')
print(f"KALMAN failure: {kalman_failures}")
print(f"Network failure: {network_failures}")

print(f"KALMAN tracked perf: {kalman_tracked_perf}")
print(f"Network tracked perf: {network_tracked_perf}")


# In[ ]:





# In[ ]:





# In[21]:


if SIMULATED_ and SAVE_VIDEO: 
    grid_input_mock = np.zeros((SIZE_X,SIZE_X))
    grid_output_mock = np.zeros((SIZE_X,SIZE_X))
    for ind_ in spike_index_plot[(0< spike_times_plot) & (spike_times_plot < 100.0)]:
        grid_output_mock[ind_//SIZE_X][ind_%SIZE_X]+=1
    for ind_ in input_spike_index_plot[(0 < input_spike_times_plot) & (input_spike_times_plot < 100.0)]:
        grid_input_mock[ind_//SIZE_X][ind_%SIZE_X]+=1


    fig,(ax1, ax2, ax3, ax4)= plt.subplots(1, 4, figsize=(12,4), dpi=100)

    # Input spikes
    ax1.tick_params(axis='both',labelsize=4)
    ax1.set_title("Input",fontsize=5)
    input_map = ax1.imshow(np.array(grid_input_mock), cmap='hot', interpolation='none')

    ax1.set_xticks([0, (SIZE_X-1)/2, SIZE_X-1])
    ax1.set_yticks([0,
                   SIZE_X*1/5-1,
                   SIZE_X*2/5-1,
                   SIZE_X*3/5-1,
                   SIZE_X*4/5-1,
                   SIZE_X-1])

    ax1.set_yticklabels([50, 40, 30, 20, 10, 0])
    ax1.set_xticklabels([-90, 0, 90])
    ax1.set_ylabel('Distance (m)', fontsize=5)
    ax1.set_xlabel('Angle (Degree)', fontsize=5)

    # Output neuron Spikes
    ax2.tick_params(axis='both',labelsize=4)
    ax2.set_title(f"Output",fontsize=5)

    output_map = ax2.imshow(np.array(grid_output_mock), cmap='hot', interpolation='none')

    ax2.set_xticks([0, (SIZE_X-1)/2, SIZE_X-1])
    ax2.set_yticks([0,
                   SIZE_X*1/5-1,
                   SIZE_X*2/5-1,
                   SIZE_X*3/5-1,
                   SIZE_X*4/5-1,
                   SIZE_X-1])
    ax2.set_yticklabels([50, 40, 30, 20, 10, 0])
    ax2.set_xticklabels([-90, 0, 90])
    ax2.set_ylabel('Distance (m)', fontsize=5)
    ax2.set_xlabel('Angle (Degree)', fontsize=5)
    
     # Expected output map
    ax3.tick_params(axis='both',labelsize=4)
    ax3.set_title(f"Expected Output",fontsize=5)

    expect_output_map = ax3.imshow(np.array(expected_response_map[0]), cmap='hot', interpolation='none')

    ax3.set_xticks([0, (SIZE_X-1)/2, SIZE_X-1])
    ax3.set_yticks([0,
                   SIZE_X*1/5-1,
                   SIZE_X*2/5-1,
                   SIZE_X*3/5-1,
                   SIZE_X*4/5-1,
                   SIZE_X-1])
    ax3.set_yticklabels([50, 40, 30, 20, 10, 0])
    ax3.set_xticklabels([-90, 0, 90])
    ax3.set_ylabel('Distance (m)', fontsize=5)
    ax3.set_xlabel('Angle (Degree)', fontsize=5)
    
     # GT vs Pred Neuron Spikes Expected responses
    ax4.tick_params(axis='both',labelsize=4)
    ax4.set_title("Expected response vs Real response",fontsize=5)

    expected_resp_act_colored = np.zeros((LEN_DATA-1, 64,64,3))
    expected_resp_act_colored[:,:,:,2] = (np.array(output_spike_map_data) / np.max(np.array(output_spike_map_data)))
    expected_resp_act_colored[:,:,:,1] = np.array(expected_response_map) / 2
    expected_resp_act_colored[:,:,:,0] = np.array(kalman_output_map)

    expected_resp_graph_ax = ax4.imshow(np.array(expected_resp_act_colored[0]),interpolation='none')

    ax4.set_xticks([0, (SIZE_X-1)/2, SIZE_X-1])
    ax4.set_yticks([0,
                   SIZE_X*1/5-1,
                   SIZE_X*2/5-1,
                   SIZE_X*3/5-1,
                   SIZE_X*4/5-1,
                   SIZE_X-1])
    ax4.set_yticklabels([50, 40, 30, 20, 10, 0])
    ax4.set_xticklabels([-90, 0, 90])
    ax4.set_ylabel('Distance (m)', fontsize=5)
    ax4.set_xlabel('Angle (Degree)', fontsize=5)

    
    def init():
        input_map.set_data(np.array(input_spike_map_data[0]))
        output_map.set_data(np.array(output_spike_map_data[0]))
        expect_output_map.set_data(np.array(expected_response_map[0]))
        expected_resp_graph_ax.set_data(np.array(expected_resp_act_colored[0]))

        return [input_map, output_map, expect_output_map, expected_resp_graph_ax]


    # animation function.  This is called sequentially
    def animate(i):
        input_map.set_data(np.array(input_spike_map_data[i]))
        output_map.set_data(np.array(output_spike_map_data[i]))
        expect_output_map.set_data(np.array(expected_response_map[i]))
        expected_resp_graph_ax.set_data(np.array(expected_resp_act_colored[i]))

        return [input_map, output_map, expect_output_map, expected_resp_graph_ax]


    FFwriter = animation.FFMpegWriter(fps=10, extra_args=["-vcodec", "libx264"])
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(ds)-1, interval=1, blit=True
    )

    anim.save(f"animations/sim_spike_eval_tau_{int(TAU)}_sExc_{int(10*SIGMA_EXC)}_sInh_{int(10*SIGMA_INH)}_inExc_{int(10*INTENSITY_EXC)}_inInh_{int(INTENSITY_INH)}_.mp4", FFwriter)
    
elif SAVE_VIDEO:
    npdense_ = np.array(dense_)
    npdense_[npdense_ < 0.1] = -0.01
    npdense_[npdense_ >= 0.1] = 0.1

    npdense_vis = np.array(dense_)
    npdense_vis[npdense_vis < 0.1] = 0.0
    npdense_vis[npdense_vis >= 0.1] = 1.0

    grid_input_mock = np.zeros((SIZE_X,SIZE_X))
    grid_output_mock = np.zeros((SIZE_X,SIZE_X))
    for ind_ in spike_index_plot[(4000< spike_times_plot) & (spike_times_plot < 5100.0)]:
        grid_output_mock[ind_//SIZE_X][ind_%SIZE_X]+=1
    for ind_ in input_spike_index_plot[(4000 < input_spike_times_plot) & (input_spike_times_plot < 5100.0)]:
        grid_input_mock[ind_//SIZE_X][ind_%SIZE_X]+=1


    fig,((ax1, ax2, ax3), (ax4, ax5, ax6))= plt.subplots(2, 3, figsize=(9,6), dpi=200)

    # Input spikes
    ax1.tick_params(axis='both',labelsize=4)
    ax1.set_title("Input",fontsize=5)
    input_map = ax1.imshow(np.array(grid_input_mock), cmap='hot', interpolation='none')

    ax1.set_xticks([0, (SIZE_X-1)/2, SIZE_X-1])
    ax1.set_yticks([0,
                   SIZE_X*1/5-1,
                   SIZE_X*2/5-1,
                   SIZE_X*3/5-1,
                   SIZE_X*4/5-1,
                   SIZE_X-1])

    ax1.set_yticklabels([50, 40, 30, 20, 10, 0])
    ax1.set_xticklabels([-90, 0, 90])
    ax1.set_ylabel('Distance (m)', fontsize=5)
    ax1.set_xlabel('Angle (Degree)', fontsize=5)

    # Output neuron Spikes
    ax2.tick_params(axis='both',labelsize=4)
    ax2.set_title(f"Output",fontsize=5)

    output_map = ax2.imshow(np.array(grid_output_mock), cmap='hot', interpolation='none')

    ax2.set_xticks([0, (SIZE_X-1)/2, SIZE_X-1])
    ax2.set_yticks([0,
                   SIZE_X*1/5-1,
                   SIZE_X*2/5-1,
                   SIZE_X*3/5-1,
                   SIZE_X*4/5-1,
                   SIZE_X-1])
    ax2.set_yticklabels([50, 40, 30, 20, 10, 0])
    ax2.set_xticklabels([-90, 0, 90])
    ax2.set_ylabel('Distance (m)', fontsize=5)
    ax2.set_xlabel('Angle (Degree)', fontsize=5)

    # Annotated Neuron Activities
    ax3.tick_params(axis='both',labelsize=4)
    ax3.set_title("Neuron Spikes with Annotation",fontsize=5)

    annotated_neuron_act = np.zeros((LEN_DATA-1, 64,64,3))
    annotated_neuron_act[:,:,:,0] = (np.array(output_spike_map_data) / np.max(np.array(output_spike_map_data)))
    annotated_neuron_act[:,:,:,1] = np.array(npdense_vis[:LEN_DATA-1])
    
    

    spike_map_w_annot = ax3.imshow(np.array(annotated_neuron_act[0]), cmap='hot', interpolation='none')

    ax3.set_xticks([0, (SIZE_X-1)/2, SIZE_X-1])
    ax3.set_yticks([0,
                   SIZE_X*1/5-1,
                   SIZE_X*2/5-1,
                   SIZE_X*3/5-1,
                   SIZE_X*4/5-1,
                   SIZE_X-1])
    ax3.set_yticklabels([50, 40, 30, 20, 10, 0])
    ax3.set_xticklabels([-90, 0, 90])
    ax3.set_ylabel('Distance (m)', fontsize=5)
    ax3.set_xlabel('Angle (Degree)', fontsize=5)

    # GT vs Pred Neuron Spikes Expected responses
    ax4.tick_params(axis='both',labelsize=4)
    ax4.set_title("Expected response vs Real response",fontsize=5)

    expected_resp_act_colored = np.zeros((LEN_DATA-1, 64,64,3))
    expected_resp_act_colored[:,:,:,2] = (np.array(output_spike_map_data) / np.max(np.array(output_spike_map_data)))
    expected_resp_act_colored[:,:,:,1] = np.array(expected_response_map) / 2
    expected_resp_act_colored[:,:,:,0] = np.array(kalman_output_map)

    expected_resp_graph_ax = ax4.imshow(np.array(expected_resp_act_colored[0]),interpolation='none')

    ax4.set_xticks([0, (SIZE_X-1)/2, SIZE_X-1])
    ax4.set_yticks([0,
                   SIZE_X*1/5-1,
                   SIZE_X*2/5-1,
                   SIZE_X*3/5-1,
                   SIZE_X*4/5-1,
                   SIZE_X-1])
    ax4.set_yticklabels([50, 40, 30, 20, 10, 0])
    ax4.set_xticklabels([-90, 0, 90])
    ax4.set_ylabel('Distance (m)', fontsize=5)
    ax4.set_xlabel('Angle (Degree)', fontsize=5)

    # Raw RGB Camera data
    ax5.tick_params(axis='both',labelsize=6)
    metrics = ['KalmanEuc', 'Euclidian']#, 'ExpResp']
    scores_of_metrics = [kalman_euclidian_err[0], euclidian_dists[0]]#,expected_response_errs[0]]
    ax5.set_ylim(0, 10.0)#  max(max(np.max(expected_response_errs), np.max(euclidian_dists))#, np.max(kalman_euclidian_err)))
    bar_graph = ax5.bar(metrics,scores_of_metrics)
    ax5.set_title("Scores",fontsize=5)
    if LOAD_CAM_DATA:
        # Raw RGB Camera data
        ax6.tick_params(axis='both',labelsize=4)
        ax6.set_title("Raw Camera RGB Data",fontsize=5)
        rgb_cam_data_2_plot = ax6.imshow(np.array(raw_camera_data[1]), interpolation='none')

        ax6.tick_params(axis=u'both', which=u'both',length=0)
    
    def init():
        input_map.set_data(np.array(input_spike_map_data[0]))
        output_map.set_data(np.array(output_spike_map_data[0]))
        spike_map_w_annot.set_data(np.array(annotated_neuron_act[0]))
        expected_resp_graph_ax.set_data(np.array(expected_resp_act_colored[0]))
        
        for rect, h in zip(bar_graph, [kalman_euclidian_err[0], euclidian_dists[0]]):#,expected_response_errs[0]]):
            rect.set_height(h)
        if LOAD_CAM_DATA:
            rgb_cam_data_2_plot.set_data(np.array(raw_camera_data[1]))
            return [input_map, output_map, spike_map_w_annot, expected_resp_graph_ax]#, rgb_cam_data_2_plot]
        else:
            return [input_map, output_map, spike_map_w_annot]#, expected_resp_graph_ax]


    # animation function.  This is called sequentially
    def animate(i):
        input_map.set_data(np.array(input_spike_map_data[i]))
        output_map.set_data(np.array(output_spike_map_data[i]))
        spike_map_w_annot.set_data(np.array(annotated_neuron_act[i]))
        expected_resp_graph_ax.set_data(np.array(expected_resp_act_colored[i]))
        for rect, h in zip(bar_graph, [kalman_euclidian_err[i], euclidian_dists[i]]):#,expected_response_errs[i]]):
            rect.set_height(h)
        if LOAD_CAM_DATA:
            rgb_cam_data_2_plot.set_data(np.array(raw_camera_data[i]))
            if i in [87,125,155]: fig.savefig(f'stats_carrada_spike/carrada_spike_{seq_name}_tau_{int(TAU)}_sExc_{int(10*SIGMA_EXC)}_sInh_{int(10*SIGMA_INH)}_inExc_{int(10*INTENSITY_EXC)}_inInh_{int(INTENSITY_INH)}_anim_fig_{i}.png')
            return [input_map, output_map, spike_map_w_annot, expected_resp_graph_ax]#, rgb_cam_data_2_plot]
        else:
            return [input_map, output_map, spike_map_w_annot]#, expected_resp_graph_ax]



    FFwriter = animation.FFMpegWriter(fps=10, extra_args=["-vcodec", "libx264"])
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=LEN_DATA-2, interval=1, blit=True
    )

    anim.save(f"stats_carrada_spike/carrada_spike_{seq_name}_tau_{int(TAU)}_sExc_{int(10*SIGMA_EXC)}_sInh_{int(10*SIGMA_INH)}_inExc_{int(10*INTENSITY_EXC)}_inInh_{int(INTENSITY_INH)}_.mp4", FFwriter)






