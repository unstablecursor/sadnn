#!/usr/bin/env python
# coding: utf-8

# In[1]:


SIMULATED_ = False
PLOT_FIGURE = True
SAVE_FIGURE = True
SAVE_VIDEO = True
LOAD_CAM_DATA = True
PERF_FILENAME = 'stats_carrada_spike/stats.csv'


# In[2]:


## Imports
import os
import cv2
import sys
import json
import time
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.feature import peak_local_max

if SIMULATED_:
    from components.SimulatedRadar import SimulatedRadar
else:
    from components.RadarLoader import RadarLoader
    seq_name = sys.argv[1]


from brian2 import *
set_device("cpp_standalone")

# In[3]:


# Network params
INTENSITY_EXC = 0.8
INTENSITY_INH = -10.5
SIGMA_EXC = 1.0 # 5?
SIGMA_INH = 10.5

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
CUTOFF_DIST_EVAL = 10.0
SIGMA_DIST_EVAL = 5.0


# In[4]:


#device.reinit()
#device.activate()
#device.delete(force=True)
#set_device('cpp_standalone')#, build_on_run=False)    


# In[5]:


LEN_DATA = 0
if SIMULATED_:
    sim_data_loader = SimulatedRadar(noise_factor=NOISE_FACTOR)
    spiking_data, spiking_indices = sim_data_loader.get_random_datastream_spiking_brian2(size_x=SIZE_X)
    ds = sim_data_loader.get_random_datastream()
    gt_paths = sim_data_loader.get_paths()
    LEN_DATA = len(ds)
    print(f"Simulation data is loaded. Total number of frames: {LEN_DATA}")
else:
    data_loader = RadarLoader(seq_name)
    if LOAD_CAM_DATA:
        raw_camera_data = data_loader.get_color_image_datastream(resize=(SIZE_X,SIZE_Y))
    raw_data = np.load(f"carrada_processed/{seq_name}_raw_data.npy")
    spiking_data = np.load(f"carrada_processed/{seq_name}_spiking_data.npy")
    spiking_indices = np.load(f"carrada_processed/{seq_name}_spiking_indices.npy")
    dense_ = np.load(f"carrada_processed/{seq_name}_dense_.npy")
    LEN_DATA = len(raw_data-1)
    print(f"CARRADA data is loaded. Total number of frames: {LEN_DATA}")


# In[ ]:





# In[6]:


# np.save("raw_data.npy", np.asanyarray(raw_data))
# #np.save("raw_camera_data.npy", np.asanyarray(raw_camera_data))
# np.save("spiking_data.npy", np.asanyarray(spiking_data))
# np.save("spiking_indices.npy", np.asanyarray(spiking_indices))
# np.save("dense_.npy", np.asanyarray(dense_))
#raw_data = np.load("raw_data.npy")
#raw_camera_data = np.load("raw_camera_data.npy")
#spiking_data = np.load("spiking_data.npy")
#spiking_indices = np.load("spiking_indices.npy")
#dense_ = np.load("dense_.npy")


# In[7]:


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


# In[8]:


run((LEN_DATA-1)*TIME_BETWEEN_FRAMES*ms, report="stdout", report_period=30000*ms)


# In[9]:



kalman_spiking_input = []
for qqq in range(0,LEN_DATA-1):
    i___ = 0
    x___ = 0
    y___ = 0
    for ind_ in spiking_indices[(qqq*100.0< spiking_data) & (spiking_data < (qqq+1)*100.0)]:
        x___ += ind_//SIZE_X
        y___ += ind_%SIZE_X
        i___ +=1
    if i___ > 0:
        kalman_spiking_input.append((x___/i___, y___/i___))
    elif len(kalman_spiking_input) == 0:
        kalman_spiking_input.append((31,31))
    else:
        kalman_spiking_input.append(kalman_spiking_input[-1])
    #spiking_locs.append((spike_ind // 64 , spike_ind % 64))

measurements = np.asarray(kalman_spiking_input)
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

kalman_locs = []
kalman_output_map = []
for i in range(len(qq_x)):
    kalman_grid = np.zeros((SIZE_X,SIZE_X))
    kalman_locs.append([qq_x[i], qq_y[i]])
    kalman_grid[qq_x[i]][qq_y[i]] = 1
    kalman_output_map.append(kalman_grid.copy())


# In[10]:


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
        peak_coordinates = peak_local_max(grid_output, min_distance=10)
        err_1 = np.sqrt((gt_path_points[0][0] - kalman_locs[i][0])**2 + (gt_path_points[0][1] - kalman_locs[i][1])**2)
        err_2 = np.sqrt((gt_path_points[1][1] - kalman_locs[i][0])**2 + (gt_path_points[1][1] - kalman_locs[i][1])**2)
        kalman_euclidian_err.append(min(err_1, err_2))
        
        if len(peak_coordinates) > 0:
            peak_1 = peak_coordinates[0]
            if len(peak_coordinates) > 1:
                peak_2 = peak_coordinates[1]
            for it_ in gt_path_points:
                    dist = np.sqrt((it_[0] - peak_1[0]) ** 2 + (it_[1] - peak_1[1]) ** 2)
                    
                    if dist < max_err:
                        max_err = dist
                    if len(peak_coordinates) > 1:
                        dist_2 = np.sqrt((it_[0] - peak_2[0]) ** 2 + (it_[1] - peak_2[1]) ** 2)
                        if dist_2 < max_err:
                            max_err_2 = dist
        euclidian_dists.append(max_err)
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
            grid[i//SIZE_X][i%SIZE_Y] = transfer_func + transfer_func_2
        
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
            err_1 = np.sqrt((mean_point[0] - kalman_locs[i][0])**2 + (mean_point[1] - kalman_locs[i][1])**2)
            
            # Euclidian distance error
            peak_coordinates = peak_local_max(grid_output, min_distance=10)
            if len(peak_coordinates) > 0:
                peak_1 = peak_coordinates[0]
                dist = np.sqrt((mean_point[0] - peak_1[0]) ** 2 + (mean_point[1] - peak_1[1]) ** 2)
                if dist < max_err:
                    max_err = dist
            
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
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


if PLOT_FIGURE:
    #210.0 in spikes_in.segments[0].spiketrains[37].times
    fig2,((ax_1), (ax_2), (ax_3))= plt.subplots(3, 1, figsize=(12,8), dpi=200)

    #ax_1.tick_params(axis='both',labelsize=8)
    ax_1.set_title("Input", fontsize=10)
    input_spike_train_plot = ax_1.plot(M_spike_input.t/ms, M_spike_input.i, '.k', ms=1)
    ax_1.set_xlabel('Time (ms)', fontsize=8)
    ax_1.set_ylabel('Neuron Index', fontsize=8)
    ax_1.tick_params(axis='both',labelsize=4)

    ax_2.set_title("Output",fontsize=10)
    input_spike_train_plot = ax_2.plot(M_spike.t/ms, M_spike.i, '.k', ms=1)
    ax_2.set_xlabel('Time (ms)',fontsize=8)
    ax_2.set_ylabel('Neuron Index', fontsize=8)
    ax_2.tick_params(axis='both',labelsize=4)

    ax_3.set_title("Euclidian distance error",fontsize=10)
    input_spike_train_plot = ax_3.plot(euclidian_dists, label="Evaluation Performance", color="red")
    input_spike_train_plot = ax_3.plot(kalman_euclidian_err, label="Kalman Performance", color="green")
    ax_4 = ax_3.twinx()
    input_spike_train_plot = ax_4.plot(expected_response_errs, label="Expected response error", color="blue")
    ax_3.set_xlabel('Frame number',fontsize=8)
    ax_3.set_ylabel('Euclidian distance score, Kalman(green)', fontsize=8, color="yellow")
    ax_4.set_ylabel('Expected response score', fontsize=8, color="blue")
    ax_3.tick_params(axis='both',labelsize=4)
    ax_4.tick_params(axis='both',labelsize=4)

    # ax_3.set_title("Expected Response diff",fontsize=10)
    # input_spike_train_plot = ax_3.plot(expected_response_errs, label="Expected response error)
    # ax_3.set_xlabel('Time (ms)',fontsize=8)
    # ax_3.set_ylabel('Neuron Index', fontsize=8)
    # ax_3.tick_params(axis='both',labelsize=4)
    if SAVE_FIGURE and not SIMULATED_:
        fig2.savefig(f"plots_carrada_spike/carrada_{seq_name}_eval_tau_{int(TAU)}_sExc_{int(10*SIGMA_EXC)}_sInh_{int(10*SIGMA_INH)}_inExc_{int(10*INTENSITY_EXC)}_inInh_{int(INTENSITY_INH)}.png")


# In[12]:



euclidian_error_mean = np.mean(euclidian_dists)
if SIMULATED_:
    euclidian_error_mean_2 = np.mean(euclidian_dists_2)
expected_response_error_mean = np.mean(expected_response_errs)
kalman_euclidian_err_mean = np.mean(kalman_euclidian_err)
print(f"Euclidian error: {euclidian_error_mean}")
if SIMULATED_:
    print(f"Euclidian error2: {euclidian_error_mean_2}")
print(f"Kalman Euclidian error: {kalman_euclidian_err_mean}")
print(f"Expected resp error: {expected_response_error_mean}")
if SIMULATED_:
    with open(PERF_FILENAME, 'a') as file:
        file.write(f'{euclidian_error_mean},{euclidian_error_mean_2},{expected_response_error_mean},{SIGMA_EXC},{SIGMA_INH},{INTENSITY_EXC},{INTENSITY_INH}\n')
with open(PERF_FILENAME, 'a') as file:
    file.write(f'"{seq_name}",{kalman_euclidian_err_mean},{euclidian_error_mean},{expected_response_error_mean},{SIGMA_EXC},{SIGMA_INH},{INTENSITY_EXC},{INTENSITY_INH}\n')


# In[ ]:





# In[13]:


if SIMULATED_ and SAVE_VIDEO: 
    grid_input_mock = np.zeros((SIZE_X,SIZE_X))
    grid_output_mock = np.zeros((SIZE_X,SIZE_X))
    for ind_ in spike_index_plot[(0< spike_times_plot) & (spike_times_plot < 100.0)]:
        grid_output_mock[ind_//SIZE_X][ind_%SIZE_X]+=1
    for ind_ in input_spike_index_plot[(0 < input_spike_times_plot) & (input_spike_times_plot < 100.0)]:
        grid_input_mock[ind_//SIZE_X][ind_%SIZE_X]+=1


    fig,(ax1, ax2, ax3)= plt.subplots(1, 3, figsize=(8,5), dpi=100)

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
    
    def init():
        input_map.set_data(np.array(input_spike_map_data[0]))
        output_map.set_data(np.array(output_spike_map_data[0]))
        expect_output_map.set_data(np.array(expected_response_map[0]))
        return [input_map, output_map, expect_output_map]


    # animation function.  This is called sequentially
    def animate(i):
        input_map.set_data(np.array(input_spike_map_data[i]))
        output_map.set_data(np.array(output_spike_map_data[i]))
        expect_output_map.set_data(np.array(expected_response_map[i]))
        return [input_map, output_map, expect_output_map]


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

    annotated_neuron_act = np.zeros((len(npdense_vis), 64,64,3))
    annotated_neuron_act[:,:,:,0] = (np.array(output_spike_map_data) / np.max(np.array(output_spike_map_data)))
    annotated_neuron_act[:,:,:,1] = np.array(npdense_vis)
    
    

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

    expected_resp_act_colored = np.zeros((len(npdense_vis), 64,64,3))
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
    metrics = ['KalmanEuc', 'Euclidian', 'ExpResp']
    scores_of_metrics = [kalman_euclidian_err[0], euclidian_dists[0],expected_response_errs[0]]
    ax5.set_ylim(0, max(max(np.max(expected_response_errs), np.max(euclidian_dists)), np.max(kalman_euclidian_err)))
    bar_graph = ax5.bar(metrics,scores_of_metrics)
    
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
        
        for rect, h in zip(bar_graph, [kalman_euclidian_err[0], euclidian_dists[0],expected_response_errs[0]]):
            rect.set_height(h)
        if LOAD_CAM_DATA:
            rgb_cam_data_2_plot.set_data(np.array(raw_camera_data[1]))
            return [input_map, output_map, spike_map_w_annot, expected_resp_graph_ax, rgb_cam_data_2_plot]
        else:
            return [input_map, output_map, spike_map_w_annot, expected_resp_graph_ax]


    # animation function.  This is called sequentially
    def animate(i):
        input_map.set_data(np.array(input_spike_map_data[i]))
        output_map.set_data(np.array(output_spike_map_data[i]))
        spike_map_w_annot.set_data(np.array(annotated_neuron_act[i]))
        expected_resp_graph_ax.set_data(np.array(expected_resp_act_colored[i]))
        for rect, h in zip(bar_graph, [kalman_euclidian_err[i], euclidian_dists[i],expected_response_errs[i]]):
            rect.set_height(h)
        if LOAD_CAM_DATA:
            rgb_cam_data_2_plot.set_data(np.array(raw_camera_data[i]))
            return [input_map, output_map, spike_map_w_annot, expected_resp_graph_ax, rgb_cam_data_2_plot]
        else:
            return [input_map, output_map, spike_map_w_annot, expected_resp_graph_ax]



    FFwriter = animation.FFMpegWriter(fps=10, extra_args=["-vcodec", "libx264"])
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=LEN_DATA-2, interval=1, blit=True
    )

    anim.save(f"stats_carrada_spike/carrada_spike_{seq_name}_tau_{int(TAU)}_sExc_{int(10*SIGMA_EXC)}_sInh_{int(10*SIGMA_INH)}_inExc_{int(10*INTENSITY_EXC)}_inInh_{int(INTENSITY_INH)}_.mp4", FFwriter)






# In[14]:


len(expected_response_errs)


# In[ ]:





# In[15]:


# #if PLOT_FIGURE and SAVE_VIDEO: 
# %matplotlib inline
# import matplotlib
# from IPython.display import HTML
# matplotlib.rcParams['animation.embed_limit'] = 2**128
# HTML(anim.to_jshtml())


# In[16]:



device.delete(force=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




