#!/usr/bin/python3
# coding: utf-8
import os
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from brian2 import *

from components.RadarLoader import RadarLoader

# from components.SimulatedRadar import SimulatedRadar
import sys

matplotlib.use("Agg")

set_device("cpp_standalone")

seq_name = "2020-02-28-13-13-43"

SIZE_X = 64
SIZE_Y = 64
INPUT_WEIGHT = 0.2
TIME_BETWEEN_FRAMES = 100.0
DATA_FREQ_MULTIPLIER = 2.0
GRID_DISTANCE = 100.0
TAU = 250  # ms
SIGMA = 0.5
CONN_CUTOFF = 7
INTENSITY_EXC = 0.3
INTENSITY_INH = -4.0


data_loader = RadarLoader(seq_name)
# sim_data_loader = SimulatedRadar()

raw_data, size_bf = data_loader.get_range_angle_stream_data(
    clip_and_normalize=True, resize=(SIZE_X, SIZE_X)
)
raw_camera_data = data_loader.get_color_image_datastream(resize=(SIZE_X, SIZE_Y))
(
    spiking_data,
    spiking_indices,
) = data_loader.get_spiking_ra_stream_differentiated_normalized_brian2(
    size_x=SIZE_X, time_bw_frames=TIME_BETWEEN_FRAMES, data_factor=DATA_FREQ_MULTIPLIER
)

# sim_spiking_data, sim_spiking_indices = sim_data_loader.get_random_datastream_spiking_brian2(size_x=SIZE_X)
# ds = sim_data_loader.get_random_datastream()

dense_, sparse_, box_, sp_mp_, sp_mp_vis_ = data_loader.visualize_annotations(
    differentiated=True, size_bf=size_bf, size_x_=(SIZE_X, SIZE_Y)
)


# print(f"Total nr of frames: {(len(raw_camera_data))}. Starting...")


def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), "ok", ms=10)
    plot(ones(Nt), arange(Nt), "ok", ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], "-k")
    xticks([0, 1], ["Source", "Target"])
    ylabel("Neuron index")
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, "ok")
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel("Source neuron index")
    ylabel("Target neuron index")


N = SIZE_X * SIZE_Y
rows = SIZE_X
cols = SIZE_Y
# initialize the grid positions
grid_dist = GRID_DISTANCE * umeter
conn_distance = CONN_CUTOFF * grid_dist


SIGMA_EXC = float(sys.argv[1])
SIGMA_INH = float(sys.argv[2])
INTENSITY_EXC = float(sys.argv[3])
INTENSITY_INH = float(sys.argv[4])

sigma_exc = SIGMA_EXC * grid_dist
sigma_inh = SIGMA_INH * grid_dist
intensity_exc = INTENSITY_EXC
intensity_inh = INTENSITY_INH

print(
    f"Start wit \nh Sigma_Exc={SIGMA_EXC} , Sigma_inh={SIGMA_INH}, I_Exc={intensity_exc}, I_Inh={intensity_inh}"
)


# print("Build net...")
start_scope()
tau = TAU * ms

eqs = """
dv/dt = -v/tau : 1
x : meter
y : meter
"""

N = SIZE_X * SIZE_Y
rows = SIZE_X
cols = SIZE_Y
# initialize the grid positions
grid_dist = GRID_DISTANCE * umeter
sigma_exc = SIGMA_EXC * grid_dist
sigma_inh = SIGMA_INH * grid_dist
intensity_exc = INTENSITY_EXC
intensity_inh = INTENSITY_INH
conn_distance = CONN_CUTOFF * grid_dist

G = NeuronGroup(N, eqs, threshold="v>1.0", reset="v = 0.0", method="exact")
G.x = "(i // rows) * grid_dist - rows/2.0 * grid_dist"
G.y = "(i % rows) * grid_dist - cols/2.0 * grid_dist"

# Synapses
S = Synapses(G, G, "w : 1", on_pre="v_post += w")
S.connect(
    condition="(i!=j) and (sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) < conn_distance)"
)
# Weight varies with distance
S.w = "intensity_exc*exprel(-(sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2))**2/(2*sigma_exc**2))"

S_inhib = Synapses(
    G, G, "w : 1", on_pre="v_post = clip(w + v_post, 0, inf)"
)  # TODO: inf oder eins??
S_inhib.connect(
    condition="(i!=j)"
)  # and (sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) > conn_distance)')
# Weight varies with distance
S_inhib.w = "intensity_inh*exprel(-(sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2))**2/(2*sigma_inh**2))"


G_input = SpikeGeneratorGroup(N, spiking_indices, spiking_data * ms)
weight = INPUT_WEIGHT  # 0.1 works
S_input = Synapses(G_input, G, on_pre="v += weight")
S_input.connect(i="j")

M_spike = SpikeMonitor(G)
M_spike_input = SpikeMonitor(G_input)

M_state = StateMonitor(G, "v", record=True)

# print("Connections defined")


print("Running sim...")
run(
    (len(raw_camera_data)) * TIME_BETWEEN_FRAMES * ms,
    report="stdout",
    report_period=100000 * ms,
)
print("Finished sim...")

npdense_ = np.array(dense_)
npdense_[npdense_ < 0.1] = -0.01
npdense_[npdense_ >= 0.1] = 1

npdense_vis = np.array(dense_)
npdense_vis[npdense_vis < 0.1] = 0.0
npdense_vis[npdense_vis >= 0.1] = 1.0


spike_times_plot = np.array(M_spike.t / ms)
spike_index_plot = np.array(M_spike.i)

input_spike_times_plot = np.array(M_spike_input.t / ms)
input_spike_index_plot = np.array(M_spike_input.i)


input_spike_map_data = []
output_spike_map_data = []
eval_spike_map_data_dense = []

eval_spikes_perf = []


for i_ in range(0, len(raw_camera_data)):
    grid_input = np.zeros((SIZE_X, SIZE_X))
    grid_output = np.zeros((SIZE_X, SIZE_X))
    for ind_ in spike_index_plot[
        (i_ * 100.0 < spike_times_plot) & (spike_times_plot < i_ * 100.0 + 100.0)
    ]:
        grid_output[ind_ // SIZE_X][ind_ % SIZE_X] += 1
    for ind_ in input_spike_index_plot[
        (i_ * 100.0 < input_spike_times_plot)
        & (input_spike_times_plot < i_ * 100.0 + 100.0)
    ]:
        grid_input[ind_ // SIZE_X][ind_ % SIZE_X] += 1

    temp__ = npdense_[i_] * grid_output
    if np.sum(temp__) > 0:
        eval_spike_map_data_dense.append(temp__ / np.max(temp__))
    else:
        eval_spike_map_data_dense.append(temp__ + 0.00001)
    if np.sum(dense_[i_]) > 0:
        eval_spikes_perf.append(temp__)
    input_spike_map_data.append((grid_input).copy())
    output_spike_map_data.append((grid_output).copy())


t_sum_eval = np.sum(np.sum(eval_spikes_perf, axis=1), axis=1)
t_sum_eval_all = np.sum(np.sum(eval_spike_map_data_dense, axis=1), axis=1)
eval_score_final = np.sum(t_sum_eval) / len(t_sum_eval)
# In[43]:
# print(f"Start with eval_{int(eval_score_final)}_tau_{int(TAU)}_sigma_{int(10*SIGMA)}_inExc_{int(10*INTENSITY_EXC)}_inInh_{int(INTENSITY_INH)}_cutoff_{int(CONN_CUTOFF)}.jpg")
# print(f"work  with eval_{int(eval_score_final)}_tau_{int(TAU)}_sigma_{int(10*SIGMA)}_inExc_{int(10*INTENSITY_EXC)}_inInh_{int(INTENSITY_INH)}_cutoff_{int(CONN_CUTOFF)}.jpg")
fig2, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(12, 3), dpi=200)

# ax_1.tick_params(axis='both',labelsize=8)
ax_1.set_title("Input", fontsize=18)
input_spike_train_plot = ax_1.plot(M_spike_input.t / ms, M_spike_input.i, ".k", ms=1)
ax_1.set_xlabel("Time (ms)", fontsize=10)
ax_1.set_ylabel("Neuron Index", fontsize=10)
ax_1.tick_params(axis="both", labelsize=8)

ax_2.set_title("Outpu", fontsize=18)
input_spike_train_plot = ax_2.plot(M_spike.t / ms, M_spike.i, ".k", ms=1)
ax_2.set_xlabel("Time (ms)", fontsize=10)
ax_2.set_ylabel("Neuron Index", fontsize=10)
ax_2.tick_params(axis="both", labelsize=8)

ax_3.set_title("EvalMetric", fontsize=18)
input_spike_train_plot = ax_3.plot(t_sum_eval_all)
ax_3.set_xlabel("Time (ms)", fontsize=10)
ax_3.set_ylabel("Neuron Index", fontsize=10)
ax_3.tick_params(axis="both", labelsize=8)

fig2.savefig(
    f"plots_brian/eval4_{int(eval_score_final)}_sE_{int(10*SIGMA_EXC)}_sI_{int(10*SIGMA_INH)}_iE_{int(10*INTENSITY_EXC)}_iI_{int(10*INTENSITY_INH)}.jpg"
)
plt.close(fig2)


device.delete(force=True)
