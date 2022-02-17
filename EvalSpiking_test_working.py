# coding: utf-8
import os
import cv2
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from components.RadarLoader import RadarLoader

# from components.SimulatedRadar import SimulatedRadar

from brian2 import *
import sys

seq_name = "2020-02-28-13-13-43"

matplotlib.use("Agg")

set_device("cpp_standalone")

SIZE_X = 64
SIZE_Y = 64
INPUT_WEIGHT = 1.0
TIME_BETWEEN_FRAMES = 100.0
DATA_FREQ_MULTIPLIER = 2.0
TAU = 200  # ms
GRID_DISTANCE = 100.0
SIGMA_EXC = 5.5  # 5?
SIGMA_INH = 15.0
CONN_CUTOFF = 120
INTENSITY_EXC = 40.5
INTENSITY_INH = -36.0
LOWER_VOLT_THRESH = -1.0

SIGMA_EXC = float(sys.argv[1])
SIGMA_INH = float(sys.argv[2])
INTENSITY_EXC = float(sys.argv[3])
INTENSITY_INH = float(sys.argv[4])
LOWER_VOLT_THRESH = float(sys.argv[5])


raw_data = np.load("raw_data.npy")
spiking_data = np.load("spiking_data.npy")
spiking_indices = np.load("spiking_indices.npy")
dense_ = np.load("dense_.npy")


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


start_scope()
tau = TAU * ms

# initialize the grid positions and variables
N = SIZE_X * SIZE_Y
rows = SIZE_X
cols = SIZE_Y
grid_dist = GRID_DISTANCE * umeter
sigma_exc = SIGMA_EXC * grid_dist
sigma_inh = SIGMA_INH * grid_dist
intensity_exc = INTENSITY_EXC / (2 * math.pi * SIGMA_EXC * 1e2)
intensity_inh = INTENSITY_INH / (2 * math.pi * SIGMA_INH * 1e2)
conn_distance = CONN_CUTOFF * grid_dist
lower_volt_thresh = LOWER_VOLT_THRESH

eqs = """
dv/dt = -v/tau : 1
x : meter
y : meter
"""

G = NeuronGroup(N, eqs, threshold="v>1.0", reset="v = 0.0", method="exact")
G.x = "(i // rows) * grid_dist - rows/2.0 * grid_dist"
G.y = "(i % rows) * grid_dist - cols/2.0 * grid_dist"

# Synapses
S = Synapses(G, G, "w : 1", on_pre="v_post += w")
S.connect(
    condition="(i!=j) and (sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) < conn_distance)"
)
# Weight varies with distance
S.w = "(intensity_exc*exprel(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigma_exc**2))))"

S_inhib = Synapses(
    G, G, "w : 1", on_pre="v_post = clip((w + v_post), lower_volt_thresh, inf)"
)  # TODO: inf oder eins??
S_inhib.connect(
    condition="(i!=j) and (sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) < conn_distance)"
)
# Weight varies with distance
S_inhib.w = "(intensity_inh*exprel(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigma_inh**2))))"

# Connect input
G_input = SpikeGeneratorGroup(N, spiking_indices, spiking_data * ms)
weight = INPUT_WEIGHT  # 0.1 works
S_input = Synapses(G_input, G, on_pre="v += weight")
S_input.connect(i="j")

# Add spike monitor
M_spike = SpikeMonitor(G)
M_spike_input = SpikeMonitor(G_input)

M_state = StateMonitor(G, "v", record=True)

# Run simulation

run(216 * TIME_BETWEEN_FRAMES * ms, report="stdout", report_period=10000 * ms)


numpy.set_printoptions(threshold=sys.maxsize)
npdense_ = np.array(dense_)
npdense_[npdense_ < 0.1] = -0.01
npdense_[npdense_ >= 0.1] = 0.1

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


for i in range(0, 217):
    grid_input = np.zeros((SIZE_X, SIZE_X))
    grid_output = np.zeros((SIZE_X, SIZE_X))
    for ind_ in spike_index_plot[
        (i * 100.0 < spike_times_plot) & (spike_times_plot < i * 100.0 + 100.0)
    ]:
        grid_output[ind_ // SIZE_X][ind_ % SIZE_X] += 1
    for ind_ in input_spike_index_plot[
        (i * 100.0 < input_spike_times_plot)
        & (input_spike_times_plot < i * 100.0 + 100.0)
    ]:
        grid_input[ind_ // SIZE_X][ind_ % SIZE_X] += 1

    temp__ = npdense_[i] * grid_output
    if np.sum(temp__) > 0:
        eval_spike_map_data_dense.append(temp__ / np.max(temp__))
    else:
        eval_spike_map_data_dense.append(temp__ + 0.00001)
    if np.sum(dense_[i]) > 0:
        eval_spikes_perf.append(temp__)
    input_spike_map_data.append((grid_input).copy())
    output_spike_map_data.append((grid_output).copy())


t_sum_eval = np.sum(np.sum(eval_spikes_perf, axis=1), axis=1)
t_sum_eval_all = np.sum(np.sum(eval_spike_map_data_dense, axis=1), axis=1)
eval_score_final = np.sum(t_sum_eval) / len(t_sum_eval)


# 210.0 in spikes_in.segments[0].spiketrains[37].times
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


SIGMA_EXC = float(sys.argv[1])
SIGMA_INH = float(sys.argv[2])
INTENSITY_EXC = float(sys.argv[3])
INTENSITY_INH = float(sys.argv[4])
LOWER_VOLT_THRESH = float(sys.argv[5])


fig2.savefig(
    f"plots_brian/eval4_{int(eval_score_final)}_sE_{int(10*SIGMA_EXC)}_sI_{int(10*SIGMA_INH)}_iE_{int(10*INTENSITY_EXC)}_iI_{int(10*INTENSITY_INH)}_vt_{LOWER_VOLT_THRESH}.jpg"
)
plt.close(fig2)


with open('eval4_performance.csv', 'a') as file:
    file.write(f'{eval_score_final},{SIGMA_EXC},{SIGMA_INH},{INTENSITY_EXC},{INTENSITY_INH}')

device.delete(force=True)
