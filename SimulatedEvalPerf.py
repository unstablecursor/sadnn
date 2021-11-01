#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.feature import peak_local_max

from components.AttractorLayerModifiedClipBelowZero import (
    AttractorLayerModifiedClipBelowZero,
)
from components.SimulatedRadar import SimulatedRadar

seq_name = "2020-02-28-13-13-43"

data_loader = SimulatedRadar()
data = data_loader.get_random_datastream()
gt_paths = data_loader.get_paths()

TAU = 1  # TODO: I don't know whether this is important
INTENSITY = 0.1
BETA = 20  # TODO : Check this
SIGMA = 3.0  # 0.5
SHIFT = 0.001  # TODO : Check this
CUTOFF_DIST = 15  # TODO : Change this
X_EYE = 1.0
Y_EYE = 1.0
NR_OF_PASSES = 4
MIN_CLIP = -0.01  # -0.0001
DECAY = 0.2
K_INHIB = 5.0

for decay_ in [0.15, 0.2, 0.25]:
    for beta_ in [10, 15, 20, 25]:
        for shift_ in [0.001, 0.003, 0.005]:
            for sigma_ in [2.0, 3.0, 4.0]:
                for nr_of_passes_ in [3, 4, 5]:
                    print(f"START... ")
                    DECAY = decay_  # * 100
                    BETA = beta_
                    SIGMA = sigma_
                    NR_OF_PASSES = nr_of_passes_
                    SHIFT = shift_  # * 1000

                    attr_layer = AttractorLayerModifiedClipBelowZero(
                        tau=TAU,
                        intensity=INTENSITY,
                        cutoff_dist=CUTOFF_DIST,
                        sigma=SIGMA,
                        shift=SHIFT,
                        beta=BETA,
                        k=K_INHIB,
                        clip=True,
                        x_eye=X_EYE,
                        y_eye=Y_EYE,
                        min_clip=MIN_CLIP,
                        decay=DECAY,
                    )

                    attr_layer.set_weights()
                    attr_layer.save_network(file_path=f"eh_meh_iste.npy")

                    neuron_act = []
                    for data_entry in data:
                        neuron_act.append(
                            attr_layer.forward_pass_visualization(
                                data_entry.flatten(), number_of_passes=NR_OF_PASSES
                            )[NR_OF_PASSES - 1]
                        )

                    def eval_perf(gt_paths_, neuron_acts):
                        gt_point_map = []
                        predicted_point_map = []
                        max_pnt_maps = []
                        score = []
                        for i in range(0, len(neuron_acts)):

                            gt_point_sub_map = np.zeros((64, 64))
                            predicted_point_sub_map = np.zeros((64, 64))
                            max_pts = np.zeros((64, 64))

                            gt_path_points = gt_paths_[:, i]
                            coordinates = peak_local_max(
                                neuron_act[i],
                                min_distance=10,
                                threshold_rel=(np.sum(neuron_act[i]) / (64 * 64)),
                            )

                            loc_ = np.argmax(neuron_acts[i])
                            max_pts[loc_ // 64][loc_ % 64] = 1.0

                            tmp_score = []
                            for ittt_ in gt_path_points:
                                gt_point_sub_map[ittt_[0]][ittt_[1]] = 1.0
                                min_score = 64

                                for coor_ in coordinates:
                                    dist = np.sqrt(
                                        (ittt_[0] - coor_[0]) ** 2
                                        + (ittt_[1] - coor_[1]) ** 2
                                    )
                                    if dist < min_score:
                                        min_score = dist
                                tmp_score.append(min_score)
                            score.append(tmp_score.copy())

                            for ittt_ in coordinates:
                                predicted_point_sub_map[ittt_[0]][ittt_[1]] = 1.0

                            max_pnt_maps.append(max_pts.copy())
                            gt_point_map.append(gt_point_sub_map.copy())
                            predicted_point_map.append(predicted_point_sub_map.copy())
                        return (
                            score,
                            gt_point_map,
                            predicted_point_map,
                            max_pnt_maps,
                        )

                    (
                        score,
                        gt_point_map,
                        predicted_point_map,
                        max_pnt_maps,
                    ) = eval_perf(gt_paths, neuron_act)

                    # Ground Truth Comparison
                    gt_comparison = np.zeros((len(neuron_act), 64, 64, 3))
                    gt_comparison[:, :, :, 1] = np.array(gt_point_map).astype(int)
                    gt_comparison[:, :, :, 2] = np.array(predicted_point_map).astype(
                        int
                    )
                    gt_comparison[:, :, :, 0] = np.array(max_pnt_maps).astype(int)

                    performance_ = np.array(score)

                    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
                        1, 5, figsize=(20, 4), dpi=100
                    )
                    fig.suptitle(
                        f"Simulated Results: MeanSqrdErr_1: {(np.mean(performance_[:,1])/64):.2f} MeanSqrdErr_2: {(np.mean(performance_[:,0])/64):.2f} \n ",
                        fontsize=15,
                    )

                    ax1.tick_params(axis="both", labelsize=8)
                    ax1.set_title("Neuron Activities", fontsize=10)
                    ax2.tick_params(axis="both", labelsize=8)
                    ax2.set_title("Simulated Input", fontsize=10)

                    ax3.tick_params(axis="both", labelsize=8)
                    ax3.set_title("Predicted vs GT", fontsize=10)

                    ax4.tick_params(axis="both", labelsize=8)
                    ax4.set_title("Euclidian dist pred vs GT", fontsize=10)

                    ax5.tick_params(axis="both", labelsize=8)
                    ax5.set_title("Euclidian dist pred vs GT 2", fontsize=10)

                    cell_act_map = ax1.imshow(
                        np.array(neuron_act[0]), cmap="hot", interpolation="none"
                    )
                    simulated_input_map = ax2.imshow(
                        data[0], cmap="hot", interpolation="none"
                    )
                    gt_prediction_comp = ax3.imshow(
                        gt_comparison[0], cmap="hot", interpolation="none"
                    )
                    gt_prec_losses = ax4.plot(performance_[:, 1])
                    gt_prec_losses_2 = ax5.plot(performance_[:, 0])

                    def init():

                        cell_act_map.set_data(np.array(neuron_act[0]))
                        simulated_input_map.set_data(np.array(data[0]))
                        gt_prediction_comp.set_data(np.array(gt_comparison[0]))
                        return [cell_act_map, simulated_input_map]

                    # animation function.  This is called sequentially
                    def animate(i):
                        cell_act_map.set_data(np.array(neuron_act[i]))
                        simulated_input_map.set_data(np.array(data[i]))
                        gt_prediction_comp.set_data(np.array(gt_comparison[i]))
                        return [cell_act_map, simulated_input_map]

                    anim = animation.FuncAnimation(
                        fig,
                        animate,
                        init_func=init,
                        frames=len(data) - 1,
                        interval=1,
                        blit=True,
                    )

                    DECAY = decay_  # * 100
                    BETA = beta_
                    SIGMA = sigma_
                    NR_OF_PASSES = nr_of_passes_
                    SHIFT = shift_  # * 1000

                    anim.save(
                        f"animations/mod_clipbz_{int(MIN_CLIP*1000)}_b_{int(BETA)}_dec_{int(DECAY*10)}_sig_{int(SIGMA)}_pass_{int(NR_OF_PASSES)}.gif",
                        fps=10,
                    )
