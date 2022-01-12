import os
import cv2
import json
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation


## %run ../carrada_utils/scripts/set_path.py './components/carrada_datasets/'

TAU = 1
INTENSITY = 1
BETA = 1
SIGMA = 1.3
SHIFT = 0.0
CUTOFF_DIST = 10
X_EYE = 10.0
Y_EYE = 1.0

RES_X = 64


class SimulatedRadar:
    def __init__(self, x_start=30, y_start=1, sigma=SIGMA, noise_factor=0.1):
        self.x_start = x_start
        self.y_start = y_start
        self.sigma = SIGMA

        self.path = []
        self.path_2 = []
        self.gt_path = []
        self.gt_path_2 = []
        x = self.x_start
        y = self.y_start
        self.noise_factor = noise_factor
        while 0 < x < RES_X - 1 and 0 < y < RES_X - 1:
            x += random.randint(-1, 1)
            y += random.randint(0, 1)
            if random.randint(1, 20) % 10 == 0:
                self.path.append(random.randint(0, 63) + random.randint(0, 63) * RES_X)
            else:
                self.path.append(x + y * RES_X)
            self.gt_path.append(x + y * RES_X)

        x = self.x_start + 10
        y = self.y_start + 61
        while 0 < x < RES_X - 1 and 0 < y < RES_X - 1:
            x += random.randint(-1, 1)
            y -= random.randint(0, 1)
            if random.randint(1, 20) % 10 == 0:
                self.path_2.append(
                    random.randint(0, 63) + random.randint(0, 63) * RES_X
                )
            else:
                self.path_2.append(x + y * RES_X)
            self.gt_path_2.append(x + y * RES_X)

    def get_paths(self):
        min_length = min((len(self.path), len(self.path_2)))
        paths = np.empty([2, min_length, 2], dtype=int)

        for i in range(0, min_length):
            paths[0][i] = [self.gt_path[i] // 64, self.gt_path[i] % 64]

        for i in range(0, min_length):
            paths[1][i] = [self.gt_path_2[i] // 64, self.gt_path_2[i] % 64]

        return paths

    def get_random_datastream(self, resize=(64, 64)):
        datastream = []
        path = self.path
        path_2 = self.path_2

        for point_nr in range(0, min(len(path), len(path_2))):
            activations = []
            j = path[point_nr]
            jj = path_2[point_nr]
            grid = np.empty((RES_X, RES_X))
            if random.randint(0, 100) % 50 == 0:
                datastream.append(cv2.resize(grid, resize).copy())
                continue

            for i in range(0, RES_X * RES_X):
                dist = np.sqrt(
                    X_EYE * (i // RES_X - j // RES_X) ** 2
                    + Y_EYE * (i % RES_X - j % RES_X) ** 2
                )
                dist_2 = np.sqrt(
                    X_EYE * (i // RES_X - jj // RES_X) ** 2
                    + Y_EYE * (i % RES_X - jj % RES_X) ** 2
                )
                if dist > CUTOFF_DIST:
                    transfer_func_1 = 0
                else:
                    transfer_func_1 = BETA * (
                        (INTENSITY * np.exp(-dist / (SIGMA ** 2))) - SHIFT
                    )
                if dist_2 > CUTOFF_DIST:
                    transfer_func_2 = 0
                else:
                    transfer_func_2 = BETA * (
                        (INTENSITY * np.exp(-dist_2 / (SIGMA ** 2))) - SHIFT
                    )
                transfer_func = transfer_func_1  # + transfer_func_2 # enable second object

                activations.append(transfer_func)
                if random.randint(0, 45) % 40 == 0:
                    grid[i // RES_X][i % RES_X] = 0
                else:
                    grid[i // RES_X][i % RES_X] = transfer_func
                if np.max(grid) > 0.001:
                    resized_grid = cv2.resize((grid / np.max(grid)), resize).copy()
                else:
                    resized_grid = np.zeros((RES_X, RES_X))
            for i in range(0, RES_X * RES_X):
                if random.randint(0, 100) % 50 == 0:
                    resized_grid[i // RES_X][i % RES_X] = np.max(resized_grid) * self.noise_factor
                    
            datastream.append(resized_grid / np.max(resized_grid))

        return datastream

    def get_random_datastream_spiking(
        self, clip_and_normalize=True, size_x=64, encoding_type="rate"
    ):
        time_bw_frames = 100.0  # miliseconds
        data = self.get_random_datastream(resize=(size_x, size_x))
        if encoding_type == "rate":
            data_spike = [[] for _ in range(size_x * size_x)]
            time_step = 0.0
            for datum in data:
                for i in np.linspace(time_step, time_step + 100.0 - 1, 100):
                    if i - time_step == 0:
                        continue
                    indices = np.argwhere(
                        (i - time_step) % (1 / (datum + 0.0000001)) < 1
                    )
                    for indi in indices:
                        try:
                            data_spike[indi[0] * size_x + indi[1]].append(i)
                        except Exception as e:
                            print(indi)
                time_step += 100.0
            return data_spike
        else:
            return []

    def get_random_datastream_spiking_brian2(
        self, clip_and_normalize=True, size_x=64, encoding_type="rate"
    ):
        data_spike = []
        data_spike_index = []
        time_bw_frames = 100.0  # miliseconds
        data = self.get_random_datastream(resize=(size_x, size_x))
        if encoding_type == "rate":
            time_step = 0.0
            for datum in data:
                for i in np.linspace(time_step, time_step + 100.0 - 1, 100):
                    if i - time_step == 0:
                        continue
                    indices = np.argwhere(
                        (i - time_step) % (1 / (datum + 0.0000001)) < 1
                    )
                    for indi in indices:
                        try:
                            data_spike.append(i)
                            data_spike_index.append(int(indi[0] * size_x + indi[1]))
                        except Exception as e:
                            print(indi)
                time_step += 100.0
            return data_spike, data_spike_index
        else:
            return data_spike, data_spike_index
