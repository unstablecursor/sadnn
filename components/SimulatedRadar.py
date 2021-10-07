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
    def __init__(self, x_start=30, y_start=1, sigma=SIGMA):
        self.x_start = x_start
        self.y_start = y_start
        self.sigma = SIGMA

        self.path = []
        x = self.x_start
        y = self.y_start
        while 0 < x < RES_X - 1 and 0 < y < RES_X - 1:
            x += random.randint(-1, 1)
            y += random.randint(0, 1)
            if random.randint(0, 20) % 5 == 0:
                self.path.append(random.randint(0, 64) + random.randint(0, 64) * RES_X)
            else:
                self.path.append(x + y * RES_X)

    def get_path(self):
        return self.path

    def get_random_datastream(self):
        datastream = []
        path = self.get_path()
        for point in path:
            activations = []
            distances = []
            j = point
            grid = np.empty((RES_X, RES_X))
            if random.randint(0, 100) % 20 == 0:
                datastream.append(grid.copy())
                continue

            for i in range(0, RES_X * RES_X):
                dist = np.sqrt(
                    X_EYE * (i // RES_X - j // RES_X) ** 2
                    + Y_EYE * (i % RES_X - j % RES_X) ** 2
                )
                if dist > CUTOFF_DIST:
                    transfer_func = 0
                else:
                    transfer_func = BETA * (
                        (INTENSITY * np.exp(-dist / (SIGMA ** 2))) - SHIFT
                    )

                distances.append(dist)
                activations.append(transfer_func)
                if random.randint(0, 45) % 40 == 0:
                    grid[i // RES_X][i % RES_X] = 0
                else:
                    grid[i // RES_X][i % RES_X] = transfer_func
            datastream.append(grid.copy())

        return datastream
