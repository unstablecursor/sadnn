import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

# TODO: Training by genetic algorithm generating these values?
TAU = 0.95  # TODO: I don't know whether this is important
INTENSITY = 0.3  # TODO : Check this
CUTOFF_DIST = 10  # TODO : Change this
SIGMA = 0.24  # TODO: Check this
SHIFT = 0.05  # TODO : Check this
BETA = 1.0
K_INHIB = 1.0
X_EYE = 1.0
Y_EYE = 1.0
DECAY = 0.0
MIN_CLIP = 0.0


class AttractorLayerModifiedClipBelowZero:
    def __init__(
        self,
        n_x=64,
        n_y=64,
        tau=TAU,
        intensity=INTENSITY,
        cutoff_dist=CUTOFF_DIST,
        sigma=SIGMA,
        shift=SHIFT,
        beta=BETA,
        k=K_INHIB,
        x_eye=X_EYE,
        y_eye=Y_EYE,
        decay=DECAY,
        min_clip=MIN_CLIP,
        clip=False,
    ):
        self.n_x = n_x  # X length of the grid points
        self.n_y = n_y  # Y length of the grid points
        self.neuron_activities = np.zeros(self.n_x * self.n_y)
        self.new_neuron_potentials = np.zeros(self.n_x * self.n_y)
        self.new_neuron_activities = np.zeros(self.n_x * self.n_y)
        self.inter_neuron_connections = np.zeros(
            (self.n_x * self.n_y, self.n_x * self.n_y)
        )
        self.tau = tau
        self.intensity = intensity
        self.cutoff_dist = cutoff_dist
        self.sigma = sigma
        self.shift = shift
        self.clip = clip
        self.beta = beta
        self.k = k
        self.x_eye = x_eye
        self.y_eye = y_eye
        self.decay = decay
        self.min_clip = min_clip

    def visualize_neuron_activities(self, neurons=None):
        # Visualize cell_dists
        if neurons:
            neuron_activities_2d = np.reshape(neurons, (-1, self.n_y))
        else:
            neuron_activities_2d = np.reshape(self.neuron_activities, (-1, self.n_y))
        fig, ax = plt.subplots()

        cell_dists_map = ax.imshow(neuron_activities_2d)
        fig.colorbar(cell_dists_map)
        plt.show()

    def transfer_function(self, i):
        """
        Computes total excitation from other neurons
        :param i: Neuron index
        :return: overall transfer from other neurons
        """
        return np.sum(self.neuron_activities * self.inter_neuron_connections[i])

    def update_potential(self, i, v_ext):
        """
        Updates neuron activity of neuron i
        :param i: Neuron index
        """
        tf_result = self.beta * self.transfer_function(i)
        self.new_neuron_potentials[i] = (
            tf_result + v_ext + self.neuron_activities[i] - self.decay
        )

    def update_activities(self, external_input):
        for i in range(0, self.n_x * self.n_y):
            self.update_potential(i, external_input[i])
        if self.clip:
            self.new_neuron_potentials = self.new_neuron_potentials.clip(
                min=self.min_clip, max=1
            )
        self.neuron_activities = self.new_neuron_potentials.copy()
        # sqrd_potentials = self.new_neuron_potentials * self.new_neuron_potentials
        # self.neuron_activities = sqrd_potentials

        # self.neuron_activities = sqrd_potentials / self.k * np.sum(sqrd_potentials)
        # Max-out self.neuron_activities = sqrd_potentials / np.max(sqrd_potentials)

    def get_distance_bw_neurons(self, i, j):
        """
        Get distance between two neurons
        :param i: Neuron index of first neuron
        :param j: Neuron index of second neuron
        :return: Distance between two neurons
        """
        return np.sqrt(
            self.x_eye * (i // self.n_x - j // self.n_x) ** 2
            + self.y_eye * (i % self.n_x - j % self.n_x) ** 2
        )

    def set_weight(self, i, j):
        """
        Set weight between neurons
        :param i: Index of first neuron
        :param j: Index of second neuron
        """
        if i != j:
            dist = self.get_distance_bw_neurons(i, j)
            if dist < self.cutoff_dist:
                self.inter_neuron_connections[i][j] = (
                    self.intensity * np.exp(-(dist ** 2) / (self.sigma ** 2))
                ) / (math.pi * 2 * (self.sigma ** 2)) - self.shift
            else:
                self.inter_neuron_connections[i][j] = 0
        else:
            self.inter_neuron_connections[i][j] = 0

    def set_weights(self):
        # TODO: This function sucks. Please make this more eloquent
        for i in range(0, self.n_x * self.n_y):
            for j in range(0, self.n_x * self.n_y):
                self.set_weight(i, j)

    def forward_pass(self, data_entry: np.ndarray, number_of_passes=1):
        self.neuron_activities = data_entry / np.max(data_entry)  # TODO: Check this
        for i in range(0, number_of_passes):
            self.update_activities(data_entry)

    def forward_pass_visualization(self, data_entry: np.ndarray, number_of_passes=1):
        data = []
        self.update_activities(data_entry)
        data.append(np.reshape(self.neuron_activities, (-1, self.n_y)).copy())
        for i in range(0, number_of_passes - 1):
            self.update_activities(np.zeros((self.n_x * self.n_y)))
            data.append(np.reshape(self.neuron_activities, (-1, self.n_y)).copy())
        return data

    def process_data(self, data_stream: [np.ndarray]):
        # TODO: Implement this after testing in notebook
        pass

    def process_data_visalization(self, data_stream: [np.ndarray]):
        # TODO: Implement this after testing in notebook
        pass

    def load_network(self, file_path):
        self.inter_neuron_connections = np.load(file_path)

    def save_network(self, file_path):
        np.save(file_path, self.inter_neuron_connections)
