import numpy as np
import matplotlib.pyplot as plt


class AttractorLayer:
    def __init__(self, n_x=64, n_y=64):
        self.n_x = n_x  # X length of the grid points
        self.n_y = n_y  # Y length of the grid points
        self.neuron_activities = np.zeros((self.n_x, self.n_y))
        self.recurrent_connections = np.zeros(
            (self.n_x * self.n_y, self.n_x * self.n_y)
        )

    def visualize_neuron_activities(self):
        # Visualize cell_dists
        fig, ax = plt.subplots()
        cell_dists_map = ax.imshow(self.neuron_activities)
        fig.colorbar(cell_dists_map)
        plt.show()
