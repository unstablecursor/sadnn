import numpy as np
import matplotlib.pyplot as plt

# TODO: Training by genetic algorithm generating these values?
TAU = 0.95  # TODO: I don't know whether this is important
INTENSITY = 0.3  # TODO : Check this
CUTOFF_DIST = 10  # TODO : Change this
SIGMA = 0.24  # TODO: Check this
SHIFT = 0.05  # TODO : Check this
# TODO : Also maybe plot the equation like in the paper
# A MODEL OF GRID CELLS BASED ON
# A TWISTED TORUS TOPOLOGY


class AttractorLayer:
    def __init__(self, n_x=64, n_y=64):
        self.n_x = n_x  # X length of the grid points
        self.n_y = n_y  # Y length of the grid points
        self.neuron_activities = np.zeros(self.n_x * self.n_y)
        self.new_neuron_activities = np.zeros(self.n_x * self.n_y)
        self.inter_neuron_connections = np.zeros(
            (self.n_x * self.n_y, self.n_x * self.n_y)
        )

    def visualize_neuron_activities(self):
        # Visualize cell_dists
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

    def update_activity(self, i):
        """
        Updates neuron activity of neuron i
        :param i: Neuron index
        """
        # TODO : Incomplete, also need to include the neuron itself.
        tf_result = self.transfer_function(i)
        self.new_neuron_activities[i] = tf_result * (
            1 - TAU
        ) + TAU * tf_result / np.sum(self.neuron_activities)

    def get_distance_bw_neurons(self, i, j):
        """
        Get distance between two neurons
        :param i: Neuron index of first neuron
        :param j: Neuron index of second neuron
        :return: Distance between two neurons
        """
        return np.sqrt(
            (i / self.n_x - j / self.n_x) ** 2 + (i % self.n_x - j % self.n_x) ** 2
        )

    def set_weight(self, i, j):
        """
        Set weight between neurons
        :param i: Index of first neuron
        :param j: Index of second neuron
        """
        if i != j:
            dist = self.get_distance_bw_neurons(i, j)
            if dist < CUTOFF_DIST:
                self.inter_neuron_connections[i][j] = (
                    INTENSITY * np.exp(-dist / (SIGMA ** 2)) - SHIFT
                )
            else:
                self.inter_neuron_connections[i][j] = 0
        else:
            self.inter_neuron_connections[i][j] = 0

    def set_weights(self):
        # TODO: This function sucks. Please make this more eloquent
        for i in range(0, self.n_x * self.n_y):
            for j in range(0, self.n_x * self.n_y):
                self.set_weight(i, j)

    def evaluate_input(self, data_entry):
        pass

    def forward_pass(self, data_entry):
        pass
