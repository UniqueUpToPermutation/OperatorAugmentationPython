import networkx as nx
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import diagnostics as dgn
import augmentation as aug
import scipy.sparse as sci


# Returns a random walk on the graph, starting at a random vertex
def perform_random_walks(adjacency, walk_length, num_walks, stationary_dist):

    n = adjacency.shape[0]
    walk = [-1]

    for walk_j in range(0, num_walks):
        # Sample from stationary distribution
        walk.append(np.random.choice(range(0, n), stationary_dist))

        # Perform random walk
        for i in range(0, walk_length):
            current = walk[-1]
            current_row = adjacency.getrow(current)
            # Sample based on adjacency weight
            current_row = current_row / current_row.sum()
            indices = current_row.nonzero()
            values = [current_row[index] for index in indices]
            next_index = np.random.choice(np.arange(0, len(values)), p=values)
            next_vertex = indices[next_index]
            walk.append(next_vertex)

        walk.append(-1)

    return walk


# Create an approximate graph from a random walk above
# Separate different runs of the walk by a -1
def form_sampled_graph(g, walk):
    new_g = nx.Graph()
    new_g.add_nodes_from(range(0, len(g)))

    edges = [(walk[i - 1], walk[i]) for i in range(1, len(walk)) if walk[i] != -1 and walk[i - 1] != -1]
    edges = [(min(a, b), max(a, b)) for (a, b) in edges]
    normalization_factor = float(len(edges))
    edges_hist = [(list(next(iter(group)))[0], list(next(iter(group)))[0][1],
                   float(len(group)) / normalization_factor) for key, group in groupby(edges)]
    new_g.add_weighted_edges_from(edges_hist)
    return new_g


class LaplacianFromRandomWalkParameters:
    def __init__(self, graph: nx.Graph, aux_graph: nx.Graph):
        self.graph = graph
        self.aux_graph = aux_graph


class LaplacianFromRandomWalkHyperparameters:
    def __init__(self, num_walks: int, steps_per_walk: int, num_walks_aux: int,
                 steps_per_walk_aux: int, gamma: float, b_aux_share_randomness: bool):
        self.num_walks = num_walks
        self.steps_per_walk = steps_per_walk
        self.num_walks_aux = num_walks_aux
        self.steps_per_walk_aux = steps_per_walk_aux
        self.gamma = gamma
        self.b_aux_share_randomness = b_aux_share_randomness


class LaplacianFromRandomWalkDistribution(dgn.MatrixParameterDistribution):
    def __init__(self, parameters: LaplacianFromRandomWalkParameters,
                 hyperparameters: LaplacianFromRandomWalkHyperparameters):
        self.true_graph = parameters.graph
        self.num_walks = hyperparameters.num_walks
        self.steps_per_walk = hyperparameters.steps_per_walk
        self.num_walks_aux = hyperparameters.num_walks_aux
        self.steps_per_walk_aux = hyperparameters.steps_per_walk_aux
        self.gamma = hyperparameters.gamma
        self.b_aux_share_randomness = hyperparameters.b_aux_share_randomness
        self.adjacency = nx.linalg.adjacency_matrix(self.true_graph)

        self.stationary_dist = [d for (v, d) in self.true_graph.degree(weight='weight')]
        self.stationary_dist /= np.sum(self.stationary_dist)

        dgn.MatrixParameterDistribution.__init__(self, parameters, hyperparameters)

    def draw_parameters(self):
        # Gather data to form
        walk_data = perform_random_walks(self.adjacency, self.steps_per_walk, self.num_walks, self.stationary_dist)
        sampled_graph = form_sampled_graph(self.true_graph, walk_data)
        if not self.b_aux_share_randomness:
            walk_data_aux = perform_random_walks(self.adjacency, self.steps_per_walk_aux,
                                                 self.num_walks_aux, self.stationary_dist)
            sampled_graph_aux = form_sampled_graph(self.true_graph, walk_data_aux)
        else:
            sampled_graph_aux = sampled_graph
        return LaplacianFromRandomWalkParameters(sampled_graph, sampled_graph_aux)

    # Returns D - gamma A
    def convert(self, matrix_parameters) -> aug.MatrixSampleInterface:
        lap = nx.linalg.laplacian_matrix(matrix_parameters.graph)
        adj = nx.linalg.adjacency_matrix(matrix_parameters.graph)
        return aug.DefaultMatrixSample(lap + (1 - self.gamma) * adj)

    # Returns D
    def convert_auxiliary(self, matrix_parameters) -> aug.MatrixSampleInterface:
        n = self.get_dimension()
        ds = [d for (v, d) in self.true_graph.degree(weight='weight')]
        return aug.DefaultMatrixSample(sci.spdiags(ds, 0, n, n))

    def get_dimension(self) -> int:
        return len(self.true_graph)


def main():
    graph = nx.read_edgelist("data/aves-wildbird-network-1/aves-wildbird-network-1.edges",
                             comments="%", data=(('weight', float),))

    gamma = 0.97
    lap = nx.linalg.laplacian_matrix(graph)
    adj = nx.linalg.adjacency_matrix(graph)
    deg = lap + adj
    scaled_lap = deg - gamma * adj

    pos = nx.spring_layout(graph)

    k = 10
    (val, modes) = spla.eigsh(lap, k=k, sigma=-0.0001)

    print(val)

    for i in range(1, len(val)):
        nx.draw(graph, node_size=40, pos=pos, node_color=modes[:, i], cmap=plt.get_cmap("Reds"))
        plt.show()


if __name__ == "__main__":
    main()
