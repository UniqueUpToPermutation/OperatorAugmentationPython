import networkx as nx
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla


# Returns a random walk on the graph, starting at a random vertex
def perform_random_walk(adjacency, length):
    walk = [np.random.randint(0, adjacency.shape[0])]

    for i in range(1, length):
        current = walk[-1]
        current_row = adjacency.getrow(current)
        # Sample based on adjacency weight
        current_row = current_row / current_row.sum()
        indices = current_row.nonzero()
        values = [current_row[index] for index in indices]
        next_index = np.random.choice(np.arange(0, len(values)), p=values)
        next_vertex = indices[next_index]
        walk.append(next_vertex)

    return walk


# Create an approximate graph from a random walk above
# Separate different runs of the walk by a -1
def form_walk_sampled_graph(g, walk):
    new_g = nx.Graph()
    new_g.add_nodes_from(range(0, len(g)))

    edges = [(walk[i - 1], walk[i]) for i in range(1, len(walk)) if walk[i] != -1 and walk[i - 1] != -1]
    edges_hist = [(group[0][0], group[0][1], len(list(group))) for key, group in groupby(edges)]
    new_g.add_weighted_edges_from(edges_hist)
    return new_g


# Use a second order estimator
def compute_beta_approx(graph, samples):
    return 0


def main():
    graph = nx.read_edgelist("data/aves-wildbird-network-1/aves-wildbird-network-1.edges",
                             comments="%", data=(('weight', float),))

    gamma = 0.95
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
