import networkx as nx
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import augmentation as aug


def add_noise_to_graph(graph : nx.Graph, variance):
    g_new = nx.Graph(graph)

    k = 1 / variance
    theta = variance

    dict = {}

    for (u, v, w) in g_new.edges.data('weight'):
        new_w = w * np.random.gamma(k, theta)
        dict[(u, v)] = new_w

    nx.set_edge_attributes(g_new, dict, 'weight')

    return g_new


class NoisyLaplacianDistribution(aug.MatrixDistributionInterface):
    def __init__(self, tru_graph, interior, variance):
        self.tru_graph = tru_graph
        self.interior = interior
        self.variance = variance

    def draw_graph_sample(self):
        return add_noise_to_graph(self.tru_graph, self.variance)

    def convert_to_matrix(self, noisy_graph):
        lap = nx.linalg.laplacian_matrix(noisy_graph)
        return lap[self.interior, :][:, self.interior]

    def draw_sample(self) -> aug.MatrixSampleInterface:
        noisy_g = self.draw_graph_sample()
        return aug.DefaultSparseMatrixSample(self.convert_to_matrix(noisy_g))


def run_test(tru_graph, runs, samples_per_run, boundary, variance):

    interior = list(set(range(0, len(tru_graph))).difference(boundary))
    interior_dimension = len(interior)

    # Let the right hand side be a complete random realization
    b_dist = lambda: np.random.randn(interior_dimension)
    q_u_dist = aug.IdenticalVectorPairDistributionFromLambda(b_dist)
    q_dist = aug.VectorDistributionFromLambda(b_dist)

    tru_dist = NoisyLaplacianDistribution(tru_graph, interior, variance)
    tru_mat = tru_dist.convert_to_matrix(tru_graph)

    errs_l2_naive_squared = []
    errs_l2_aug_squared = []
    errs_l2_en_aug_squared = []
    errs_energy_naive_squared = []
    errs_energy_aug_squared = []
    errs_energy_en_aug_squared = []

    for run in range(0, runs):

        print(f'Run {run+1} / {runs}...')

        b = b_dist()
        noisy_graph = tru_dist.draw_graph_sample()
        bootstrap_dist = NoisyLaplacianDistribution(noisy_graph, interior, variance)
        noisy_mat = bootstrap_dist.convert_to_matrix(noisy_graph)

        op_Ahat_inv = lambda rhs: spla.spsolve(noisy_mat, rhs)
        op_Ahat = lambda rhs: noisy_mat @ rhs

        tru_solution = spla.spsolve(tru_mat, b)
        naive_solution = op_Ahat_inv(b)

        # Perform simple augmentation using a scaled identity prior on b
        aug_solution = aug.aug_sym(samples_per_run, 1, b,
                                   op_Ahat_inv, bootstrap_dist, q_u_dist)
        # Perform energy-augmentation using a scaled identity prior on b
        en_aug_solution = aug.en_aug(samples_per_run, 1, b,
                                     op_Ahat_inv, op_Ahat, bootstrap_dist, q_dist)

        err_naive = naive_solution - tru_solution
        errs_l2_naive_squared.append(np.dot(err_naive, err_naive))
        errs_energy_naive_squared.append(np.dot(err_naive, tru_mat @ err_naive))

        err_aug = aug_solution - tru_solution
        errs_l2_aug_squared.append(np.dot(err_aug, err_aug))
        errs_energy_aug_squared.append(np.dot(err_aug, tru_mat @ err_aug))

        err_en_aug = en_aug_solution - tru_solution
        errs_l2_en_aug_squared.append(np.dot(err_en_aug, err_en_aug))
        errs_energy_en_aug_squared.append(np.dot(err_en_aug, tru_mat @ err_en_aug))

    errs_l2_naive_squared = np.array(errs_l2_naive_squared)
    errs_l2_aug_squared = np.array(errs_l2_aug_squared)
    errs_l2_en_aug_squared = np.array(errs_l2_en_aug_squared)

    errs_energy_naive_squared = np.array(errs_energy_naive_squared)
    errs_energy_aug_squared = np.array(errs_energy_aug_squared)
    errs_energy_en_aug_squared = np.array(errs_energy_en_aug_squared)

    sqrt_runs = np.sqrt(runs)

    print(f'Average naive L2 error squared: {np.mean(errs_l2_naive_squared)} +- '
          f'{2 * np.std(errs_l2_naive_squared) / sqrt_runs}')
    print(f'Average augmented L2 error squared: {np.mean(errs_l2_aug_squared)} +- '
          f'{2 * np.std(errs_l2_aug_squared) / sqrt_runs}')
    print(f'Average energy-augmented L2 error squared: {np.mean(errs_l2_en_aug_squared)} +- '
          f'{2 * np.std(errs_l2_en_aug_squared) / sqrt_runs}')
    print('')
    print(f'Average naive energy error squared: {np.mean(errs_energy_naive_squared)} +- '
          f'{2 * np.std(errs_energy_naive_squared) / sqrt_runs}')
    print(f'Average augmented energy error squared: {np.mean(errs_energy_aug_squared)} +- '
          f'{2 * np.std(errs_energy_aug_squared) / sqrt_runs}')
    print(f'Average energy-augmented energy error squared: {np.mean(errs_energy_en_aug_squared)} +- '
          f'{2 * np.std(errs_energy_en_aug_squared) / sqrt_runs}')


def main():
    graph = nx.read_edgelist("data/aves-wildbird-network-1/aves-wildbird-network-1.edges",
                             comments="%", data=(('weight', float),))

    boundary = {4, 20, 35, 80, 100, 110}

    nx.draw(graph, node_size=40)
    plt.show()

    run_test(graph, 100, 10, boundary, 0.5)


if __name__ == "__main__":
    main()
