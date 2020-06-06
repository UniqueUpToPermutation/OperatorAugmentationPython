import networkx as nx
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import diagnostics as dgn
import augmentation as aug
import scipy.sparse as sci
import pickle


# Returns a random walk on the graph, starting at a random vertex
def perform_random_walks(graph: nx.Graph, walk_length: int, num_walks: int, stationary_dist: np.ndarray):

    n = len(graph)
    walk = [-1]

    for walk_j in range(0, num_walks):
        # Sample from stationary distribution
        walk.append(np.random.choice(range(0, n), p=stationary_dist))

        # Perform random walk
        for i in range(0, walk_length):
            current = walk[-1]
            edge_iter = graph.edges(current, 'weight', default=1)
            outgoing = [v for (u, v, w) in edge_iter]
            weights = np.array([w for (u, v, w) in edge_iter])
            weights /= np.sum(weights)
            walk.append(np.random.choice(outgoing, p=weights))

        walk.append(-1)

    return walk


# Create an approximate graph from a random walk above
# Separate different runs of the walk by a -1
def form_sampled_graph(g: nx.Graph, walk: list):
    new_g = nx.Graph()
    new_g.add_nodes_from(range(0, len(g)))

    edges = [(walk[i - 1], walk[i]) for i in range(1, len(walk)) if walk[i] != -1 and walk[i - 1] != -1]
    edges = [(min(a, b), max(a, b)) for (a, b) in edges]
    normalization_factor = float(len(edges))
    edges_hist = [(g[0][0], g[0][1], len(list(g[1])) / normalization_factor) for g in groupby(edges)]
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

        self.stationary_dist = np.array([d for (v, d) in self.true_graph.degree(weight='weight')])
        self.stationary_dist /= np.sum(self.stationary_dist)

        dgn.MatrixParameterDistribution.__init__(self, parameters, hyperparameters)

    def draw_parameters(self):
        # Gather data to form
        walk_data = perform_random_walks(self.true_graph, self.steps_per_walk, self.num_walks, self.stationary_dist)
        sampled_graph = form_sampled_graph(self.true_graph, walk_data)

        degenerate_verts = [v for (v, d) in sampled_graph.degree(weight='weight') if d == 0.0]
        sampled_graph.add_edges_from([(v, v) for v in degenerate_verts], weight=1.0)

        if not self.b_aux_share_randomness:
            walk_data_aux = perform_random_walks(self.true_graph, self.steps_per_walk_aux,
                                                 self.num_walks_aux, self.stationary_dist)
            sampled_graph_aux = form_sampled_graph(self.true_graph, walk_data_aux)
        else:
            sampled_graph_aux = sampled_graph
        return LaplacianFromRandomWalkParameters(sampled_graph, sampled_graph_aux)

    # Returns D - gamma A
    def convert(self, matrix_parameters) -> aug.MatrixSampleInterface:
        lap = nx.linalg.laplacian_matrix(matrix_parameters.graph)
        adj = nx.linalg.adjacency_matrix(matrix_parameters.graph)
        return aug.DefaultSparseMatrixSample(lap + (1 - self.gamma) * adj)

    # Returns D
    def convert_auxiliary(self, matrix_parameters) -> aug.MatrixSampleInterface:
        n = self.get_dimension()
        ds = [d for (v, d) in self.true_graph.degree(weight='weight')]
        return aug.DefaultSparseMatrixSample(sci.spdiags(ds, 0, n, n))

    def get_dimension(self) -> int:
        return len(self.true_graph)


def main():
    graph = nx.read_edgelist("data/aves-wildbird-network-1/aves-wildbird-network-1.edges",
                             comments="%", data=(('weight', float),))

    graph = nx.convert_node_labels_to_integers(graph)

    gamma = 0.97

    num_walks = 10
    walk_length = 100

    num_sub_runs = 10
    samples_per_sub_run = 10

    params = LaplacianFromRandomWalkParameters(graph, graph)
    hyperparams = LaplacianFromRandomWalkHyperparameters(num_walks, walk_length, 0, 0, gamma, True)
    true_mat_dist = LaplacianFromRandomWalkDistribution(params, hyperparams)
    problem_def = dgn.ProblemDefinition(true_mat_dist)
    diagnostics = dgn.DiagnosticRun(problem_def)

    # For the naive run, we share randomness between the matrix and the rhs operator
    hyperparams_naive = LaplacianFromRandomWalkHyperparameters(num_walks, walk_length, 0, 0, gamma, True)
    true_mat_dist_naive = LaplacianFromRandomWalkDistribution(params, hyperparams)
    problem_def_naive = dgn.ProblemDefinition(true_mat_dist_naive)
    diagnostics_naive = dgn.DiagnosticRun(problem_def_naive)

    # Naive run
    run_naive = dgn.NaiveProblemRun(problem_def_naive)
    run_naive.num_sub_runs = num_sub_runs * samples_per_sub_run
    diagnostics_naive.add_run(run_naive)
    diagnostics_naive.run()

    # Run with basic semi-Bayesian operator augmentation
    run = dgn.AugProblemRun(problem_def)
    run.num_sub_runs = num_sub_runs
    run.samples_per_sub_run = samples_per_sub_run
    diagnostics.add_run(run)

    # Run with energy norm semi-Bayesian operator augmentation
    run = dgn.EnAugProblemRun(problem_def)
    run.num_sub_runs = num_sub_runs
    run.samples_per_sub_run = samples_per_sub_run
    diagnostics.add_run(run)

    '''
    # Run with truncated energy norm semi-Bayesian operator augmentation
    run = dgn.TruncEnAugProblemRun(problem_def, 2)
    run.num_sub_runs = num_sub_runs
    run.samples_per_sub_run = samples_per_sub_run
    diagnostics.add_run(run)

    # Run with truncated energy norm semi-Bayesian operator augmentation
    run = dgn.TruncEnAugProblemRun(problem_def, 4)
    run.num_sub_runs = num_sub_runs
    run.samples_per_sub_run = samples_per_sub_run
    diagnostics.add_run(run)

    # Run with accelerated truncated energy norm semi-Bayesian operator augmentation
    run = dgn.TruncEnAugAccelProblemRun(problem_def, 2)
    run.num_sub_runs = num_sub_runs
    run.samples_per_sub_run = samples_per_sub_run
    diagnostics.add_run(run)

    # Run with accelerated truncated energy norm semi-Bayesian operator augmentation
    run = dgn.TruncEnAugAccelProblemRun(problem_def, 4)
    run.num_sub_runs = num_sub_runs
    run.samples_per_sub_run = samples_per_sub_run
    diagnostics.add_run(run)

    # Run with accelerated truncated energy norm semi-Bayesian operator augmentation
    run = dgn.TruncEnAugAccelProblemRun(problem_def, 6)
    run.num_sub_runs = num_sub_runs
    run.samples_per_sub_run = samples_per_sub_run
    diagnostics.add_run(run)

    # Run hard window truncated energy norm operator augmentation
    run = dgn.TruncEnAugProblemRun(problem_def, 2, window_funcs='hard')
    run.num_sub_runs = num_sub_runs
    run.samples_per_sub_run = samples_per_sub_run
    diagnostics.add_run(run)

    run = dgn.TruncEnAugProblemRun(problem_def, 4, window_funcs='hard')
    run.num_sub_runs = num_sub_runs
    run.samples_per_sub_run = samples_per_sub_run
    diagnostics.add_run(run)
    '''

    # Run all diagnostics and print results
    diagnostics.run()
    print()
    diagnostics_naive.print_results()
    diagnostics.print_results()
    diagnostics.results.append(diagnostics_naive.results[0])
    pickle.dump(diagnostics.results, open('dgn_random_walk_adj.pkl', 'wb'))


if __name__ == "__main__":
    main()
