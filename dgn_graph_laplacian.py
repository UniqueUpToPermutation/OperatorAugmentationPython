import diagnostics as dgn
import numpy as np
import augmentation as aug
import scipy.sparse.linalg as spla
import pickle
import networkx as nx
import matplotlib.pyplot as plt


def add_noise_to_graph(graph: nx.Graph, variance):
    g_new = nx.Graph(graph)

    k = 1 / variance
    theta = variance

    e_dict = {}

    for (u, v, w) in g_new.edges.data('weight'):
        new_w = w * np.random.gamma(k, theta)
        e_dict[(u, v)] = new_w

    nx.set_edge_attributes(g_new, e_dict, 'weight')

    return g_new


class GraphLaplacianParameters:
    def __init__(self, graph):
        self.graph = graph


class GraphLaplacianHyperparameters:
    def __init__(self, variance, interior):
        self.variance = variance
        self.interior = interior


class GraphLaplacianDistribution(dgn.MatrixParameterDistribution):
    def __init__(self, parameters: GraphLaplacianParameters, hyperparameters: GraphLaplacianHyperparameters):
        self.true_graph = parameters.graph
        self.variance = hyperparameters.variance
        self.interior = hyperparameters.interior
        dgn.MatrixParameterDistribution.__init__(self, parameters, hyperparameters)

    def draw_parameters(self):
        noisy_graph = add_noise_to_graph(self.true_graph, self.variance)
        return GraphLaplacianParameters(noisy_graph)

    def convert(self, matrix_parameters) -> aug.MatrixSampleInterface:
        lap = nx.linalg.laplacian_matrix(matrix_parameters.graph)
        sub_lap = lap[self.interior, :][:, self.interior]
        return aug.DefaultMatrixSample(sub_lap)

    def get_dimension(self) -> int:
        return len(self.interior)


def main():
    graph = nx.read_edgelist("data/aves-wildbird-network-1/aves-wildbird-network-1.edges",
                             comments="%", data=(('weight', float),))

    boundary = {0, 20, 30, 60, 100, 110}
    interior = list(set(range(0, len(graph))).difference(boundary))
    variance = 0.5

    nx.draw(graph, node_size=40)
    plt.show()

    params = GraphLaplacianParameters(graph)
    hyperparams = GraphLaplacianHyperparameters(variance, interior)
    true_mat_dist = GraphLaplacianDistribution(params, hyperparams)
    problem_def = dgn.ProblemDefinition(true_mat_dist)
    diagnostics = dgn.DiagnosticRun(problem_def)

    num_sub_runs = 100
    samples_per_sub_run = 20

    # Naive run
    run = dgn.NaiveProblemRun(problem_def)
    run.num_sub_runs = num_sub_runs
    diagnostics.add_run(run)

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

    # Run all diagnostics and print results
    diagnostics.run()
    print()
    diagnostics.print_results()
    pickle.dump(diagnostics.results, open('dgn_graph_laplacian.pkl', 'wb'))


if __name__ == "__main__":
    main()