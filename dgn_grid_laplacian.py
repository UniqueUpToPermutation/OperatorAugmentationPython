import diagnostics as dgn
import numpy as np
import augmentation as aug
from scipy.linalg import solveh_banded
import pickle

def form_laplacian(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    h = 1.0/n
    mid = (a[:-1] + a[1:]) / (h ** 2)
    left = -a[1:-1] / (h ** 2)
    left = np.concatenate((np.array([0.0]), left))
    return np.vstack((left, mid))


def perturb_background(a: np.ndarray, std_dev: float) -> np.ndarray:
    tmp = std_dev * np.sign(np.random.rand(a.shape[0]) - 0.5) + 1.0
    return a * tmp


class GridLaplacianMatrixSample(aug.MatrixSampleInterface):
    def __init__(self, mat_diags):
        self.matrix_diags = mat_diags

    def preprocess(self):
        pass

    def solve(self, b: np.ndarray) -> np.ndarray:
        return solveh_banded(self.matrix_diags, b)

    def apply(self, b: np.ndarray) -> np.ndarray:
        upper_prod = np.concatenate((self.matrix_diags[0, 1:] * b[1:], np.array([0.0])))
        lower_prod = np.concatenate((np.array([0.0]), self.matrix_diags[0, 1:] * b[:-1]))
        return self.matrix_diags[1, :] * b + upper_prod + lower_prod


class GridLaplacianParameters:
    def __init__(self, true_a):
        self.true_a = true_a


class GridLaplacianHyperparameters:
    def __init__(self, std_dev):
        self.std_dev = std_dev


class GridLaplacianDistribution(dgn.MatrixParameterDistribution):
    def __init__(self, parameters: GridLaplacianParameters, hyperparameters: GridLaplacianHyperparameters):
        self.true_a = parameters.true_a
        self.std_dev = hyperparameters.std_dev
        self.dimension = parameters.true_a.shape[0] - 1
        dgn.MatrixParameterDistribution.__init__(self, parameters, hyperparameters)

    def draw_parameters(self):
        noisy_a = perturb_background(self.true_a, self.std_dev)
        return GridLaplacianParameters(noisy_a)

    def convert(self, matrix_parameters) -> aug.MatrixSampleInterface:
        return GridLaplacianMatrixSample(form_laplacian(matrix_parameters.true_a))

    def get_dimension(self) -> int:
        return self.dimension


def main():
    n = 128
    std_dev = 0.5
    true_a = np.ones(n)
    xs = np.arange(1, n) / n
    b_distribution = lambda: np.cos(2.0 * np.pi * xs)

    params = GridLaplacianParameters(true_a)
    hyperparams = GridLaplacianHyperparameters(std_dev)
    true_mat_dist = GridLaplacianDistribution(params, hyperparams)
    problem_def = dgn.ProblemDefinition(true_mat_dist)
    problem_def.b_distribution = aug.VectorDistributionFromLambda(b_distribution) #Set distribution of rhs
    diagnostics = dgn.DiagnosticRun(problem_def)

    # Naive run
    run = dgn.NaiveProblemRun(problem_def)
    run.num_sub_runs = 100
    diagnostics.add_run(run)

    # Run with basic semi-Bayesian operator augmentation
    run = dgn.AugProblemRun(problem_def)
    run.num_sub_runs = 100
    run.samples_per_sub_run = 100
    diagnostics.add_run(run)

    # Run with energy norm semi-Bayesian operator augmentation
    run = dgn.EnAugProblemRun(problem_def)
    run.num_sub_runs = 100
    run.samples_per_sub_run = 100
    diagnostics.add_run(run)

    # Run with truncated energy norm semi-Bayesian operator augmentation
    run = dgn.TruncEnAugProblemRun(problem_def, 2)
    run.num_sub_runs = 100
    run.samples_per_sub_run = 100
    diagnostics.add_run(run)

    # Run with truncated energy norm semi-Bayesian operator augmentation
    run = dgn.TruncEnAugProblemRun(problem_def, 4)
    run.num_sub_runs = 100
    run.samples_per_sub_run = 100
    diagnostics.add_run(run)

    # Run with accelerated truncated energy norm semi-Bayesian operator augmentation
    run = dgn.TruncEnAugAccelProblemRun(problem_def, 2)
    run.num_sub_runs = 100
    run.samples_per_sub_run = 100
    diagnostics.add_run(run)

    # Run with accelerated truncated energy norm semi-Bayesian operator augmentation
    run = dgn.TruncEnAugAccelProblemRun(problem_def, 4)
    run.num_sub_runs = 100
    run.samples_per_sub_run = 100
    diagnostics.add_run(run)

    # Run with accelerated truncated energy norm semi-Bayesian operator augmentation
    run = dgn.TruncEnAugAccelProblemRun(problem_def, 6)
    run.num_sub_runs = 100
    run.samples_per_sub_run = 100
    diagnostics.add_run(run)

    # Run hard window truncated energy norm operator augmentation
    run = dgn.TruncEnAugProblemRun(problem_def, 2, window_funcs='hard')
    run.num_sub_runs = 100
    run.samples_per_sub_run = 100
    diagnostics.add_run(run)

    run = dgn.TruncEnAugProblemRun(problem_def, 4, window_funcs='hard')
    run.num_sub_runs = 100
    run.samples_per_sub_run = 100
    diagnostics.add_run(run)

    # Run all diagnostics and print results
    diagnostics.run()
    print()
    diagnostics.print_results()
    pickle.dump(diagnostics.results, open('dgn_grid_laplacian.pkl', 'wb'))


if __name__ == "__main__":
    main()
