import diagnostics as dgn
import numpy as np
import augmentation as aug
from scipy.linalg import solveh_banded
import pickle
import scipy.sparse as scsp
import scipy.sparse.linalg as spla


def form_laplacian2d(a: np.ndarray, h: float) -> scsp.spmatrix:
    n, m = a.shape
    incidence_mat_n = scsp.diags([1, -1], [0, -1], shape=(n-1, n-2))
    incidence_mat_m = scsp.diags([1, -1], [0, -1], shape=(m-1, m-2))
    id_n = scsp.identity(n - 2)
    id_m = scsp.identity(m - 2)

    incidence_mat_rows = scsp.kron(id_m, incidence_mat_n)
    incidence_mat_cols = scsp.kron(incidence_mat_m, id_n)

    a_fac_cols = (a[:-1, 1:-1] + a[1:, 1:-1]) / 2.0
    a_fac_rows = (a[1:-1, :-1] + a[1:-1, 1:]) / 2.0

    a_fac_rows = np.reshape(a_fac_rows, incidence_mat_rows.shape[0])
    a_fac_cols = np.reshape(a_fac_cols, incidence_mat_cols.shape[0])

    row_contrib = incidence_mat_rows.transpose() @ \
                  scsp.spdiags(a_fac_rows, 0, a_fac_rows.shape[0], a_fac_rows.shape[0]) @ incidence_mat_rows
    col_contrib = incidence_mat_cols.transpose() @ \
                  scsp.spdiags(a_fac_cols, 0, a_fac_cols.shape[0], a_fac_cols.shape[0]) @ incidence_mat_cols

    return -(col_contrib + row_contrib) / (h ** 2)


def perturb_background2d(a: np.ndarray, std_dev: float) -> np.ndarray:
    tmp = std_dev * np.sign(np.random.rand(a.shape[0]) - 0.5) + 1.0
    return a * tmp


class GridLaplacianParameters2D:
    def __init__(self, true_a):
        self.true_a = true_a


class GridLaplacianHyperparameters2D:
    def __init__(self, std_dev, h):
        self.std_dev = std_dev
        self.h = h


class GridLaplacianDistribution2D(dgn.MatrixParameterDistribution):
    def __init__(self, parameters: GridLaplacianParameters2D, hyperparameters: GridLaplacianHyperparameters2D):
        self.true_a = parameters.true_a
        self.std_dev = hyperparameters.std_dev
        self.h = hyperparameters.h
        self.dimension = parameters.true_a.shape[0] - 1
        dgn.MatrixParameterDistribution.__init__(self, parameters, hyperparameters)

    def draw_parameters(self):
        noisy_a = perturb_background2d(self.true_a, self.std_dev)
        return GridLaplacianParameters2D(noisy_a)

    def convert(self, matrix_parameters) -> aug.MatrixSampleInterface:
        return aug.DefaultMatrixSample(form_laplacian2d(matrix_parameters.true_a, self.h))

    def get_dimension(self) -> int:
        return self.dimension


def main():

    n = 128
    h = 1.0 / (n - 1)
    std_dev = 0.5
    true_a = np.ones((n, n))
    xs = np.arange(1, n - 1) * h
    ys = np.arange(1, n - 1) * h
    grid_x, grid_y = np.meshgrid(xs, ys)
    b_distribution = lambda: (np.cos(2.0 * np.pi * grid_x) * np.sin(4.0 * np.pi * grid_y)).flatten()

    params = GridLaplacianParameters2D(true_a)
    hyperparams = GridLaplacianHyperparameters2D(std_dev, h)
    true_mat_dist = GridLaplacianDistribution2D(params, hyperparams)
    problem_def = dgn.ProblemDefinition(true_mat_dist)
    problem_def.b_distribution = aug.VectorDistributionFromLambda(b_distribution) #Set distribution of rhs
    diagnostics = dgn.DiagnosticRun(problem_def)

    # Naive run
    run = dgn.NaiveProblemRun(problem_def)
    run.num_sub_runs = 100 * 100
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
    pickle.dump(diagnostics.results, open('dgn_grid_laplacian2d.pkl', 'wb'))


if __name__ == "__main__":
    main()
