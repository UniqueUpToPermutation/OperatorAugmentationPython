import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy.linalg import solveh_banded
import augmentation as aug


def form_laplacian(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    h = 1.0/n
    mid = (a[:-1] + a[1:]) / (h ** 2)
    left = -a[1:-1] / (h ** 2)
    left = np.concatenate((np.array([0.0]), left))
    return np.vstack((left, mid))


def perturb_background(a: np.ndarray, std_dev:float) -> np.ndarray:
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


class GridLaplacianDistribution(aug.MatrixDistributionInterface):
    def __init__(self, tru_a, std_dev):
        self.tru_a = tru_a
        self.std_dev = std_dev

    def draw_a_sample(self):
        return perturb_background(self.tru_a, self.std_dev)

    def convert_to_matrix(self, noisy_a):
        return form_laplacian(noisy_a)

    def draw_sample(self) -> aug.MatrixSampleInterface:
        return GridLaplacianMatrixSample(self.convert_to_matrix(self.draw_a_sample()))


def run_test(tru_a, runs, samples_per_run, std_dev):

    n = tru_a.shape[0]

    xs = np.arange(1, n) / n
    b = np.cos(2.0 * np.pi * xs)

    # Let the right hand side be a complete random realization
    b_dist = lambda: b
    q_u_dist = aug.IdenticalVectorPairDistributionFromLambda(b_dist)
    q_dist = aug.VectorDistributionFromLambda(b_dist)

    tru_dist = GridLaplacianDistribution(tru_a, std_dev)
    tru_mat = GridLaplacianMatrixSample(tru_dist.convert_to_matrix(tru_a))

    errs_l2_naive_squared = []
    errs_l2_aug_squared = []
    errs_l2_en_aug_squared = []
    errs_energy_naive_squared = []
    errs_energy_aug_squared = []
    errs_energy_en_aug_squared = []

    for run in range(0, runs):

        print(f'Run {run+1} / {runs}...')

        b = b_dist()
        noisy_a = tru_dist.draw_a_sample()
        bootstrap_dist = GridLaplacianDistribution(noisy_a, std_dev)
        noisy_mat = GridLaplacianMatrixSample(bootstrap_dist.convert_to_matrix(noisy_a))

        op_Ahat_inv = lambda rhs: noisy_mat.solve(rhs)
        op_Ahat = lambda rhs: noisy_mat.apply(rhs)

        tru_solution = tru_mat.solve(b)
        naive_solution = op_Ahat_inv(b)

        # Perform simple augmentation using a scaled identity prior on b
        aug_solution = aug.aug_sym(samples_per_run, 1, b,
                                   op_Ahat_inv, bootstrap_dist, q_u_dist)
        # Perform energy-augmentation using a scaled identity prior on b
        en_aug_solution = aug.en_aug(samples_per_run, 1, b,
                                     op_Ahat_inv, op_Ahat, bootstrap_dist, q_dist)

        err_naive = naive_solution - tru_solution
        errs_l2_naive_squared.append(np.dot(err_naive, err_naive))
        errs_energy_naive_squared.append(np.dot(err_naive, tru_mat.apply(err_naive)))

        err_aug = aug_solution - tru_solution
        errs_l2_aug_squared.append(np.dot(err_aug, err_aug))
        errs_energy_aug_squared.append(np.dot(err_aug, tru_mat.apply(err_aug)))

        err_en_aug = en_aug_solution - tru_solution
        errs_l2_en_aug_squared.append(np.dot(err_en_aug, err_en_aug))
        errs_energy_en_aug_squared.append(np.dot(err_en_aug, tru_mat.apply(err_en_aug)))

    errs_l2_naive_squared = np.array(errs_l2_naive_squared)
    errs_l2_aug_squared = np.array(errs_l2_aug_squared)
    errs_l2_en_aug_squared = np.array(errs_l2_en_aug_squared)

    errs_energy_naive_squared = np.array(errs_energy_naive_squared)
    errs_energy_aug_squared = np.array(errs_energy_aug_squared)
    errs_energy_en_aug_squared = np.array(errs_energy_en_aug_squared)

    sqrt_runs = np.sqrt(runs)

    print(f'Average naive L2 error squared: {np.mean(errs_l2_naive_squared)} +- '
          f'{2 *np.std(errs_l2_naive_squared) / sqrt_runs}')
    print(f'Average augmented L2 error squared: {np.mean(errs_l2_aug_squared)} +- '
          f'{2 * np.std(errs_l2_aug_squared) / sqrt_runs}')
    print(f'Average energy-augmented L2 error squared: {np.mean(errs_l2_en_aug_squared)} +- '
          f'{2 *np.std(errs_l2_en_aug_squared) / sqrt_runs}')
    print('')
    print(f'Average naive energy error squared: {np.mean(errs_energy_naive_squared)} +- '
          f'{2 *np.std(errs_energy_naive_squared) / sqrt_runs}')
    print(f'Average augmented energy error squared: {np.mean(errs_energy_aug_squared)} +- '
          f'{2 *np.std(errs_energy_aug_squared) / sqrt_runs}')
    print(f'Average energy-augmented energy error squared: {np.mean(errs_energy_en_aug_squared)} +- '
          f'{2 *np.std(errs_energy_en_aug_squared) / sqrt_runs}')


def main():
    n = 128
    run_test(np.ones(n), 100, 100, 0.5)


if __name__ == '__main__':
    main()
