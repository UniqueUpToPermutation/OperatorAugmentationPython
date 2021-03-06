import numpy as np
import scipy.sparse as spla
from scipy.sparse.linalg import spsolve


class VectorPairDistributionInterface:
    def are_equal(self) -> bool:
        raise Exception('are_equal not implemented!')

    def draw_sample(self) -> (np.ndarray, np.ndarray):
        raise Exception('draw_sample not implemented!')


class VectorDistributionInterface:
    def draw_sample(self) -> np.ndarray:
        raise Exception('draw_sample not implemented!')


class VectorDistributionFromLambda(VectorDistributionInterface):
    def __init__(self, sample_func):
        self.sample_func = sample_func

    def draw_sample(self) -> np.ndarray:
        return self.sample_func()


class IdenticalVectorPairDistributionFromLambda(VectorPairDistributionInterface):
    def __init__(self, sample_func):
        self.sample_func = sample_func

    def are_equal(self) -> bool:
        return True

    def draw_sample(self) -> (np.ndarray, np.ndarray):
        sample = self.sample_func()
        return sample, sample


class MatrixSampleInterface:
    def preprocess(self):
        pass

    def solve(self, b: np.ndarray) -> np.ndarray:
        raise Exception('Solve not implemented!')

    def apply(self, b: np.ndarray) -> np.ndarray:
        raise Exception('Apply not implemented!')


class DefaultSparseMatrixSample(MatrixSampleInterface):
    def __init__(self, mat: spla.spmatrix):
        self.matrix = mat

    def solve(self, b: np.ndarray):
        return spsolve(self.matrix, b)

    def apply(self, b: np.ndarray) -> np.ndarray:
        return self.matrix @ b

# Draws from the distribution of Ahat where Ahat x = b
class MatrixDistributionInterface:
    def draw_sample(self) -> MatrixSampleInterface:
        raise Exception('draw_sample not implemented!')

    def is_dual_distribution(self) -> bool:
        return False

# Draws from the joint distribution of (Ahat, Mhat) where Ahat x = Mhat b
class DualMatrixDistributionInterface(MatrixDistributionInterface):
    def draw_dual_sample(self) -> (MatrixSampleInterface, MatrixSampleInterface):
        raise Exception('draw_sample not implemented!')

    def draw_sample(self) -> MatrixSampleInterface:
        a, b = self.draw_dual_sample()
        return a

    # Returns true if Mhat is not the identity
    def is_dual_distribution(self) -> bool:
        return False


class IdentityMatrixSample(MatrixSampleInterface):
    def preprocess(self):
        pass

    def solve(self, b: np.ndarray) -> np.ndarray:
        return b

    def apply(self, b: np.ndarray) -> np.ndarray:
        return b


def soft_window_func_numerator(N, k):
    if k < N:
        return k
    elif k == N:
        return (k - 1.0) / 2.0
    else:
        return 0


def hard_window_func_numerator(N, k):
    if k <= N:
        return k
    else:
        return 0


def soft_window_func_denominator(N, k):
    if k < N:
        return k + 1
    elif k == N:
        return k / 2.0
    else:
        return 0


def hard_window_func_denominator(N, k):
    if k <= N:
        return k + 1
    else:
        return 0


def soft_shifted_window_func_numerator(N, k, alpha):
    if k <= N:
        return (k + 1) - sum((1 - 1 / alpha) ** (j - k) for j in range(k, N + 1))
    else:
        return 0


def hard_shifted_window_func_numerator(N, k, alpha):
    if k <= N:
        return (k + 1) - alpha
    else:
        return 0


def soft_shifted_window_func_denominator(N, k, alpha):
    if k <= N:
        return k + 1
    else:
        return 0


def hard_shifted_window_func_denominator(N, k, alpha):
    if k <= N:
        return k + 1
    else:
        return 0

# Implement baseline operator augmentation
def aug_fac(num_system_samples: int,
            num_per_system_samples: int,
            dimension: int,
            op_Ahat_inv,
            bootstrap_mat_dist: MatrixDistributionInterface,
            q_u_dist: VectorPairDistributionInterface = None,
            op_R = lambda x: x,
            op_B = lambda x: x):

    numerator = 0.0
    denominator = 0.0

    are_q_u_equal = True

    for i_system in range(0, num_system_samples):
        Ahat_bootstrap = bootstrap_mat_dist.draw_sample()
        Ahat_bootstrap.preprocess()

        for i_rhs in range(0, num_per_system_samples):
            if q_u_dist is None:
                q = np.random.randn(dimension)
                u = q
            else:
                q, u = q_u_dist.draw_sample()
                are_q_u_equal = q_u_dist.are_equal()

            a_boot_inv_q = Ahat_bootstrap.solve(q)
            a_inv_q = op_Ahat_inv(q)
            w_a_boot_inv_q = op_R(op_B(a_boot_inv_q))
            numerator += np.dot(w_a_boot_inv_q, a_boot_inv_q - a_inv_q)

            if are_q_u_equal:
                a_boot_inv_u = a_boot_inv_q
                wb_a_boot_inv_u = op_B(w_a_boot_inv_q)
            else:
                a_boot_inv_u = Ahat_bootstrap.solve(u)
                wb_a_boot_inv_u = op_R(op_B(op_B(a_boot_inv_u)))

            denominator += np.dot(wb_a_boot_inv_u, a_boot_inv_u)

    return max(numerator / denominator, 0.0)


def aug(num_system_samples: int,
        num_per_system_samples: int,
        rhs: np.ndarray,
        op_Ahat_inv,
        bootstrap_mat_dist: MatrixDistributionInterface,
        q_u_dist: VectorPairDistributionInterface = None,
        op_R = lambda x: x,
        op_B = lambda x: x):

    if (bootstrap_mat_dist.is_dual_distribution()):
        raise Exception("Dual distribution not supported by aug!")

    beta = aug_fac(num_system_samples,
                   num_per_system_samples,
                   len(rhs), op_Ahat_inv,
                   bootstrap_mat_dist,
                   q_u_dist, op_R, op_B)

    return pre_aug(beta, rhs, op_Ahat_inv, op_R, op_B)


def pre_aug(beta,
            rhs: np.ndarray,
            op_Ahat_inv,
            op_R = lambda x: x,
            op_B = lambda x: x):

    xhat = op_Ahat_inv(rhs)
    augmentation = beta * op_R(op_Ahat_inv(op_B(rhs)))
    return xhat - augmentation



# Implements energy norm operator augmentation
def en_aug_fac(num_system_samples: int,
               num_per_system_samples: int,
               dimension: int,
               op_Ahat,
               bootstrap_mat_dist: MatrixDistributionInterface,
               q_dist: VectorDistributionInterface = None):

    numerator = 0.0
    denominator = 0.0

    for i_system in range(0, num_system_samples):
        Ahat_bootstrap = bootstrap_mat_dist.draw_sample()
        Ahat_bootstrap.preprocess()

        for i_rhs in range(0, num_per_system_samples):

            if q_dist is None:
                q = np.random.randn(dimension)
            else:
                q = q_dist.draw_sample()

            Ahat_bootstrap_inv_q = Ahat_bootstrap.solve(q)
            term1 = np.dot(Ahat_bootstrap_inv_q, op_Ahat(Ahat_bootstrap_inv_q))
            term2 = np.dot(q, Ahat_bootstrap_inv_q)

            numerator += term1 - term2
            denominator += term1

    return max(numerator / denominator, 0.0)


def en_aug(num_system_samples: int,
           num_per_system_samples: int,
           rhs: np.ndarray,
           op_Ahat_inv,
           op_Ahat,
           bootstrap_mat_dist: MatrixDistributionInterface,
           q_dist: VectorDistributionInterface = None,
           op_C = lambda x: x):

    if (bootstrap_mat_dist.is_dual_distribution()):
        raise Exception("Dual distribution not supported by aug!")

    beta = en_aug_fac(num_system_samples,
                      num_per_system_samples,
                      len(rhs),
                      op_Ahat,
                      bootstrap_mat_dist,
                      q_dist)

    return pre_en_aug(beta, rhs, op_Ahat_inv, op_C)


def pre_en_aug(beta,
               rhs: np.ndarray,
               op_Ahat_inv,
               op_C = lambda x: x):
    xhat = op_Ahat_inv(rhs)
    augmentation = beta * op_Ahat_inv(op_C(rhs))
    return xhat - augmentation


# Implements energy norm operator augmentation
def en_aug_trunc_fac(num_system_samples: int,
                     num_per_system_samples: int,
                     dimension: int,
                     order: int,
                     op_Ahat_inv,
                     op_Ahat,
                     bootstrap_mat_dist: MatrixDistributionInterface,
                     q_dist: VectorDistributionInterface = None,
                     window_func_numerator=soft_window_func_numerator,
                     window_func_denominator=soft_window_func_denominator):

    numerator = 0.0
    denominator = 0.0

    for i_system in range(0, num_system_samples):
        Ahat_bootstrap = bootstrap_mat_dist.draw_sample()

        for i_rhs in range(0, num_per_system_samples):

            if q_dist is None:
                q = np.random.randn(dimension)
            else:
                q = q_dist.draw_sample()

            def op(x):
                return x - Ahat_bootstrap.apply(op_Ahat_inv(x))

            pows_q = [q]
            for i in range(1, order + 1):
                pows_q.append(op(pows_q[-1]))

            Ahat_inv_q = op_Ahat_inv(q)
            dots = [np.dot(Ahat_inv_q, pows_q[k]) for k in range(0, order + 1)]

            numerator += sum(window_func_numerator(order, k) * dots[k] for k in range(0, order + 1))
            denominator += sum(window_func_denominator(order, k) * dots[k] for k in range(0, order + 1))

    return numerator / denominator


def en_aug_trunc(num_system_samples: int,
                 num_per_system_samples: int,
                 rhs: np.ndarray,
                 order: int,
                 op_Ahat_inv,
                 op_Ahat,
                 bootstrap_mat_dist: MatrixDistributionInterface,
                 q_dist: VectorDistributionInterface = None,
                 op_C = lambda x: x,
                 window_func_numerator=soft_window_func_numerator,
                 window_func_denominator=soft_window_func_denominator):

    if (bootstrap_mat_dist.is_dual_distribution()):
        raise Exception("Dual distribution not supported by aug!")

    beta = en_aug_trunc_fac(num_system_samples,
                            num_per_system_samples,
                            len(rhs),
                            order,
                            op_Ahat_inv,
                            op_Ahat,
                            bootstrap_mat_dist,
                            q_dist,
                            window_func_numerator,
                            window_func_denominator)

    return pre_en_aug_trunc(beta, rhs, op_Ahat_inv, op_C)


def pre_en_aug_trunc(beta,
                     rhs: np.ndarray,
                     op_Ahat_inv,
                     op_C = lambda x: x):
    xhat = op_Ahat_inv(rhs)
    augmentation = beta * op_Ahat_inv(op_C(rhs))
    return xhat - augmentation


# Implements shifted energy norm operator augmentation
def en_aug_shift_trunc_fac(num_system_samples: int,
                           num_per_system_samples: int,
                           dimension: int,
                           order: int,
                           alpha,
                           op_Ahat_inv,
                           op_Ahat,
                           bootstrap_mat_dist: MatrixDistributionInterface,
                           q_dist: VectorDistributionInterface = None,
                           window_func_numerator=soft_shifted_window_func_numerator,
                           window_func_denominator=soft_shifted_window_func_denominator):

    numerator = 0.0
    denominator = 0.0

    for i_system in range(0, num_system_samples):
        Ahat_bootstrap = bootstrap_mat_dist.draw_sample()

        for i_rhs in range(0, num_per_system_samples):

            if q_dist is None:
                q = np.random.randn(dimension)
            else:
                q = q_dist.draw_sample()

            def op(x):
                return x - alpha * Ahat_bootstrap.apply(op_Ahat_inv(x))

            pows_q = [q]
            for i in range(1, order + 1):
                pows_q.append(op(pows_q[-1]))

            Ahat_inv_q = op_Ahat_inv(q)
            dots = [np.dot(Ahat_inv_q, pows_q[k]) for k in range(0, order + 1)]

            numerator += sum(alpha ** (-k - 2) * soft_shifted_window_func_numerator(order, k, alpha) *
                             dots[k] for k in range(0, order + 1))
            denominator += sum(alpha ** (-k - 2) * soft_shifted_window_func_denominator(order, k, alpha) *
                               dots[k] for k in range(0, order + 1))

    return min(max(numerator / denominator, 0.0), 1.0)


def en_aug_shift_trunc(num_system_samples: int,
                       num_per_system_samples: int,
                       rhs: np.ndarray,
                       order: int,
                       alpha,
                       op_Ahat_inv,
                       op_Ahat,
                       bootstrap_mat_dist: MatrixDistributionInterface,
                       q_dist: VectorDistributionInterface = None,
                       op_C = lambda x: x,
                       window_func_numerator=soft_shifted_window_func_numerator,
                       window_func_denominator=soft_shifted_window_func_denominator):

    if (bootstrap_mat_dist.is_dual_distribution()):
        raise Exception("Dual distribution not supported by aug!")

    beta = en_aug_shift_trunc_fac(num_system_samples,
                                  num_per_system_samples,
                                  len(rhs),
                                  order,
                                  alpha,
                                  op_Ahat_inv,
                                  op_Ahat,
                                  bootstrap_mat_dist,
                                  q_dist,
                                  window_func_numerator,
                                  window_func_denominator)

    return pre_en_aug_shift_trunc(beta, rhs, op_Ahat_inv, op_C)


def pre_en_aug_shift_trunc(beta,
                           rhs: np.ndarray,
                           op_Ahat_inv,
                           op_C = lambda x: x):
    xhat = op_Ahat_inv(rhs)
    augmentation = beta * op_Ahat_inv(op_C(rhs))
    return xhat - augmentation


def mod_pow_method(Ahat_bootstrap, op_Ahat_inv, eps, dimension):
    v_last = np.random.randn(dimension)
    a_inv_v_last = op_Ahat_inv(v_last)
    v = Ahat_bootstrap.apply(a_inv_v_last)
    a_inv_v = op_Ahat_inv(v)

    last_eig = np.dot(a_inv_v, v) / np.dot(a_inv_v_last, v_last)

    while True:
        v_last = v
        a_inv_v_last = a_inv_v

        v = Ahat_bootstrap.apply(a_inv_v_last)
        v = v / np.linalg.norm(v)
        a_inv_v = op_Ahat_inv(v)

        eig = np.dot(a_inv_v, v) / np.dot(a_inv_v_last, v_last)

        if abs(eig - last_eig) <= eps:
            break

        last_eig = eig

    return eig


# Implements accelerated shifted energy norm operator augmentation
def en_aug_accel_shift_trunc_fac(num_system_samples: int,
                                 num_per_system_samples: int,
                                 dimension: int,
                                 order: int,
                                 eps,
                                 op_Ahat_inv,
                                 op_Ahat,
                                 bootstrap_mat_dist: MatrixDistributionInterface,
                                 q_dist: VectorDistributionInterface = None,
                                 window_func_numerator=soft_shifted_window_func_numerator,
                                 window_func_denominator=soft_shifted_window_func_denominator):

    numerator = 0.0
    denominator = 0.0

    for i_system in range(0, num_system_samples):
        Ahat_bootstrap = bootstrap_mat_dist.draw_sample()

        alpha = mod_pow_method(Ahat_bootstrap, op_Ahat_inv, eps, dimension)

        for i_rhs in range(0, num_per_system_samples):

            if q_dist is None:
                q = np.random.randn(dimension)
            else:
                q = q_dist.draw_sample()

            def op(x):
                return x - alpha * Ahat_bootstrap.apply(op_Ahat_inv(x))

            pows_q = [q]
            for i in range(1, order + 1):
                pows_q.append(op(pows_q[-1]))

            Ahat_inv_q = op_Ahat_inv(q)
            dots = [np.dot(Ahat_inv_q, pows_q[k]) for k in range(0, order + 1)]

            numerator += sum(alpha ** (-k - 2) * window_func_numerator(order, k, alpha) *
                             dots[k] for k in range(0, order + 1))
            denominator += sum(alpha ** (-k - 2) * window_func_denominator(order, k, alpha) *
                               dots[k] for k in range(0, order + 1))

    return min(max(numerator / denominator, 0.0), 1.0)


def en_aug_accel_shift_trunc(num_system_samples: int,
                             num_per_system_samples: int,
                             rhs: np.ndarray,
                             order: int,
                             op_Ahat_inv,
                             op_Ahat,
                             bootstrap_mat_dist: MatrixDistributionInterface,
                             eps=0.01,
                             q_dist: VectorDistributionInterface = None,
                             op_C = lambda x: x,
                             window_func_numerator=soft_shifted_window_func_numerator,
                             window_func_denominator=soft_shifted_window_func_denominator):

    if (bootstrap_mat_dist.is_dual_distribution()):
        raise Exception("Dual distribution not supported by aug!")

    beta = en_aug_accel_shift_trunc_fac(num_system_samples,
                                        num_per_system_samples,
                                        len(rhs),
                                        order,
                                        eps,
                                        op_Ahat_inv,
                                        op_Ahat,
                                        bootstrap_mat_dist,
                                        q_dist,
                                        window_func_numerator,
                                        window_func_denominator)

    return pre_en_aug_accel_shift_trunc(beta, rhs, op_Ahat_inv, op_C)


def pre_en_aug_accel_shift_trunc(beta,
                                 rhs: np.ndarray,
                                 op_Ahat_inv,
                                 op_C = lambda x: x):
    xhat = op_Ahat_inv(rhs)
    augmentation = beta * op_Ahat_inv(op_C(rhs))
    return xhat - augmentation