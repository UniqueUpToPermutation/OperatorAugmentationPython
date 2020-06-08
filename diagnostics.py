import augmentation as aug
import numpy as np
from tqdm import tqdm
import threading

class MatrixParameterDistribution(aug.MatrixDistributionInterface):
    def __init__(self, matrix_parameters, fixed_hyper_parameters):
        self.matrix_parameters = matrix_parameters
        self.fixed_hyper_parameters = fixed_hyper_parameters

    def draw_parameters(self):
        raise Exception('draw not implemented!')

    # Sample only the A matrix for Ax = b
    def draw_sample(self) -> aug.MatrixSampleInterface:
        return self.convert(self.draw_parameters())

    # Sample both the A and M matrices in Ax = Mb
    def draw_dual_sample(self) -> (aug.MatrixSampleInterface, aug.MatrixSampleInterface):
        params = self.draw_parameters()
        return self.convert(params), self.convert_auxiliary(params)

    # Convert matrix parameters into an actual matrix
    def convert(self, matrix_parameters) -> aug.MatrixSampleInterface:
        raise Exception('convert not implemented!')

    # Convert matrix parameters into auxilliary matrix M
    def convert_auxiliary(self, matrix_parameters) -> aug.MatrixSampleInterface:
        return aug.IdentityMatrixSample()

    def get_dimension(self) -> int:
        raise Exception('get_dimension not implemented!')


class ProblemDefinition:
    def __init__(self, tru_distribution: MatrixParameterDistribution):
        self.tru_distribution = tru_distribution
        self.b_distribution = aug.VectorDistributionFromLambda(
            lambda: np.random.randn(self.tru_distribution.get_dimension()))

        self.tru_mat = self.tru_distribution.convert(self.tru_distribution.matrix_parameters)
        self.tru_aux = self.tru_distribution.convert_auxiliary(self.tru_distribution.matrix_parameters)
        self.energy_norm_squared = lambda x: np.sqrt(np.dot(x, self.tru_mat.apply(x)))
        self.l2_norm_squared = lambda x: np.sqrt(np.dot(x, x))


class ProblemRun:
    def __init__(self, parent: ProblemDefinition, name):
        self.parent = parent
        self.q_distribution = self.parent.b_distribution
        self.q_u_distribution = aug.IdenticalVectorPairDistributionFromLambda(self.q_distribution.draw_sample)
        self.num_sub_runs = 10
        self.samples_per_sub_run = 10
        self.samples_per_system = 1
        self.name = name
        self.err_arrays = []
        self.norms = []
        self.norm_names = []
        self.norms.append(parent.l2_norm_squared)
        self.norms.append(parent.energy_norm_squared)
        self.norm_names.append('L2 norm')
        self.norm_names.append('Energy norm')

    def sub_run(self, bootstrap_distribution: MatrixParameterDistribution,
                b: np.ndarray) -> np.ndarray:
        raise Exception('sub_run not implemented!')


# Do nothing besides solve the noisy system
class NaiveProblemRun(ProblemRun):
    def __init__(self, parent: ProblemDefinition):
        name = 'Naive'
        ProblemRun.__init__(self, parent, name)

    def sub_run(self, bootstrap_distribution: MatrixParameterDistribution,
                b: np.ndarray):
        sampled_mat = bootstrap_distribution.convert(bootstrap_distribution.matrix_parameters)
        sampled_mat.preprocess()
        aux_mat = bootstrap_distribution.convert_auxiliary(bootstrap_distribution.matrix_parameters)
        aux_mat.preprocess()
        return sampled_mat.solve(aux_mat.apply(b))


class AugProblemRun(ProblemRun):
    def __init__(self, parent: ProblemDefinition, op_R = lambda x: x, op_B = lambda x: x):
        name = 'Augmentation'
        ProblemRun.__init__(self, parent, name)
        self.op_B = op_B
        self.op_R = op_R

    def sub_run(self, bootstrap_distribution: MatrixParameterDistribution,
                b: np.ndarray):
        sampled_mat = bootstrap_distribution.convert(bootstrap_distribution.matrix_parameters)
        sampled_mat.preprocess()
        op_Ahat_inv = lambda x: sampled_mat.solve(x)
        return aug.aug_sym(self.samples_per_sub_run, self.samples_per_system, b,
                           op_Ahat_inv, bootstrap_distribution, self.q_u_distribution, self.op_R, self.op_B)


class EnAugProblemRun(ProblemRun):
    def __init__(self, parent: ProblemDefinition, op_C = lambda x: x):
        name = 'Energy-Norm Augmentation'
        ProblemRun.__init__(self, parent, name)
        self.op_C = op_C

    def sub_run(self, bootstrap_distribution: MatrixParameterDistribution,
                b: np.ndarray):
        sampled_mat = bootstrap_distribution.convert(bootstrap_distribution.matrix_parameters)
        sampled_mat.preprocess()
        op_Ahat_inv = lambda x: sampled_mat.solve(x)
        op_Ahat = lambda x: sampled_mat.apply(x)
        return aug.en_aug(self.samples_per_sub_run, self.samples_per_system, b,
                          op_Ahat_inv, op_Ahat, bootstrap_distribution, self.q_distribution, self.op_C)


class TruncEnAugProblemRun(ProblemRun):
    def __init__(self, parent: ProblemDefinition, order, window_funcs='default', op_C = lambda x: x):
        if window_funcs == 'default':
            name = f'Trunc. En-Norm Augmentation (Order {order})'
        else:
            name = f'Trunc. En-Norm Augmentation (Order {order}, {window_funcs.capitalize()} Window)'
        self.order = order
        self.op_C = op_C
        self.window_funcs = window_funcs
        ProblemRun.__init__(self, parent, name)

    def sub_run(self, bootstrap_distribution: MatrixParameterDistribution,
                b: np.ndarray):
        sampled_mat = bootstrap_distribution.convert(bootstrap_distribution.matrix_parameters)
        sampled_mat.preprocess()
        op_Ahat_inv = lambda x: sampled_mat.solve(x)
        op_Ahat = lambda x: sampled_mat.apply(x)
        if self.window_funcs == 'hard':
            return aug.en_aug_trunc(self.samples_per_sub_run, self.samples_per_system, b, self.order,
                                    op_Ahat_inv, op_Ahat, bootstrap_distribution, self.q_distribution, self.op_C,
                                    aug.hard_window_func_numerator,
                                    aug.hard_window_func_denominator)
        else:
            return aug.en_aug_trunc(self.samples_per_sub_run, self.samples_per_system, b, self.order,
                                    op_Ahat_inv, op_Ahat, bootstrap_distribution, self.q_distribution, self.op_C)


class TruncEnAugShiftedProblemRun(ProblemRun):
    def __init__(self, parent: ProblemDefinition, order, alpha, window_funcs='default', op_C=lambda x: x):
        if window_funcs == 'default':
            name = f'Shifted Trunc. En-Norm Augmentation (Order {order})'
        else:
            name = f'Shifted Trunc. En-Norm Augmentation (Order {order}, {window_funcs.capitalize()} Window)'
        self.order = order
        self.op_C = op_C
        self.alpha = alpha
        self.window_funcs = window_funcs
        ProblemRun.__init__(self, parent, name)

    def sub_run(self, bootstrap_distribution: MatrixParameterDistribution,
                b: np.ndarray):
        sampled_mat = bootstrap_distribution.convert(bootstrap_distribution.matrix_parameters)
        sampled_mat.preprocess()
        op_Ahat_inv = lambda x: sampled_mat.solve(x)
        op_Ahat = lambda x: sampled_mat.apply(x)
        if self.window_funcs == 'hard':
            return aug.en_aug_shift_trunc(self.samples_per_sub_run, self.samples_per_system,
                                          b, self.order, self.alpha, op_Ahat_inv, op_Ahat,
                                          bootstrap_distribution,
                                          self.q_distribution, self.op_C,
                                          aug.hard_shifted_window_func_numerator,
                                          aug.hard_shifted_window_func_denominator)

        else:
            return aug.en_aug_shift_trunc(self.samples_per_sub_run, self.samples_per_system,
                                          b, self.order, self.alpha, op_Ahat_inv, op_Ahat,
                                          bootstrap_distribution,
                                          self.q_distribution, self.op_C)


class TruncEnAugAccelProblemRun(ProblemRun):
    def __init__(self, parent: ProblemDefinition, order, window_funcs='default', op_C = lambda x: x):
        if window_funcs == 'default':
            name = f'Accel. Shifted Trunc. En-Norm Augmentation (Order {order})'
        else:
            name = f'Accel. Shifted Trunc. En-Norm Augmentation (Order {order}, {window_funcs.capitalize()} Window)'
        self.order = order
        self.op_C = op_C
        self.eps = 0.01
        self.window_funcs = window_funcs
        ProblemRun.__init__(self, parent, name)

    def sub_run(self, bootstrap_distribution: MatrixParameterDistribution,
                b: np.ndarray):
        sampled_mat = bootstrap_distribution.convert(bootstrap_distribution.matrix_parameters)
        sampled_mat.preprocess()
        op_Ahat_inv = lambda x: sampled_mat.solve(x)
        op_Ahat = lambda x: sampled_mat.apply(x)
        if self.window_funcs == 'hard':
            return aug.en_aug_accel_shift_trunc(self.samples_per_sub_run, self.samples_per_system,
                                                b, self.order, op_Ahat_inv, op_Ahat,
                                                bootstrap_distribution, self.eps,
                                                self.q_distribution, self.op_C,
                                                aug.hard_shifted_window_func_numerator,
                                                aug.hard_shifted_window_func_denominator)
        else:
            return aug.en_aug_accel_shift_trunc(self.samples_per_sub_run, self.samples_per_system,
                                                b, self.order, op_Ahat_inv, op_Ahat,
                                                bootstrap_distribution, self.eps,
                                                self.q_distribution, self.op_C)


class ProblemRunResults:
    def __init__(self, run: ProblemRun, raw_err_data: np.ndarray, solution_norms: np.ndarray):
        self.name = run.name
        self.norm_names = run.norm_names
        self.raw_err_data = raw_err_data
        self.mean_errs = np.mean(raw_err_data, axis=1)
        self.std_errs = np.std(raw_err_data, axis=1) / np.sqrt(run.num_sub_runs)
        self.mean_sol_norms = np.mean(solution_norms, axis=1)
        self.relative_errs = self.mean_errs / self.mean_sol_norms
        self.relative_std_errs = self.std_errs / self.mean_sol_norms

    def print(self):
        for i in range(0, len(self.norm_names)):
            print(f'{self.name}: {self.norm_names[i]}: {self.mean_errs[i]:#.4g} +- {2 * self.std_errs[i]:#.4g}')
            print(f'{self.name}: {self.norm_names[i]} (Relative): {self.relative_errs[i]:#.4g} +-'
                  f' {2 * self.relative_std_errs[i]:#.4g}')



class DiagnosticThreadWorker:
    def __init__(self, tru_distribution: MatrixParameterDistribution,
                 parent_run: ProblemRun, problem: ProblemDefinition, num_sub_runs: int,
                 thread_id: int):
        self.tru_distribution = tru_distribution
        self.num_sub_runs = num_sub_runs
        self.parent_run = parent_run
        self.problem = problem
        self.thread_id = thread_id
        self.errs = np.zeros((len(self.parent_run.norms), num_sub_runs))
        self.sol_norms = np.zeros((len(self.parent_run.norms), num_sub_runs))

    def run(self):
        tru_mat = self.problem.tru_mat
        tru_distribution = self.problem.tru_distribution
        tru_aux = self.problem.tru_aux

        if self.thread_id == 0:
            iter = tqdm(range(0, self.num_sub_runs), desc=self.parent_run.name)
        else:
            iter = range(0, self.num_sub_runs)

        for i in iter:
            # Draw rhs from Bayes prior distribution
            b = self.problem.b_distribution.draw_sample()

            # Solve the true linear system b
            tru_solution = tru_mat.solve(tru_aux.apply(b))

            # Perform bootstrap operator augmentation
            noisy_parameters = self.problem.tru_distribution.draw_parameters()
            dist_type = self.problem.tru_distribution.__class__
            bootstrap_distribution = dist_type(noisy_parameters, tru_distribution.fixed_hyper_parameters)
            result = self.parent_run.sub_run(bootstrap_distribution, b)

            # Get error from dis
            for i_norm in range(0, len(self.parent_run.norms)):
                self.errs[i_norm, i] = self.parent_run.norms[i_norm](result - tru_solution)
                self.sol_norms[i_norm, i] = self.parent_run.norms[i_norm](tru_solution)


class DiagnosticRun:
    def __init__(self, problem: ProblemDefinition):
        self.runs = []
        self.results = []
        self.problem = problem

    def add_run(self, run):
        self.runs.append(run)

    def run(self, num_threads=1):
        # Preprocess true solution if necessary
        tru_mat = self.problem.tru_mat
        tru_distribution = self.problem.tru_distribution
        tru_aux = self.problem.tru_aux
        tru_mat.preprocess()

        for run in self.runs:
            sub_runs_per_thread = int(run.num_sub_runs / num_threads)
            total_sub_runs = sub_runs_per_thread * num_threads
            workers = [DiagnosticThreadWorker(tru_distribution, run, self.problem, sub_runs_per_thread, i)
                       for i in range(0, num_threads)]

            # Break up work among the threads
            if num_threads == 1:
                workers[0].run()
            else:
                # Launch threads
                worker_threads = [threading.Thread(target=DiagnosticThreadWorker.run, args=(w,), daemon=True)
                                  for w in workers]
                for t in worker_threads:
                    t.start()

                # Wait to finish
                for t in worker_threads:
                    t.join()

            # Concatenate results
            errs = np.hstack([w.errs for w in workers])
            sol_norms = np.hstack([w.sol_norms for w in workers])

            self.results.append(ProblemRunResults(run, errs, sol_norms))

    def print_results(self):
        for result in self.results:
            result.print()
            print('')


