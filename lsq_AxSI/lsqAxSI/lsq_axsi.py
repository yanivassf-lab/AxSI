import logging
from typing import Optional, Any, Callable, Tuple

import numpy as np
from numpy.linalg import norm

from lsqAxSI.default_values import reg_func, jac_calc, X0, MIN_VAL, MAX_VAL
from lsqAxSI.lsq_components import LSQResult, LSQOptions, LSQVars, LSQValues
from lsqAxSI.lsq_trust_region import Trust_region

logger = logging.getLogger(__name__)

EPS = np.finfo('float').eps

"""
Methods:
-------
__init__(reg_func, xstart, user_options, jacob, *varargin):
    Initializes the LSQ optimizer with a residual function, initial guess, options, and Jacobian function.
set_boundaries(lower_bound, upper_bound):
    Sets the lower and upper bounds for the optimization variables.
assert_x_in_bounds():
    Ensures that the current solution is within the specified bounds.
main_loop(jac, *varargin):
    Executes the main optimization loop until convergence or iteration limits are reached.
perform_iteration(jac, *varargin):
    Performs a single iteration of the optimization process, including trust-region updates
    and state advancement.
test_trial_point(new_val):
    Tests whether the proposed trial point satisfies the trust-region acceptance criteria.
determine_trust_region_correction():
    Solves the trust-region subproblem to compute the next step direction and size.
update_delta(new_val):
    Dynamically adjusts the trust-region radius based on the improvement ratio.
advance_to_next_iteration(new_values):
    Updates the optimization state using the evaluated values at the new trial point.
prepare_output():
    Prepares the final output object with optimized parameters, Jacobian, and optimization metadata.
"""


class LSQ:
    """
    Implements a nonlinear least-squares optimization algorithm using a trust-region method.

    The class handles iterative optimization of a set of parameters to minimize the sum of squared residuals,
    incorporating bound constraints, trust-region adjustments, and convergence criteria.

    Attributes:
        func (Callable): The residual function to be minimized. It computes the difference between observed and modeled
                         data given a parameter vector. output dim: (n_frames,)
        options (LSQOptions): Contains user-defined and default options for the optimization process,
                              such as tolerances, maximum iterations, and solver parameters.
        values (LSQValues): Stores the current state of the optimization, including parameter estimates,
                            residuals, Jacobian matrix, gradient, and objective function value.
        vars (LSQVars): Tracks the optimization's iterative variables, such as trust-region radius,
                        proximity to bounds, and iteration counts.
        result (LSQResult): Stores the final outcome of the optimization, including the solution, residuals,
                            and termination status.
        tregion (Trust_region): Handles trust-region calculations for adjusting the step size and direction
                                during the optimization process.
        """

    def __init__(self, reg_func: Callable, xstart: np.ndarray, user_options: dict, jacob: Callable,
                 *varargin: Any) -> None:
        """
        Initialize LSQ for least-squares optimization.

        :param reg_func: Function for the residuals, takes model parameters and data, returns errors. output dim: (n_frames,)
        :param xstart: Initial guess for the model parameters.
        :param user_options: Dictionary of user-specified options for the optimization.
        :param jacob: Jacobian function for the residuals.
        :param varargin: Additional arguments to pass to reg_func and jacob.
        :param tregion (float): Represents the radius of the trust region. This
            dynamically adjusts during the optimization process to balance
            exploration and convergence:
                - If the step improves the objective sufficiently, the trust
                  region may expand.
                - If the step does not improve the objective, the trust
                  region shrinks.
            Ensures the step size stays within bounds for stability and
            convergence efficiency.
        """
        self.func = reg_func
        # TODO: Refael: we can run it one time for all voxels
        self.options = LSQOptions(user_options, xstart)
        self.values = LSQValues(reg_func, xstart, jacob, *varargin)
        self.vars = LSQVars(self.values, self.options)
        self.vars.increment_param("numFunEvals", 3)
        self.result = LSQResult(xstart, self.values.fvec, self.values.A)
        self.tregion = None

    def set_boundaries(self, lower_bound, upper_bound):
        self.vars.set_lower_bound(lower_bound)
        self.vars.set_upper_bound(upper_bound)
        self.vars.assert_bounds()
        self.assert_x_in_bounds()

    def assert_x_in_bounds(self):
        x = self.values.x
        ub = self.vars.ub
        lb = self.vars.lb
        if min(min(ub - x), min(x - lb)) < 0:
            self.values.x = self.vars.compute_startx()

    def main_loop(self, jac, *varargin) -> None:
        """
        Executes the main optimization loop until convergence or iteration limits are reached.

        Parameters
        ----------
        jac: Jacobian function for the residuals.
        varargin: Additional arguments to pass to reg_func and jacob.
        """
        while not self.vars.ex:  # Exit flag status
            self.vars.evaluate_dnewt(self.values.fvec)
            self.update_iteration_variables()
            self.test_for_convergence()
            if not self.vars.ex:
                self.perform_iteration(jac, *varargin)
            self.check_iterations_limit()

    def perform_iteration(self, jac, *varargin) -> None:
        """
        Performs a single iteration of the optimization process, including trust-region updates and state advancement.

        Parameters
        ----------
        jac: Jacobian function for the residuals.
        varargin: Additional arguments to pass to reg_func and jacob.
        """
        reg_func = self.func
        # Computes the next candidate solution x within the trust-region radius
        newx = self.determine_trust_region_correction()
        # Applies a small perturbation to the computed candidate solution to prevent numerical instabilities
        newx = self.perturb(newx)  # TODO: Refael: continue to learn from here
        # This evaluates the residuals, the Jacobian matrix, and the objective function value at the new x
        new_values = LSQValues(reg_func, newx, jac, *varargin)
        # Increments the counter for the number of function evaluations (numFunEvals) by 3 (tracks computational cost).
        self.vars.increment_param("numFunEvals", 3)
        # Determines if the new solution leads to sufficient improvement in the objective function.
        self.test_trial_point(new_values.val)
        # Updates the optimization state using the evaluated values at new x (if accepted)
        self.advance_to_next_iteration(new_values)

    def test_trial_point(self, new_val):
        self.update_ratio(new_val)
        self.update_delta(new_val)

    def update_ratio(self, new_val):
        ss = self.tregion.ss
        dv = self.vars.dv
        g = self.values.grad
        val = self.values.val
        qp = self.tregion.qpval
        aug = 0.5 * np.dot(ss.T, (dv * abs(g)) * ss)
        self.vars.ratio = (0.5 * (new_val - val) + aug) / qp

    def update_delta(self, new_val):
        delta = self.vars.delta
        ratio = self.vars.ratio
        nrmsx = self.vars.nrmsx

        if ratio >= 0.75 and nrmsx >= 0.9 * delta:
            self.vars.delta = min(self.vars.delbnd, 2 * delta)
        elif ratio <= 0.25:
            self.vars.delta = min(nrmsx / 4, delta / 4)
        if np.isinf(new_val):
            self.vars.delta = min(nrmsx / 20, delta / 20)

    def advance_to_next_iteration(self, new_values):
        if new_values.val < self.values.val:
            self.values = new_values
        self.vars.it = self.vars.it + 1
        self.vars.vval[self.vars.it] = self.values.val

    def perturb(self, x, delta=100 * EPS):
        u = self.vars.ub
        l = self.vars.lb
        if (min(abs(u - x)) < delta) or (min(abs(x - l)) < delta):
            upperi = (u - x) < delta
            loweri = (x - l) < delta
            x[upperi] = x[upperi] - delta
            x[loweri] = x[loweri] + delta
        return x

    def update_iteration_variables(self):
        """
        Determines the proximity of the current solution x to its lower and upper bounds.
        Adjusts the gradient based on proximity to bounds (focusing optimization effort on those that are more constrained).
        Tracks the maximum adjusted gradient value for convergence monitoring.
        """
        self.vars.define_dv_and_v(self.values.x, self.values.grad)
        gopt = self.vars.v * self.values.grad
        self.vars.voptnrm[self.vars.it] = norm(gopt, np.inf)

    def test_for_convergence(self):
        """
        Check whether the optimization process has converged based on step size,
        trust-region radius, improvement ratio, and change in the objective function.

        Convergence Criteria:
        1. Trust-region and improvement-based:
           - Step size is small relative to the trust-region radius.
           - Improvement ratio is sufficient (> 0.25).
           - Change in the objective function is below a specified threshold.
        2. Step size-based:
           - The step size is below a strict tolerance.

        Updates:
        - Sets `self.vars.ex` to 1 or 2 to indicate the reason for convergence.
        - Calls `self.result.successful_termination()` if convergence is achieved.
        """

        nrmsx = self.vars.nrmsx
        delta = self.vars.delta
        ratio = self.vars.ratio
        tol1 = self.options.tol1
        val = self.values.val
        it = self.vars.it
        if it > 0:
            # vval[it-2] - Objective function value from the previous iteration, used to compute the change in the objective function
            diff = abs(self.vars.vval[it - 1] - val)
            if (nrmsx < 0.9 * delta) & (ratio > 0.25) & (diff < tol1 * (1 + abs(val))):
                self.result.successful_termination()
                self.vars.ex = 1
            elif (nrmsx < self.options.tol2):
                self.result.successful_termination()
                self.vars.ex = 2

    def check_iterations_limit(self):
        if self.vars.it > self.options.itb:
            self.max_iterations_exceeded()
        if self.vars.numFunEvals > self.options.maxfunevals:
            self.max_iterations_exceeded()

    def max_iterations_exceeded(self):
        self.vars.ex = 4
        self.vars.it = self.vars.it - 1
        self.result.iterations_exceeded()

    def determine_trust_region_correction(self) -> np.ndarray:
        """ Solves the trust-region subproblem to compute the next step direction and size.

        theta controls the size of the trust region: If the optimization is closer to convergence (smaller scaled
        gradient norm - optnorm) theta approaches 0.95, limiting the step size. Otherwise, theta decreases, allowing
        larger exploratory steps.
        """
        iter = self.vars.it
        optnorm = self.vars.voptnrm[iter]
        theta = max(0.95, 1 - optnorm)
        self.tregion = self.trust_region_trial_step(theta)  # computes the optimal step s within the trust region.
        # Updates the current parameter vector x using the computed step s and returns the new candidate solution.
        newx = self.update_tregion_values()  # TODO: Refael: continue to learn from here
        return newx

    def trust_region_trial_step(self, theta):
        x = self.values.x
        delta = self.vars.delta
        options = self.options

        tregion = Trust_region(self.values, self.vars)
        tregion.trial_step(x, theta, options.kmax, options.pcflags, options.pcgtol, delta)
        return tregion

    def update_tregion_values(self):
        iter = self.vars.it
        posdef = self.tregion.posdef
        if posdef == 0:
            posdef = self.vars.vpos[iter]
        self.vars.vpos[iter + 1] = posdef
        self.vars.nrmsx = norm(self.tregion.ss)
        self.vars.vpcg[iter + 1] = self.tregion.pcgit
        newx = self.values.x + self.tregion.s
        return newx

    def prepare_output(self):
        self.result.jacob = self.values.A
        self.set_output_values()
        self.result.x = self.values.x
        self.result.fvec = self.values.fvec

    def set_output_values(self):
        it = self.vars.it
        output = self.result.output
        output['firstorderopt'] = self.vars.voptnrm[it]
        output['iterations'] = it
        output['funcCount'] = self.vars.numFunEvals
        output['cgiterations'] = np.sum(self.vars.vpcg)


def snls(funfcn: Callable, xstart: np.ndarray, l: np.ndarray, u: np.ndarray, options: dict,
         jac: Callable, *varargin: Any) -> LSQResult:
    """
    Solves a nonlinear least squares problem with bound constraints.

    :param funfcn: The objective function to minimize.
    :param xstart: Initial guess for the solution. Shape: (n_params,), in our case (2,).
    :param l: Lower bounds for the parameters. Shape: (n_params,), in our case (2,).
    :param u: Upper bounds for the parameters. Shape: (n_params,), in our case (2,).
    :param options: Optimization options such as tolerances and maximum iterations.
    :param jac: Function to compute the Jacobian matrix of the objective function.
    :param varargin: Additional arguments passed to `funfcn` and `jac`.
    :return: LSQResult object with the result of the optimization process.
    """
    lsq_component = LSQ(funfcn, xstart, options, jac, *varargin)
    lsq_component.set_boundaries(l, u)
    lsq_component.main_loop(jac, *varargin)
    lsq_component.prepare_output()

    return lsq_component.result


def nonlinear_least_squares(reg_func: Callable = reg_func, x0: np.ndarray = X0,
                            bounds: Tuple[np.ndarray, np.ndarray] = (MIN_VAL, MAX_VAL), jac: Callable = jac_calc,
                            ftol: float = 1e-6, xtol: float = 1e-6, diff_step: float = 1e-3,
                            max_nfev: int = 20000, args: Optional[Tuple[Any, ...]] = None) -> LSQResult:
    """
    Perform nonlinear least-squares optimization using a customized solver.

    This function minimizes the sum of squares of residuals defined by the
    provided regression function `reg_func`.

    Parameters:
    ----------
    reg_func : callable
        The regression function that computes the residuals (for each frame). It must have the signature:
        `reg_func(x, *args)` where `x` is the parameter vector to optimize, and
        `*args` are additional arguments.

    x0 : array-like
        Initial guess for the parameters to be optimized. The dimension of
        `x0` corresponds to the number of parameters in the model (2 paramters in the default case).

    bounds : tuple of array-like, optional
        Lower and upper bounds for each parameter in `x0`.
        Format: `(lower_bounds, upper_bounds)`, where both are arrays of the
        same dimension as `x0`. Default is `None`, meaning no bounds are imposed.

    jac : callable, optional
        A function to compute the Jacobian matrix of the residuals. If `None`,
        the Jacobian is approximated using finite differences. The Jacobian function
        must have the signature `jac(x, *args)` and return a 2D array of size
        `(n_frames, 2)`.

    ftol : float, optional
        Tolerance for the cost function value. Iterations will stop when the
        relative change in the cost function is less than `ftol`. Default is `1e-6`.

    xtol : float, optional
        Tolerance for parameter updates. Iterations will stop when the relative
        change in the parameter vector is less than `xtol`. Default is `1e-6`.

    diff_step : float, optional
        Step size for finite-difference approximation of the Jacobian if
        `jac` is not provided. Default is `1e-3`.

    max_nfev : int, optional
        Maximum number of function evaluations before terminating the optimization.
        Default is `20,000`.

    args : tuple, optional
        Additional arguments to pass to the `reg_func` and `jac`. These are passed
        as `*args`.

    Returns:
    -------
    result : LSQResult object
        An object containing the optimized parameter vector, cost function value,
        Jacobian matrix, and additional optimization details.

    Notes:
    ------
    This function is a wrapper for a custom solver, which uses trust-region methods
    for robust nonlinear least-squares optimization. It extends standard least-squares
    techniques with additional flexibility and control over parameters.
    """

    # xstart = np.array(x0, dtype='float')   # Refael: it is already np.array
    # l = np.array(bounds[0], dtype='float') # Refael: it is already np.array
    # u = np.array(bounds[1], dtype='float') # Refael: it is already np.array

    options = {'TolFun': ftol, 'TolX': xtol, 'MaxFunEvals': max_nfev, 'diff_step': diff_step}
    varargin = []
    for i in range(len(args) - 1):
        varargin.append(np.array(args[i]).flatten())
    varargin.append(args[-1])

    # result = snls(reg_func, xstart, l, u, options, jac, *varargin) # Refael
    result = snls(reg_func, x0, bounds[0], bounds[1], options, jac, *varargin)  # Refael
    return result
