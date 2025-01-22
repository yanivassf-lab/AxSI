import logging
from typing import Any, Callable, Tuple

import numpy as np
from numpy.linalg import norm

from lsqAxSI.matrix_operations import atamult

logger = logging.getLogger(__name__)

EPS = np.finfo('float').eps
EMPTY_ARRAY = np.array([])
INF = np.inf
defaultopt = {
    'ActiveConstrTol': np.sqrt(EPS),
    'PrecondBandWidth': INF,
    'tolPCG': 0.1,
    'MaxPCGIter': 'max(1,floor(numberofvariables/2))',
    "TypicalX": 'ones(numberofvariables,1)',
    "maxfunevals": '100*numberofvariables',
    'MaxIter': 20000
}


class LSQResult:
    """
    Represents the result of a least-squares optimization process.

    Attributes:
        x (np.ndarray): The final optimized parameter vector.
        fvec (np.ndarray): The residual vector at the solution.
        jacob (np.ndarray): The Jacobian matrix at the solution.
        exitflag (int): Status flag indicating success or failure (1 for success, 0 for failure).
        output (Dict[str, str]): Dictionary containing details about the optimization algorithm.
        msg (str): A message describing the result or status of the optimization.
    """

    def __init__(self, xstart: np.ndarray, fvec: np.ndarray, jacob: np.ndarray) -> None:
        """
        Initializes an LSQResult instance.

        Args:
            xstart (np.ndarray): Initial parameter vector.
            fvec (np.ndarray): Residual vector.
            jacob (np.ndarray): Jacobian matrix.
        """
        self.x = xstart
        self.fvec = fvec
        self.jacob = jacob
        self.exitflag = 0
        self.output = {'algorithm': 'trust-region-reflective'}
        self.msg = ""

    def successful_termination(self):
        self.exitflag = 1
        self.msg = "Optimization terminated successfully"

    def iterations_exceeded(self):
        self.exitflag = 0
        self.msg = "Maximum number of iterations exceeded"


class LSQOptions():
    """
    Class for managing and validating the options for the least squares optimization.
    This includes user-specified options and default options for the solver.

    Attributes:
        user_options (dict): Dictionary of user-provided options.
        default_options (dict): Default options for the solver.
        active_tol (float): Tolerance for active constraints.
        gradflag (bool): Whether the Jacobian is active.
        typx (np.ndarray): Typical values for the optimization variables.
        pcflags (float): Preconditioning bandwidth flag.
        tol2 (float): Tolerance for x values.
        tol1 (float): Tolerance for function values.
        itb (int): Maximum number of iterations for the solver.
        maxfunevals (int): Maximum number of function evaluations.
        pcgtol (float): Tolerance for preconditioned conjugate gradient.
        kmax (int): Maximum iterations for preconditioned conjugate gradient.
        numberOfVariables (int): Number of variables to optimize.
    """

    def __init__(self, user_options: dict, xstart: np.ndarray):
        """
        Initialize LSQOptions with user-specified options and default settings.

        Args:
            user_options (dict): A dictionary of user-defined options.
            xstart (np.ndarray): Initial guess for the optimization variables.
        """

        self.user_options = user_options
        self.default_options = defaultopt
        # Refael. we can do it simpler (do it for all keys) - can delete self.optimget method:
        # TODO: test it
        self.default_options.update(user_options)
        self.active_tol = self.default_options.get('ActiveConstrTol')  # get value or None if not exists
        self.gradflag = self.default_options.get('Jacobian') == 'on'
        self.typx = self.default_options.get('TypicalX')
        self.pcflags = self.default_options.get('PrecondBandWidth')
        self.tol2 = self.default_options.get('TolX')
        self.tol1 = self.default_options.get('TolFun')
        self.itb = self.default_options.get('MaxIter')
        self.maxfunevals = self.default_options.get('MaxFunEvals')
        self.pcgtol = self.default_options.get('tolPCG')
        self.kmax = self.default_options.get('MaxPCGIter')
        # self.active_tol = self.optimget('ActiveConstrTol')
        # self.gradflag = self.optimget('Jacobian') == 'on'
        # self.typx = self.optimget('TypicalX')
        # self.pcflags = self.optimget('PrecondBandWidth')
        # self.tol2 = self.optimget('TolX')
        # self.tol1 = self.optimget('TolFun')
        # self.itb = self.optimget('MaxIter')
        # self.maxfunevals = self.optimget('MaxFunEvals')
        # self.pcgtol = self.optimget('tolPCG')
        # self.kmax = self.optimget('MaxPCGIter')
        self.numberOfVariables = xstart.shape[0]
        self.assert_options()

    # Refael - delete it
    # def optimget(self, key:str, default_value=None) -> Any:
    #     if key in self.user_options.keys():
    #         return self.user_options[key]
    #     if key in self.default_options.keys():
    #         return self.default_options[key]
    #     return default_value

    def assert_options(self):
        """Validate that the options are correct and raise exception if necessary."""
        if self.numberOfVariables == 0:
            # TODO: Refael: need to through exception
            print("Warning: Number of variables must be positive")
        if self.pcgtol <= 0:
            self.pcgtol = 0.1
        self.assert_attribute("typx", 'ones(numberofvariables,1)', np.ones(self.numberOfVariables))
        self.assert_attribute("kmax", 'max(1,floor(numberofvariables/2))', max(1, self.numberOfVariables // 2))
        self.assert_attribute("maxfunevals", '100*numberofvariables', 100 * self.numberOfVariables)

    def assert_attribute(self, attribute: str, pattern: str, default_value: any):
        """
        Ensure that a given attribute matches the expected pattern,
        and set it to a default value if necessary.

         Args:
             attribute (str): The attribute to check.
             pattern (str): The expected pattern for the attribute.
             default_value (any): The default value to set if the pattern matches.
         """
        current_value = getattr(self, attribute)
        if isinstance(current_value, str):
            if current_value.lower() == pattern:
                setattr(self, attribute, default_value)
            else:
                # TODO: Refael: need to through exception
                print(f"Option {attribute} must be integer value if not the default")


class LSQValues():
    """
    Class representing the computed values for the least squares optimization problem.

    Attributes:
        x: Current estimate of the optimization variables. dim: (2, )
        fvec: Residual vector computed by the regularization function. dim: (n, 1).
        A: Jacobian matrix of the residuals with respect to the optimization variables. dim: (n, 2).
        grad: Gradient of the objective function. dim: (2, 1).
        val: Value of the objective function (sum of squared residuals). (float).
    """

    def __init__(self, reg_func: Callable, x: np.ndarray, jac: Callable, *varargin: Tuple[Any]) -> None:
        """
        Initialize LSQValues with computed residuals, Jacobian, and other derived quantities.

        Parameters:
            reg_func: Function to compute the residuals (fvec) given x and additional arguments.
            x: Current estimate of the optimization variables. dim: (2, )
            jac: Function to compute the Jacobian matrix of the residuals.
            varargin: Additional arguments to be passed to reg_func and jac.

        Attributes:
            x: Current estimate of the optimization variables. dim: (2, )
            fvec: Residual vector computed by the regularization function. dim: (n_frames, 1).
            A: Jacobian matrix of the residuals with respect to the optimization variables. dim: (n_frames, 2).
            grad: Gradient of the objective function. dim: (2, 1).
            val: Value of the objective function (sum of squared residuals). (float).
        """
        self.x = x
        self.fvec = reg_func(x, *varargin)
        self.A = jac(x, *varargin)
        self.grad = atamult(self.A, self.fvec, -1)
        self.val = np.dot(self.fvec, self.fvec)
        self.assert_fvec()

    def assert_fvec(self):
        # Refael: the dimension of x is (2, ) and the dimension of fvec is (n_frame, ).
        if self.fvec.shape[0] < self.x.shape[0]:
            # TODO: Refael: need to through exception ???
            print("the number of equations must not be less than n")


class LSQVars:
    """
    Represents the variables and parameters used during the iterative
    optimization process in a least-squares problem.

    Attributes:
        n (int): Number of optimization variables (dimensions of `x`).
        lb (np.ndarray): Lower bounds for the optimization variables.
        ub (np.ndarray): Upper bounds for the optimization variables.
        dnewt (np.ndarray): Newton direction for updates.
        it (int): Current iteration number.
        numFunEvals (int): Number of function evaluations.
        numGradEvals (int): Number of gradient evaluations.
        ex (int): Exit flag status.
        vpcg (np.ndarray): Vector storing trust-region PCG iterations.
        vpos (np.ndarray): Vector storing positivity flags for trust-region.
        vval (np.ndarray): Vector storing objective function values per iteration.
        voptnrm (np.ndarray): The infinity norm of the scaled gradient (by self.v) at the current iteration.
        delta (float): Trust-region radius, which limits the size of parameter updates in trust-region optimization methods.
        nrmsx (float): Norm of step size. presenting how much the parameters x have changed in the current iteration.
        ratio (float): Ratio of the actual improvement in the objective function to the predicted improvement, used to
                       evaluate whether the step taken was effective.
        v (np.ndarray): Vector representing proximity of `x` to the bounds (`lb` and `ub`). dim: (2,).
        dv (np.ndarray): A binary mask indicating whether each parameter is actively constrained
                         by the bounds. Values are `1` if constrained and `0` otherwise.. dim: (2,).
        delbnd (float): Upper bound for trust-region delta.
    """

    def __init__(self, values: LSQValues, options: LSQOptions):
        self.n = values.x.shape[0]  # param
        self.lb = EMPTY_ARRAY  # param
        self.ub = EMPTY_ARRAY  # param
        self.dnewt = EMPTY_ARRAY
        self.it = 0  # counters
        self.numFunEvals = 0  # counters
        self.numGradEvals = 0  # counters
        self.ex = 0
        self.vpcg = np.zeros(options.itb)  # vector of trustregion.pcgit
        self.vpos = np.ones(options.itb)  # vector of posdef
        self.vval = np.zeros(options.itb)  # vector of vals
        self.vval[self.it] = values.val
        self.voptnrm = np.zeros(options.itb)  # vector
        self.delta = 10
        self.nrmsx = 1
        self.ratio = 0
        self.v = EMPTY_ARRAY
        self.dv = EMPTY_ARRAY
        self.delbnd = max(100 * norm(values.x), 1)  # param

    def increment_param(self, name, add):
        setattr(self, name, getattr(self, name) + add)

    def set_lower_bound(self, lower_bound):
        if lower_bound.size == 0:
            lower_bound = -INF * np.ones(self.n)
        lower_bound[lower_bound <= -1e10] = -INF
        self.lb = lower_bound

    def set_upper_bound(self, upper_bound):
        if upper_bound.size == 0:
            upper_bound = INF * np.ones(self.n)
        upper_bound[upper_bound >= 1e10] = INF
        self.ub = upper_bound

    def assert_bounds(self):
        if np.any(self.ub == self.lb):
            print("equal upper and lower bound not permitted")
            exit()
        elif min(self.ub - self.lb) <= 0:
            print("inconsistent bounds")
            exit()

    def compute_startx(self):
        '''startx returns centered point'''
        arg1 = (self.ub < INF) & (self.lb == -INF)
        arg2 = (self.ub == INF) & (self.lb > -INF)
        arg3 = (self.ub < INF) & (self.lb > -INF)
        arg4 = (self.ub == INF) & (self.lb == -INF)
        return self.set_startx_values(arg1, arg2, arg3, arg4)

    def set_startx_values(self, arg1, arg2, arg3, arg4):
        xstart = np.zeros(self.n)
        w = np.maximum(abs(self.ub), np.ones(self.n))
        ww = np.maximum(abs(self.lb), np.ones(self.n))

        xstart[arg1] = self.ub[arg1] - 0.5 * w[arg1]
        xstart[arg2] = self.lb[arg2] + 0.5 * ww[arg2]
        xstart[arg3] = (self.ub[arg3] + self.lb[arg3]) / 2
        xstart[arg4] = 1
        return xstart

    def define_dv_and_v(self, x: np.ndarray, grad: np.ndarray) -> None:
        """
        Defines the proximity (`v`) and directional adjustment indicator (`dv`)
        for the optimization parameters based on the current gradient and bounds.

        Args:
            x (np.ndarray): Current parameter values.
            grad (np.ndarray): Gradient vector of the objective function.

        Attributes Updated:
            -   `self.v` measures how close the current parameters (x) are to the bounds (lb and ub)
            -   `self.dv` is a binary vector indicating whether adjustments are needed for the bounds.
                    dv[i] = 1 if the corresponding parameter (x[i]) is actively interacting with a bound.
                    dv[i] = 0 if the parameter is unconstrained.
        """
        arg1 = (grad < 0) & (self.ub < INF)
        arg2 = (grad >= 0) & (self.lb > -INF)
        arg3 = (grad < 0) & (self.ub == INF)
        arg4 = (grad >= 0) & (self.lb == -INF)

        self.define_v(x, arg1, arg2, arg3, arg4)
        self.define_dv(arg1, arg2, arg3, arg4)

    def define_v(self, x: np.ndarray, arg1: np.ndarray, arg2: np.ndarray, arg3: np.ndarray, arg4: np.ndarray) -> None:
        v = np.zeros(self.n)
        v[arg1] = x[arg1] - self.ub[arg1]
        v[arg2] = x[arg2] - self.lb[arg2]
        v[arg3] = -1
        v[arg4] = 1
        self.v = v

    def define_dv(self, arg1: np.ndarray, arg2: np.ndarray, arg3: np.ndarray, arg4: np.ndarray) -> None:
        dv = np.zeros(self.n)
        dv[arg1] = 1
        dv[arg2] = 1
        dv[arg3] = 0  # maybe redundant
        dv[arg4] = 0  # maybe redundant
        self.dv = dv

    def evaluate_dnewt(self, fvec):
        if np.any(np.isinf(fvec)):
            # TODO: Refael: what we need to do with this message ???
            print("user function is returning inf or NaN values")
        # TODO: Refael:
        # 1) When is `fvec` 2D? Does it occur at any point in the algorithm?
        # 2) What is `self.vars`? Should it be `self.dnewt` instead?
        # 3) If `fvec` is 1D, when (if at all) we update `dnewt`?
        if fvec.ndim == 2 and fvec.shape[1] == 2:
            self.vars.dnewt = fvec[:, 1]
