# The main file for AxSI measures calculations

import logging

import cvxpy as cp
import numpy as np
import rpy2
import rpy2.robjects as ro
from gurobipy import GRB
from rpy2.robjects.numpy2ri import activate
from rpy2.robjects.packages import importr
from scipy.optimize import minimize

activate()  # Enable automatic conversion between R and NumPy

# Import necessary R packages
quadprog = importr("quadprog")

logging.getLogger('rpy2').setLevel(logging.WARNING)  # Suppress DEBUG logs from rpy2

logger = logging.getLogger(__name__)


def lin_least_squares_with_constrains_r(A: np.ndarray, b: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    Solve linear least squares with constraints using R's 'quadprog' package.

    Parameters:
        A (np.ndarray): The coefficient matrix.
        b (np.ndarray): The target vector.
        lb (np.ndarray): Lower bounds on the solution.
        ub (np.ndarray): Upper bounds on the solution.
        start_p (np.ndarray): Starting point for the linear least squares.

    Returns:
        np.ndarray: The optimized solution vector.
    """
    # tic =time.time()
    # Define the D matrix (A^T A) and d vector (-A^T b) for quadprog
    D = A.T @ A
    d = A.T @ b

    D_r = ro.r.matrix(D, nrow=D.shape[0], ncol=D.shape[1])
    d_r = ro.FloatVector(d)

    # Constraints matrix: combine lb and ub as inequality constraints
    # G matrix: [-I; I] (negative and positive identity matrices for bounds)
    G = np.vstack([np.ones((A.shape[1])), np.eye(A.shape[1]), -np.eye(A.shape[1])])
    h = np.hstack([1, lb, -ub])

    G_r = ro.r.matrix(G, nrow=G.shape[0], ncol=G.shape[1])
    h_r = ro.FloatVector(h)
    try:
        # Solve the quadratic programming problem using solve.QP.
        # https://www.rdocumentation.org/packages/quadprog/versions/1.5-8/topics/solve.QP
        result = quadprog.solve_QP(Dmat=D_r, dvec=d_r, Amat=G_r.T, bvec=h_r, meq=1)
        # Extract the solution
        solution = result.rx2("solution")
        return solution
    except rpy2.rinterface_lib.embedded.RRuntimeError:
        logger.info("SolverError: The linear LSQ problem failed.")
        return np.ones(A.shape[1])


def lin_least_squares_with_constraints_gurobi(A: np.ndarray, b: np.ndarray, lb: np.ndarray,
                                              ub: np.ndarray, model) -> np.ndarray:
    # Gurobi is not thread-safe, so a separate environment is required for each process/thread when using
    # multiprocessing or multithreading.
    # However, it's not necessary for multiprocessing in our case since we are utilizing Gurobi's internal parallelism
    # rather than Python's multiprocessing.
    # As for multithreading, it appears to work fine without a separate environment, but we include it as a precaution.
    # See here for more details:
    # https://support.gurobi.com/hc/en-us/articles/360043111231-How-do-I-use-multiprocessing-in-Python-with-Gurobi

    model.setParam('OutputFlag', 0)
    # Cannot run with Python's multiprocessing when using more than 1 thread in Gurobi's internal multiprocessing
    # (Gurobi's "threads" operate as separate processes). Python threads can coexist with Gurobi's threads,
    # but we always set Gurobi's thread count to 1 to maintain control over CPU usage.
    model.setParam("Threads", 1)

    # Create decision variables x (continuous variables between lb and ub)
    x = model.addMVar(A.shape[1], lb=lb, ub=ub, name="x")  # Multi-dimensional variables

    # Objective: Minimize the quadratic term 0.5 * (A @ x - b)^2
    error = A @ x - b
    quad_expr = 0.5 * (error @ error)  # (A @ x - b)^T @ (A @ x - b)

    # Set the objective to minimize the quadratic expression
    model.setObjective(quad_expr, GRB.MINIMIZE)

    # Add the sum(x) == 1 constraint
    model.addConstr(x.sum() == 1)
    # Optimize the model
    model.optimize()
    # Check the optimization result
    if model.status == GRB.OPTIMAL:
        return x.X
    else:
        logger.info("SolverError: The linear LSQ problem failed.")
        return np.ones(x.shape)


def lin_least_squares_with_constraints_scipy(A: np.ndarray, b: np.ndarray, lb: np.ndarray,
                                             ub: np.ndarray) -> np.ndarray:
    """
    use scipy.minimize to find x such that :
        minimize sum of squares for [A @ x - b]
        lb <= x <= ub and sum(x) == 1

    Parameters
    ----------
    A: Xprim
    b: Yprim
    lb: LOWER_BOUND
    ub: UPPER_BOUND

    Returns
    -------

    """

    # Define the objective function (least squares)
    def objective(x):
        return 0.5 * np.sum((A @ x - b) ** 2)

    def constraint_sum(x):
        return np.sum(x) - 1  # sum(x) == 1

    # Define constraints
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    x0 = np.zeros(A.shape[1])

    # Solve using minimize with bounds
    # x = minimize(objective, x0, constraints=constraints, bounds=np.column_stack((lb, ub)), method='SLSQP', options={'ftol': 1e-6, 'disp': True}).x
    # x = minimize(objective, x0, constraints=constraints, bounds=np.column_stack((lb, ub)), method='trust-constr').x
    x = minimize(objective, x0, constraints=constraints, bounds=np.column_stack((lb, ub))).x
    return x


def lin_least_squares_with_constraints_cvxpy(A: np.ndarray, b: np.ndarray, lb: np.ndarray,
                                             ub: np.ndarray) -> np.ndarray:
    """
    use cvxpy to find x such that :
    minimize sum of squares for [A @ x - b]
    lb <= x <= ub and sum(x) == 1
    """
    # Define your variables and problem data
    n = A.shape[1]
    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b))
    constraints = [x >= lb, x <= ub, cp.sum(x) == 1]

    # Create the optimization problem
    prob = cp.Problem(objective, constraints)

    # Try to solve the problem
    try:
        prob.solve(warm_start=True, solver=cp.ECOS)
        # Optionally, you can check the status and the optimal value:
    except cp.SolverError:
        logger.info("SolverError: The linear LSQ problem does not have a feasible solution.")
        return np.ones(x.shape)

    return x.value
