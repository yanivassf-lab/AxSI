# This script not in use - R server is very slow
# R server enable run R with multithreads

import numpy as np
import pyRserve

from axsi import config


def lin_least_squares_with_constrains_r(A: np.ndarray, b: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                                        start_p: np.ndarray) -> np.ndarray:
    """
    Solve linear least squares with constraints using R's 'quadprog' package via Rserve.

    Parameters:
        A (np.ndarray): The coefficient matrix.
        b (np.ndarray): The target vector.
        lb (np.ndarray): Lower bounds on the solution.
        ub (np.ndarray): Upper bounds on the solution.
        start_p (np.ndarray): Starting point for the linear least squares.

    Returns:
        np.ndarray: The optimized solution vector.
    """
    # Define the D matrix (A^T A) and d vector (-A^T b) for quadprog
    D = A.T @ A
    d = A.T @ b

    # Constraints matrix: combine lb and ub as inequality constraints
    # G matrix: [-I; I] (negative and positive identity matrices for bounds)
    G = np.vstack([np.ones((A.shape[1])), np.eye(A.shape[1]), -np.eye(A.shape[1])])
    h = np.hstack([1, lb, -ub])

    try:
        # Connect to Rserve
        conn = pyRserve.connect(host="localhost", port=config.R_SERVER_PORT)

        # Send the data to R
        conn.assign("D", D.tolist())  # Send as list for R compatibility
        conn.assign("d", d.tolist())
        conn.assign("G", G.tolist())
        conn.assign("h", h.tolist())

        # Run the quadprog solver in R
        conn.void_eval("""
        library(quadprog)
        result <- solve.QP(Dmat = as.matrix(D), dvec = as.numeric(d), 
                           Amat = t(as.matrix(G)), bvec = as.numeric(h), meq = 1)
        solution <- result$solution
        """)

        # Retrieve the solution from R
        solution = np.array(conn.eval("solution"))
        conn.close()  # Close the connection to Rserve

        return solution
    except Exception as e:
        print(f"Error: {e}")
        return np.ones(A.shape[1])  # Return a default solution if the solver fails
