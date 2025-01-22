#!/usr/bin/env python

import argparse
import numpy as np
from lsqAxSI.lsq_axsi import nonlinear_least_squares
from lsqAxSI.default_values import reg_func, jac_calc


def main():
    parser = argparse.ArgumentParser(description="Perform nonlinear least-squares optimization for AxSI.")

    parser.add_argument("--x0", type=float, nargs='+', required=True,
                        help="Initial guess for the parameters (space-separated list of floats).")
    parser.add_argument("--bounds", type=float, nargs='+', required=True,
                        help="Bounds for the parameters (space-separated list of floats, format: lower1 upper1 lower2 upper2 ...).")
    parser.add_argument("--ftol", type=float, default=1e-6,
                        help="Tolerance for the cost function value (default: 1e-6).")
    parser.add_argument("--xtol", type=float, default=1e-6,
                        help="Tolerance for parameter updates (default: 1e-6).")
    parser.add_argument("--diff_step", type=float, default=1e-3,
                        help="Step size for finite-difference approximation of the Jacobian (default: 1e-3).")
    parser.add_argument("--max_nfev", type=int, default=20000,
                        help="Maximum number of function evaluations (default: 20000).")

    args = parser.parse_args()

    # Parse bounds into a tuple of arrays
    bounds_array = np.array(args.bounds)
    if len(bounds_array) % 2 != 0:
        raise ValueError("Bounds must be provided in pairs (lower and upper for each parameter).")
    lower_bounds = bounds_array[0::2]
    upper_bounds = bounds_array[1::2]
    bounds = (lower_bounds, upper_bounds)

    # Initial parameters
    x0 = np.array(args.x0)

    result = nonlinear_least_squares(reg_func=reg_func, x0=x0, bounds=bounds, jac=jac_calc, ftol=args.ftol,
                                     xtol=args.xtol, diff_step=args.diff_step, max_nfev=args.max_nfev)

    print("Optimization Result:")
    print(f"Success: {result.exitflag == 1}")
    print(f"Optimized Parameters: {result.x}")
    print(f"Residuals: {result.fvec}")
    print(f"Jacobian: {result.jacob}")
    print(f"Exit Flag: {result.exitflag}")
    print(f"Output Details: {result.output}")
    print(f"Message: {result.msg}")


if __name__ == "__main__":
    main()


