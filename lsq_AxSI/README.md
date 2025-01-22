Usage Guide
===========

## Overview

This section describes how to use the **Nonlinear Least-Squares Optimization** program.

## Run from Command Line

To execute the program via the command line, use the following syntax:

```bash
lsq_AxSI \
  --x0 1.0 2.0 \
  --bounds -10.0 10.0 -5.0 5.0 \
  --ftol 1e-6 \
  --xtol 1e-6 \
  --diff_step 1e-3 \
  --max_nfev 10000
```

### Required Arguments:

- **`--x0`**: Initial guess for the parameters (space-separated values).
- **`--bounds`**: Parameter bounds as pairs (lower and upper limits for each parameter).

### Optional Arguments:

- **`--ftol`** *(default: 1e-6)*: Tolerance for changes in the cost function value.
- **`--xtol`** *(default: 1e-6)*: Tolerance for updates to the parameter values.
- **`--diff_step`** *(default: 1e-3)*: Step size for finite-difference approximation.
- **`--max_nfev`** *(default: 10000)*: Maximum number of function evaluations allowed.

## Function Import and Customization

The program also supports direct import of the `nonlinear_least_squares` function into Python scripts. This allows advanced customization, such as using custom residual (`reg_func`) and Jacobian (`jac`) functions.

### Example:

```python
from nonlinear_least_squares import nonlinear_least_squares

def custom_reg_func(x, *args):
    # Define custom residual computation
    pass

def custom_jacobian(x, *args):
    # Define custom Jacobian computation
    pass

result = nonlinear_least_squares(
    reg_func=custom_reg_func,
    x0=[1.0, 2.0],
    bounds=([-10.0, -5.0], [10.0, 5.0]),
    jac=custom_jacobian
)

print(result)
```

## Output Description

The program provides a summary of the optimization process, including:

### Key Results:

- **`Success`**: Indicates whether the optimization converged successfully.
- **`Optimized Parameters`**: The final parameter values at the solution.
- **`Residuals`**: Values of the residual function at the solution.
- **`Jacobian`**: Jacobian matrix at the solution.
- **`Exit Flag`**: Integer code indicating the reason for termination.

### Detailed Metadata:

The output includes a dictionary with the following fields:

- **`algorithm`**: Solver used for optimization.
- **`firstorderopt`**: Measure of first-order optimality.
- **`iterations`**: Number of iterations performed.
- **`funcCount`**: Number of function evaluations.
- **`cgiterations`**: Number of conjugate gradient iterations.
- **`Message`**: Descriptive message about the termination reason.




Python version
--------------

This project is currently using Python 3.12

Installation
------------

It is recommended to use **virtualenv** to create a clean python environment.

To install lsqAxSI, use **pip**:

    pip install lsqAxSI



Execution
---------

The main script shipped with this project is **lsq_AxSI.py**, see its options by running:

    lsq_AxSI.py -h

