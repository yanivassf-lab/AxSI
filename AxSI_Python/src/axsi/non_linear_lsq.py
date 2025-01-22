import logging
from typing import Optional, Callable, Tuple, Any

import numpy as np
import rpy2.robjects as ro
from lsqAxSI.lsq_components import LSQResult, defaultopt
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

minpack = importr('minpack.lm')

logger = logging.getLogger(__name__)


def reg_func(x: np.ndarray, ydata: np.ndarray, pixpredictH: np.ndarray, pixpredictR: np.ndarray,
             pixpredictCSF: np.ndarray, prcsf: float) -> np.ndarray:
    """
    Regression function for nonlinear least squares.

    Computes the error between observed data (`ydata`) and predicted data based on
    a model combining hindered, restricted, and CSF signal components, using a set
    of parameters.

    :param x: Array of parameters for the model, where:
              - x[0]: Fraction of the hindered component.
              - x[1]: Scaling factor for the predicted signal.
    :param ydata: Observed signal data. Shape: (n_frames,).
    :param pixpredictH: Predicted hindered component of the signal. Shape: (n_frames,).
    :param pixpredictR: Predicted restricted component of the signal. Shape: (n_frames,).
    :param pixpredictCSF: Predicted CSF (cerebrospinal fluid) component of the signal. Shape: (n_frames,).
    :param prcsf: Fraction of the CSF component in the signal.
    :return: The residual error between the observed (`ydata`) and the predicted signal.
             Shape: (n_frames,).
    """
    # xt = 1 - x[0] - prcsf
    # newdata = x[1] * (x[0] * pixpredictH + xt * pixpredictR + prcsf * pixpredictCSF)
    newdata = x[1] * (x[0] * pixpredictH + (1 - x[0] - prcsf) * pixpredictR + prcsf * pixpredictCSF)
    err = newdata - ydata
    return err


def jac_calc(x, ydata, pixpredictH, pixpredictR, pixpredictCSF, prcsf) -> np.ndarray:
    """ jacobian matrix calculation for nonlinear least squares (reg_func function). dim: (n_frames, 2) """
    jac = np.zeros([len(ydata), 2])
    jac[:, 0] = x[1] * (pixpredictH - pixpredictR)
    jac[:, 1] = x[0] * pixpredictH + (1 - x[0] - prcsf) * pixpredictR + prcsf * pixpredictCSF
    return jac


# Define R function in Python
reg_func_r = ro.r('''
function(x, ydata, pixpredictH, pixpredictR, pixpredictCSF, prcsf) {
    xt <- 1 - x[1] - prcsf
    newdata <- x[2] * (x[1] * pixpredictH + xt * pixpredictR + prcsf * pixpredictCSF)
    err <- newdata - ydata
    return(err)
}
''')

jac_calc_r = ro.r('''
function(x, ydata, pixpredictH, pixpredictR, pixpredictCSF, prcsf) {
  # Initialize Jacobian matrix
  n_frames <- length(ydata)
  jac <- matrix(0, nrow = n_frames, ncol = 2)

  # Calculate the first column
  jac[, 1] <- x[2] * (pixpredictH - pixpredictR)

  # Calculate the second column
  jac[, 2] <- x[1] * pixpredictH + (1 - x[1] - prcsf) * pixpredictR + prcsf * pixpredictCSF

  return(jac)
}

''')


def least_squares_envelope_r(reg_func: Callable, x0: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray], jac: Callable,
                             ftol: float = 1e-6, xtol: float = 1e-6, diff_step: float = 1e-3,
                             max_nfev: int = 20000, args: Optional[Tuple[Any, ...]] = None) -> LSQResult:
    x0_r = FloatVector(x0)
    min_val_r = FloatVector(bounds[0])
    max_val_r = FloatVector(bounds[1])
    ydata_r = FloatVector(args[0])
    pixpredictH_r = FloatVector(args[1])
    pixpredictR_r = FloatVector(args[2])
    pixpredictCSF_r = FloatVector(args[3])
    prcsf_r = float(args[4])

    # Define control settings
    control = minpack.nls_lm_control(maxiter=defaultopt['MaxIter'], maxfev=max_nfev, ftol=ftol)

    # Run the optimization
    result = minpack.nls_lm(
        par=x0_r,
        fn=reg_func,
        jac=jac,
        lower=min_val_r,
        upper=max_val_r,
        control=control,
        ydata=ydata_r,
        pixpredictH=pixpredictH_r,
        pixpredictR=pixpredictR_r,
        pixpredictCSF=pixpredictCSF_r,
        prcsf=prcsf_r
    )
    # Extract and return the optimized parameters
    return dict(result.items())['par']
