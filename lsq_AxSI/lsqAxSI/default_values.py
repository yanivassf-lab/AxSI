import numpy as np

#########################################################################################################
# Default reg_func and jac_func functions
#########################################################################################################

X0 = np.asarray([0.5, 5000.0])
MIN_VAL = np.zeros(2)
MAX_VAL = np.array([1.0, 20000.0])


# Placeholder regression function
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


# Placeholder jacobian function
def jac_calc(x, ydata, pixpredictH, pixpredictR, pixpredictCSF, prcsf) -> np.ndarray:
    """ jacobian matrix calculation for nonlinear least squares (reg_func function). dim: (n_frames, 2) """
    jac = np.zeros([len(ydata), 2])
    jac[:, 0] = x[1] * (pixpredictH - pixpredictR)
    jac[:, 1] = x[0] * pixpredictH + (1 - x[0] - prcsf) * pixpredictR + prcsf * pixpredictCSF
    return jac

#########################################################################################################
