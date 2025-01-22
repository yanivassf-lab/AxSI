# This file provide methods for matrix calculations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def vector_len(v) -> int:
    """
    v could be either number or nd-array
    if v is instance of ndarray, return its first dimension
    else v has no shape attribute, treat it as an array of length 1
    """
    try:
        # if v is instance of nd-array
        length = v.shape[0]
    except:
        # v is a number
        length = 1
    return length


def normalize(v: np.ndarray) -> np.ndarray:
    """
    v is instance of nd-array
    returns v divided by its euclidean norm
    """
    if np.linalg.norm(v) > 0:
        return v / np.linalg.norm(v)
    return v


def atamult(A: np.ndarray, Y: np.ndarray, flag: int) -> np.ndarray:
    """ Multiply matrices A,Y according to a given flag """
    if flag == 0:
        V = np.dot(A.T, np.dot(A, Y))  # V = Trans(A) @ A @ Y
    elif flag > 0:
        V = np.dot(A, Y)  # V = A @ Y
    else:
        V = np.dot(A.T, Y)  # V = Trans(A) @ Y
    return V


def compute_w(H: np.ndarray, DM: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    A series of matrix multiplication
    Output: DM @ Trans(H) @ H @ DM @ Z
    """

    w = np.dot(DM, Z)  # w  = DM @ Z
    ww = atamult(H, w, 0)  # ww = Trans(H) @ H @ DM @ Z
    w = np.dot(DM, ww)  # w  = DM @ Trans(H) @ H @ DM @ Z
    return w
