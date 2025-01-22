# This file provide methods for matrix calculations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def array_to_sym(arr: np.ndarray) -> np.ndarray:
    """
    Converts an nd-array into a symmetric matrix.
    Expect len(arr) to be number of element in upper triangle of a squared matrix
    example for input: [1,2,3,4,5,6]:
        1, 2, 3
        2, 4, 5
        3, 5, 6

    :param arr: 1D array
    :return: 2D matrix of shape (size, size)
    """
    size = get_mat_size_by_n_elements(len(arr))
    mat = np.zeros([size, size])
    mat[np.triu_indices(size)] = arr
    mat = mat.T
    mat[np.triu_indices(size)] = arr
    return mat


def get_mat_size_by_n_elements(n: int) -> int:
    """
    return the size of squared matrix
    with n elements in upper triangle
    print message and return n if no such solution
    """
    roots = np.roots([1, 1, -2 * n])
    pos_roots = roots[roots > 0]
    if pos_roots[0].is_integer():
        return int(pos_roots[0])
    else:
        logger.info(f"could not convert array of len: {n} to symmetric matrix")
        return n


def sym_mat_to_array(mat: np.ndarray) -> np.ndarray:
    """
    1D array with matrix elements. elements with above diagonal are multiplied by 2
    for input: 1, 2, 3
               2, 4, 5
               3, 5, 6
    output:    [1, 2*2, 2*3, 4, 2*5, 6]
    :param mat: symmetric matrix
    :return:    1D array with matrix elements
    """
    mat = mat + np.triu(mat, k=1)  # multiply by two elements above diag
    arr = mat[np.triu_indices(3)]  # build array with elements of upper triangular
    return arr


def normalize_matrix_rows(mat: np.ndarray) -> np.ndarray:
    """
    Normalize each row in input separately

    :param mat: 2D matrix
    :return: normalized matrix according to rows
    """
    rows_norm = np.linalg.norm(mat, axis=1, keepdims=True)  # compute array of norm values of each row
    # divide each row by its own norm, ignoring zero rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        norm_mat = np.where(rows_norm > 0, mat / rows_norm, 0)
    return norm_mat


def exp_vec_mat_multi(vec: np.ndarray, mat: np.ndarray, coeff: float, axis=0) -> np.ndarray:
    """
    A series of matrix and vector multiplications and exponents

    :param vec:     vectors. can be several vectors (a matrix)
    :param mat:     matrix of shape ([any], vector.shape[axis], vector.shape[axis]), where axis is the parameter axis.
                    The first dimension is optional, if exists, broadcasting will be done on this dimension.
    :param coeff:   float. coefficient of vector multiplication
    :param axis:    how the vectors are stored in the vec. 0 - the vector are in rows of vec, 1 - in columns of vec.
    :return:        e^(diag(coeff * vec @ mat @ trans(vec)))
                    where diag is function that take only vector with the diagonal elements.
    """
    multi_mat_vec = np.dot(mat, vec.T)  # mat @ trans(vec)
    # array of main diagonal of vec @ mat @ trans(vec)
    multi_vec_mat_vec = np.sum(multi_mat_vec * vec.T, axis=axis)
    exp = np.exp(coeff * multi_vec_mat_vec)
    return exp
