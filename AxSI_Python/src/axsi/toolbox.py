# This file provides methods and basic functionality for AxSI analysis
import logging
import multiprocessing
import os
import time
from multiprocessing import Process, managers
from threading import Thread
from typing import Callable

import gurobipy as gp
import numpy as np
import psutil
from scipy.ndimage import gaussian_filter
from scipy.stats import gamma

from axsi import config

logger = logging.getLogger(__name__)


def set_log_config():
    # Configure logging
    logging.basicConfig(
        filename=os.path.join(config.SUBJ_FOLDER, 'run_info.log'),  # Log file
        level=logging.DEBUG if config.DEBUG_MODE else logging.INFO,
        # Minimum log level (the available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Include module name
    )

    # Create a StreamHandler to output to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if config.DEBUG_MODE else logging.INFO)  # Same log level
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logging.getLogger().addHandler(console_handler)


def monitor_memory(interval: int = 10) -> None:
    """
    Monitor memory usage over specified interval

    :param interval: monitor interval in seconds
    """
    # Configure logger inside the child process
    # set_log_config()
    # Get the parent process PID
    parent_pid = os.getppid()

    # Get the parent process using its PID
    parent_process = psutil.Process(parent_pid)

    # Monitor memory usage of the parent and its children
    while True:
        total_memory = 0
        try:
            # Add memory usage of the parent process
            total_memory += parent_process.memory_info().rss

            # Iterate over all child processes of the parent
            for child in parent_process.children(recursive=True):
                total_memory += child.memory_info().rss
        except psutil.NoSuchProcess or psutil.AccessDenied or psutil.ZombieProcess:
            continue
        # Convert memory to megabytes (MB)
        total_memory_mb = total_memory / (1024 ** 2)
        logger.debug(f"Total memory usage (Parent + All Children): {total_memory_mb:.2f} MB")

        # Sleep for a while before checking again
        time.sleep(interval)


def parallel_batches(num_items: int, num_par: int, num_thr: int, func: Callable, *args) -> None:
    # Initialize an empty array for processes
    procs = []
    nun_processes = max(num_par, num_thr)
    chunk_size = int(np.ceil(num_items / nun_processes))
    # Calculate the number of chunks (chunk_size voxels per chunk)

    # Create processes for each chunk
    for chunk_idx in range(nun_processes):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_items)  # Ensure the last chunk does not exceed bounds
        if num_thr > 1:
            proc = Thread(
                target=batch_func,
                args=(start_idx, end_idx, func, *args)
            )
        else:
            proc = Process(
                target=batch_func,
                args=(start_idx, end_idx, func, *args)
            )
        procs.append(proc)  # Append process to array
        proc.start()  # Start process

    # Wait for all processes to finish
    for proc in procs:
        proc.join()


def batch_func(start_idx, end_idx, func, *args):
    """
    Processes a batch of tasks by dynamically updating arguments based on index.

    Parameters:
        start_idx (int): Starting index for processing.
        end_idx (int): Ending index for processing.
        func (callable): The function to call with the updated arguments. The last argument must be i.
        *args: The initial arguments to pass to the function.
    """
    if func.__name__ == 'perform_iteration':
        args, gurobi_env = create_new_gurobi_env(args)
    for i in range(start_idx, end_idx):
        func(*args, i)  # Call the function with arguments
    if func.__name__ == 'perform_iteration':
        delete_gurobi_env(gurobi_env)


def create_new_gurobi_env(args):
    # 1) Gurobi does not work correctly with Python's multiprocessing when using the 'fork' start method.
    #    However, our program relies on 'fork' to pass updated configuration values to child processes,
    #    as outlined in the AxSI_main.py script. To address this, here we switch to the 'spawn' method instead.
    #
    # 2) Gurobi is not thread-safe, so a separate environment is required for each process when using
    #    multiprocessing.
    #    For multithreading, it appears to work fine without a separate environment, but we include it as a precaution.

    # See here for more details:
    # https://support.gurobi.com/hc/en-us/articles/360043111231-How-do-I-use-multiprocessing-in-Python-with-Gurobi

    len_args = len(args)
    args = list(args)
    gurobi_env = None
    if config.LINEAR_LSQ_METHOD == 'gurobi':
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method("spawn")

        gurobi_env = gp.Env()
        if len(args) == len_args:
            args = args + [gurobi_env]
        else:
            args[-1] = gurobi_env
    else:
        args = list(args) + [None]
    return args, gurobi_env


def delete_gurobi_env(gurobi_env):
    del gurobi_env


class Values:
    """
    A basic class to be extended with different attributes.
    Provide get/set methods base on given attribute.
    Used for creating a shared object in parallel prcoessing.
    """

    def get(self, attr, i=-1):
        """
        Get value of given attribute.

        :param attr: assume self uses "attr" to hold array
        :param i:    element i in the array
        :return:     element i if i >= 0, otherwise the whole array
        """
        if i == -1:
            # return array
            return getattr(self, attr)
        # return specific element by index
        return getattr(self, attr)[i]

    def set(self, attr, new_val, i=-1) -> None:
        """
        Set element if i >= 0, otherwise replace the whole array

        :param attr:    assume self uses "attr" to hold array
        :param new_val: new value to be set
        :param i:       element i in the array
        """
        if i == -1:
            # replace array
            setattr(self, attr, new_val)
        else:
            # set specific element by index
            getattr(self, attr)[i] = new_val

    @staticmethod
    def copy_values(source, target, attributes) -> None:
        """
        Copy attributes from source class to target class.
        Assume source and target holds the same attribute.

        :param source: source class
        :param target: target class
        :param attributes: attributes to be copied
        """
        for attr in attributes:
            # replace array in target with source array
            target.set(attr, source.get(attr))

    def get_dict(self, attributes) -> dict:
        """
        Assume self holds data for each attr in attributes
        :param attributes: List of attributes
        :return: dictionary with attributes as keys and data as values
        """
        d = {attr: self.get(attr) for attr in attributes}
        return d

    def nonzero(self, attr) -> None:
        """
        For given attribute in self: set 0 instead of every negative value

        :param attr: attribute to be set 0
        """
        arr = self.get(attr)
        arr[arr <= 0] = 0
        self.set(attr, arr)


# init a new manager for parallel processing 
class MyManager(managers.BaseManager):
    pass


def cart2sph_env(v: np.ndarray) -> tuple:
    """ An envelope function for specific use

    :param v: expected dimension (3, times)
    :return: r, theta, phi
    """
    # Change the sign of the third element
    return cart2sph(v[0], v[1], -v[2])


def cart2sph(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    """
    convert from 3D cartesian representation to spherical representation
    """

    xy = np.sqrt(x ** 2 + y ** 2)  # sqrt(x² + y²)
    r = np.sqrt(xy ** 2 + z ** 2)  # r = sqrt(x² + y² + z²)
    phi = np.arctan2(z, xy)
    theta = np.arctan2(y, x)

    return r, theta, phi


def smooth4dim(data: np.ndarray) -> np.ndarray:
    """
    :param data: 4D array
    :return: 4D array of the same size as input, smooth each 3D slice based on last dimension
    """
    new_data = np.zeros(data.shape)
    for i in range(data.shape[3]):
        new_data[:, :, :, i] = smooth_data_slice(data[:, :, :, i])
    return new_data


def smooth_data_slice(slice: np.ndarray) -> np.ndarray:
    """
    Use gaussian filter to smooth input data with pre-defined parameters
    :param slice: 3D array
    :return: 3D array of the same size as input, smooth each 3D slice based on last dimension
    """
    smoothed_slice = gaussian_filter(slice, sigma=0.65, truncate=3, mode='nearest')
    return smoothed_slice


def init_yd() -> np.ndarray:
    """
    Initialization of array for calc_axsi

    :return: yd.shape == ADD_VALS.shape
    """
    alpha = 3
    beta = 2
    gamma_pdf = gamma.pdf(config.ADD_VALS, a=alpha, scale=beta)
    yd = gamma_pdf * np.pi * (config.ADD_VALS / 2) ** 2
    yd = yd / np.sum(yd)
    return yd


def init_l_matrix(n) -> np.ndarray:
    """
    Initialization of array for calc_axsi

    :param n: N_SPEC (len(ADD_VALS))
    :return:  l_mat.shape == (n, n+2) = (N_SPEC, N_SPEC+2)
              for n == 3 output is:
                         1, 0, 0, 0, 0
                        -1, 1, 0, 0, 0
                         0,-1, 1, 0, 0
    """
    ones = np.ones(n)
    # create ID matrix with a diagonal of -1 below the main diag
    l_mat = np.eye(n) - np.tril(ones, -1) * np.triu(ones, -1)
    # add two columns of 0
    l_mat = np.append(l_mat, np.zeros((n, 2)), axis=1)
    return l_mat
