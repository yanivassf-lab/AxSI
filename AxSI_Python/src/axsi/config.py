import logging

import numpy as np

logger = logging.getLogger(__name__)

# parameters for later use
NUM_PROCESSES_PRED = 1  # use parallel if NUM_PROCESSES > 1 in prediction step
NUM_THREADS_PRED = 1  # Run multithreading instead of multiprocessing in prediction step
NUM_PROCESSES_AXSI = 1  # use parallel if NUM_PROCESSES > 1 in axsi step
NUM_THREADS_AXSI = 1  # Run multithreading instead of multiprocessing in axsi step
LINEAR_LSQ_METHOD = 'R-quadprog'  # Method for linear least squares, implemented using the 'R-quadprog' package
NON_LINEAR_LSQ_METHOD = 'R-minpack'  # Method for non-linear least squares, implemented using the 'R-minpack' package
ADD_VALS = np.arange(0.1, 32, 0.2)  # [0.1, 0.3, 0.5, ... , 31.9]
N_SPEC = len(ADD_VALS)  # N_SPEC == 160

DEBUG_MODE = False
SUBJ_FOLDER = ''

# least squares parameters (for calc_axsi.py file)
X0 = np.asarray([0.5, 5000.0])
MIN_VAL = np.zeros(2)
MAX_VAL = np.array([1.0, 20000.0])

# R_SERVER_PORT=6311 # not in use - R server is very slow
