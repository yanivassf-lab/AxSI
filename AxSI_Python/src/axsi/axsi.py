# The main file for running AxCaliber analysis
# Use DTI1000, shellDTI and AxSI
import logging
import time
from multiprocessing import Pool

from axsi import config
from axsi.ax_dti import dti
from axsi.calc_axsi import calc_axsi
from axsi.mri_scan import Scan

logger = logging.getLogger(__name__)


def init_axsi(file_names, *params) -> Scan:
    scan = Scan(file_names, *params)  # init Scan object
    return scan


def axsi_main(file_names, *params) -> None:
    tic = time.time()
    # initialize analysis
    scan = init_axsi(file_names, *params)  # return Scan object
    toc = time.time()
    logger.debug("Computation time initialization = " + str((toc - tic)) + " sec.")

    shell_dti = []  # array for DTI objects for each bvalue
    bval_shell = scan.get_shell()  # array of unique nonzero bvalues

    logger.info('Starting DTI-1000')
    # run DTI1000
    tic = time.time()
    # dti1000 = dti(scan, 1.0, is1000=True, save_to_file=True)
    first_shell = scan.get_first_shell()
    logger.info(f'First shell is: {first_shell}')
    dti1000 = dti(scan, first_shell, is1000=True, save_to_file=True)  # Refael
    toc = time.time()
    logger.debug("Computation time DTI on the first shell = " + str((toc - tic)) + " sec.")

    logger.info('Starting DTI-shell')
    # multiprocessing on each voxel is not efficient, instead we run each shell with all voxels in parallel.
    if config.NUM_PROCESSES_AXSI > 1 or config.NUM_PROCESSES_PRED > 1:
        tic = time.time()
        # run parallelly with maximum number of processes that user allocated.
        num_proc = min(max(config.NUM_PROCESSES_PRED, config.NUM_PROCESSES_AXSI), len(bval_shell))
        logger.info(f'Run DTI-shell parallely with {num_proc} processes/threads')
        with Pool(num_proc) as p:  # run volume_dti parallelly
            shell_dti = p.starmap(dti, [(scan, bval_shell[i]) for i in range(len(bval_shell))])
        toc = time.time()
    else:
        for i in range(len(bval_shell)):
            ax_dti = dti(scan, bval_shell[i])  # run DTI
            shell_dti.append(ax_dti)  # store result object in array

    logger.debug("Computation time DTI-shell = " + str((toc - tic)) + " sec.")

    # run AxSI analysis
    logger.info('Starting AxSI')
    calc_axsi(scan, dti1000, shell_dti)
