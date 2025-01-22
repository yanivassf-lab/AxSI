#!/usr/bin/env python

# This file is used for running AxCaliber analysis of diffusion MRI data
# This file get command line arguments and start the analysis

# Do not import NumPy here! It is imported within the set_numpy_multithreading_method() function
# (**after** thread settings are configured).

"""
Configure NumPy to use a single thread for operations that rely on BLAS libraries.

Motivation:
NumPy operations like np.dot can use multithreading when linked to BLAS libraries
(e.g., OpenBLAS, MKL) to improve performance. However, multithreading may not always
be desirable, such as when running in environments with limited CPU resources, avoiding
contention with other processes, or ensuring deterministic behavior. This function
limits NumPy to single-threaded execution by setting relevant environment variables.

Note:
This is particularly useful in scenarios like the `calc_axsi.py` script, specifically
in the `least_squares_envelope()` method, where the line `vRes = np.dot(vR, self.YD)`
may unintentionally invoke multithreading, causing performance issues.
"""
import os

# must be before ohter imports
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Limits OpenBLAS threads
os.environ["MKL_NUM_THREADS"] = "1"  # Limits MKL threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Limits NumExpr threads
os.environ["OMP_NUM_THREADS"] = "1"  # Limits OpenMP threads

import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, Any

from axsi import config
from axsi.axsi import axsi_main
from axsi.toolbox import set_log_config, monitor_memory

logger = logging.getLogger(__name__)


def set_multiprocessing_method():
    """
    Set the multiprocessing start method to 'fork' for better compatibility and performance.

    Motivation:
    On platforms like Windows and macOS, the default multiprocessing start method (e.g., 'spawn')
    creates child processes with a clean interpreter, which does not inherit the current runtime
    state (e.g., dynamically updated config from argparse). Using 'fork' ensures  that child processes
    inherit a copy of the parent process's state instead import again the config.py withthe default values.
    """
    multiprocessing.set_start_method('fork', True)  # Set to 'fork' and enforce it
    logger.debug(f'Multiprocessing method is: {multiprocessing.get_start_method()}')  # Verify the current method


def build_files_dict(*paths):
    keys = ["data", "mask", "bval", "bvec"]
    filenames = {}
    if len(keys) != len(paths):
        logger.error("Wrong number of input paths, make sure paths to data, mask, bval and bvec exists")
    for i in range(len(keys)):
        filenames[keys[i]] = assert_path(paths[i])
    return filenames


def create_output_folder(subject_folder: Path, run_name: Path):
    subject_folder = assert_path(subject_folder)
    output_folder = subject_folder / run_name
    os.makedirs(output_folder, exist_ok=False)
    return output_folder


def assert_path(p: Path):
    if not p.exists():
        logger.error(f"Path given is not valid: {p}")
        raise ValueError(f"Path given is not valid: {p}")
    return p


def validate_args(args):
    if (args.num_processes_pred > 1 and args.num_threads_pred > 1) or (
            args.num_processes_axsi > 1 and args.num_threads_axsi > 1):
        raise ValueError("Input parameters are invalid: you cannot have both more than one process and more than one "
                         "thread simultaneously in prediction step or in axsi step.")
    if args.num_threads_axsi > 1 and (
            args.linear_lsq_method == 'R-quadprog' or args.nonlinear_lsq_method == 'R-minpack'):
        raise ValueError("Invalid input parameters: currently, running R packages with multiple threads is not "
                         "supported.")
    args.subj_folder = create_output_folder(args.subj_folder, args.run_name)


def update_config_file(args):
    config.SUBJ_FOLDER = args.subj_folder
    config.NUM_PROCESSES_PRED = args.num_processes_pred
    config.NUM_THREADS_PRED = args.num_threads_pred
    config.NUM_PROCESSES_AXSI = args.num_processes_axsi
    config.NUM_THREADS_AXSI = args.num_threads_axsi
    config.DEBUG_MODE = args.debug_mode
    config.LINEAR_LSQ_METHOD = args.linear_lsq_method
    config.NON_LINEAR_LSQ_METHOD = args.nonlinear_lsq_method
    # config.R_SERVER_PORT = args.rserver_port # for Rserver - not in use, it is very slow


def run_axsi(args):
    validate_args(args)

    file_names = build_files_dict(args.data, args.mask, args.bval, args.bvec)
    small_delta = args.small_delta
    big_delta = args.big_delta
    gmax = args.gmax
    gamma_val = args.gamma_val

    update_config_file(args)
    set_log_config()
    set_multiprocessing_method()

    # R
    # set_r_environment()
    # for Rserver - not in use, it is very slow
    # server_socket_rserver = upload_rserver(config.R_SERVER_PORT)
    # if not server_socket_rserver:
    #     raise ValueError(f"The R server port {config.R_SERVER_PORT} is not available. Please select a different.")

    if args.debug_mode:
        # Start the monitor_memory function as a separate process
        monitor_process = multiprocessing.Process(target=monitor_memory)
        monitor_process.start()
    try:
        if args.subj_folder:
            logger.info("Starting AxSI")
            logger.info(f"The parameters are: subj_folder: {args.subj_folder} \nfilenames: {file_names} \nsmall_delta: "
                        f"{small_delta} \nbig_delta: {big_delta} \ngmax: {gmax} \ngamma_val: {gamma_val}"
                        f"\nnonlinear-lsq-method: {args.nonlinear_lsq_method} \nlinear-lsq-method: "
                        f"{args.linear_lsq_method} \nnum-processes-pred: {args.num_processes_pred} "
                        f"\nnum-threads-pred: {args.num_threads_pred}  \nnum-processes-axsi: {args.num_processes_axsi} "
                        f"\nnum-threads-axsi: {args.num_threads_axsi}")
            axsi_main(file_names, small_delta, big_delta, gmax, gamma_val)
    finally:
        # for Rserver - not in use, it is very slow
        # print(f"Port {config.R_SERVER_PORT} has been closed.")
        # server_socket_rserver.close()  # Ensure the port of R server is freed by closing the socket
        if args.debug_mode:
            # Ensure the memory monitor stops when the script ends
            monitor_process.terminate()
            monitor_process.join()
            logger.debug("[Main] Memory monitor terminated.")


def parse_args_from_dict(parser: argparse.ArgumentParser, test_args: Dict[str, Any]) -> argparse.Namespace:
    """
    Parse argparse arguments from a dictionary of parameter values.

    Converts the dictionary `test_args` into a list of CLI-style arguments,
    then uses `parser` to parse those arguments and return a Namespace.

    Boolean values are treated as flags:
    If True, the flag (e.g., --debug-mode) is added, if False, it is omitted.

    Args:
        parser: An argparse.ArgumentParser instance to parse arguments.
        test_args: Dictionary mapping argument names (with underscores) to values.

    Returns:
        An argparse.Namespace with parsed arguments.
    """
    cli_args = []
    for arg, val in test_args.items():
        flag = f"--{arg.replace('_', '-')}"
        if isinstance(val, bool):
            if val:
                cli_args.append(flag)  # Add flag only if True
            # If False, omit the flag (default behavior for store_true)
        else:
            cli_args.extend([flag, str(val)])
    return parser.parse_args(cli_args)


def main(test_args=None):
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='AxSI for MRI data')

    # Define command-line arguments
    parser.add_argument('--subj-folder', type=Path, help='Path to the subject folder (must exist)', required=True)
    parser.add_argument('--run-name', type=str, help='Specify the name for the run', required=True)
    parser.add_argument('--data', type=Path, help='Path to the data file', required=True)
    parser.add_argument('--bval', type=Path, help='Path to the bval file', required=True)
    parser.add_argument('--bvec', type=Path, help='Path to the bvec file', required=True)
    parser.add_argument('--mask', type=Path, help='Path to the mask file', required=True)
    parser.add_argument('--small-delta', type=float, default=15,
                        help='Gradient duration in miliseconds (default: %(default)s)')
    parser.add_argument('--big-delta', type=float, default=45,
                        help='Time to scan (time interval) in milisecond (default: %(default)s)')
    parser.add_argument('--gmax', type=float, default=7.9,
                        help='Gradient maximum amplitude in G/cm (default: %(default)s)')
    parser.add_argument('--gamma-val', type=int, default=4257, help='Gyromagnetic ratio (default: %(default)s)')
    parser.add_argument('--num-processes-pred', type=int, default=1,
                        help='Number of processes to run in parallel in prediction step (default: %(default)s)')
    parser.add_argument('--num-threads-pred', type=int, default=1,
                        help='Number of threads to run in parallel in prediction step (default: %(default)s)')
    parser.add_argument('--num-processes-axsi', type=int, default=1,
                        help='Number of processes to run in parallel in AxSI step (default: %(default)s)')
    parser.add_argument('--num-threads-axsi', type=int, default=1,
                        help='Number of threads to run in parallel in AxSI step (default: %(default)s)')
    parser.add_argument('--nonlinear-lsq-method', type=str, default='R-minpack',
                        choices=['R-minpack', 'scipy', 'lsq-axsi'],
                        help="Method for linear least squares. Choose from 'R-minpack', 'scipy', or 'lsq-axsi'."
                             " (default: %(default)s).")
    parser.add_argument('--linear-lsq-method', type=str, default='R-quadprog',
                        choices=['R-quadprog', 'gurobi', 'scipy', 'cvxpy'],
                        help="Method for linear least squares. Choose from 'R-quadprog', 'gurobi', 'scipy', or 'cvxpy'."
                             " (default: %(default)s).")
    parser.add_argument('--debug-mode', action='store_true', help='Enable debug mode (default is disabled).')
    # parser.add_argument('--rserver-port', type=int, help='Port number for R server.', default=6311) # for Rserver - not in use, it is very slow

    if test_args is not None:
        # For pytest
        args = parse_args_from_dict(parser, test_args)
    else:
        # Normal CLI from terminal
        args = parser.parse_args()
    run_axsi(args)


if __name__ == "__main__":
    main()

# Example of run commenad (run from AxSI_python folder):

# Full input files:
# =================
# python AxSI_main.py --run-name test_run --subj-folder full-output-temp  --bval '/Users/user/Library/CloudStorage/GoogleDrive-refaelkohen@mail.tau.ac.il/My Drive/TLV-U-drive/BrainWork/AxSI-pipeline/input-data/sub-CLMC10/ses-202407110849/dwi/sub-CLMC10_ses-202407110849_space-dwi_desc-preproc_dwi.bval' --bvec '/Users/user/Library/CloudStorage/GoogleDrive-refaelkohen@mail.tau.ac.il/My Drive/TLV-U-drive/BrainWork/AxSI-pipeline/input-data/sub-CLMC10/ses-202407110849/dwi/sub-CLMC10_ses-202407110849_space-dwi_desc-preproc_dwi.bvec' --mask '/Users/user/Library/CloudStorage/GoogleDrive-refaelkohen@mail.tau.ac.il/My Drive/TLV-U-drive/BrainWork/AxSI-pipeline/input-data/sub-CLMC10/ses-202407110849/dwi/sub-CLMC10_ses-202407110849_space-dwi_desc-brain_mask.nii.gz' --data '/Users/user/Library/CloudStorage/GoogleDrive-refaelkohen@mail.tau.ac.il/My Drive/TLV-U-drive/BrainWork/AxSI-pipeline/input-data/sub-CLMC10/ses-202407110849/dwi/sub-CLMC10_ses-202407110849_space-dwi_desc-preproc_dwi.nii.gz' --debug-mode --num-processes 35 --num-threads 1

# Toy example - 15000 voxels:
# ===========================
# AxSI_main.py --run-name expected_outputs --subj-folder tests/test_data/output_files --bval "tests/test_data/input_files/voxels_50_52_bval.txt" --bvec "tests/test_data/input_files/voxels_50_52_bvec.txt" --data "tests/test_data/input_files/voxels_50_52_data.nii" --mask "tests/test_data/input_files/voxels_50_52_mask.nii"