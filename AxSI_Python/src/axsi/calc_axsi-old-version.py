# The main file for AxSI measures calculations
import logging
import time
from pathlib import Path

import numpy as np
from lsqAxSI.lsq_axsi import nonlinear_least_squares
from scipy.optimize import least_squares

from axsi import config
from axsi.ax_dti import DTI_Values
from axsi.linear_lsq import lin_least_squares_with_constrains_r, lin_least_squares_with_constraints_gurobi, \
    lin_least_squares_with_constraints_scipy, lin_least_squares_with_constraints_cvxpy
from axsi.matrix_operations import exp_vec_mat_multi
from axsi.mri_scan import Scan
from axsi.non_linear_lsq import least_squares_envelope_r, reg_func_r, jac_calc_r, reg_func, jac_calc
from axsi.predictions import Predictions
from axsi.predictions import predict
from axsi.toolbox import MyManager, Values, parallel_batches
from axsi.toolbox import init_l_matrix, init_yd

logger = logging.getLogger(__name__)

# attributes of DTI calculations
KEYS = ['pfr', 'ph', 'pcsf', 'pasi', 'paxsi', 'CMDfr', 'CMDfh']


class Calc_AxSI_Values(Values):
    """An extended class for Values with relevant attributes for AxSI """

    def __init__(self, n: int) -> None:
        """
        Parameters
        ----------
        n: number of voxels
        """
        self.pfr = np.zeros(n)
        self.ph = np.zeros(n)
        self.pcsf = np.zeros(n)
        self.pasi = np.zeros(n)
        self.paxsi = np.zeros((n, config.N_SPEC))  # N_SPEC is defined in toolbox
        self.CMDfr = np.zeros(n)
        self.CMDfh = np.zeros((n, 2))


# register class to MyManager, used for multiprocessing
MyManager.register('Calc_AxSI_Values', Calc_AxSI_Values)


class Calc_AxSI():
    """ The main object for performing AxSI analysis """

    def __init__(self, scan: Scan, decays: Predictions) -> None:
        """
        Use Scan for init, don't keep it
        
        :param scan:    scan object
        :param decays:  object with the decays of hindered, restricted and csf.
        """
        self.YD = init_yd()
        self.L_MATRIX = init_l_matrix(config.N_SPEC)  # dim: (N_SPEC, N_SPEC+2)
        self.lower_bound = np.zeros(config.N_SPEC + 1)
        self.upper_bound = np.ones(config.N_SPEC + 1)
        self.decays = decays  # Predictions object
        self.data = scan.get_smoothed_data()  # smoothed MRI signal
        self.n_vols = scan.get_num_of_voxels()  # number of brain voxels in mask
        # in multiprocessing (except of gurobi) the shared memory initialized in main() method.
        if config.NUM_PROCESSES == 1 or config.LINEAR_LSQ_METHOD == 'gurobi':
            self.values = Calc_AxSI_Values(self.n_vols)  # init struct with zero values
        # compute in advance
        self.vCSF = init_pixpredictCSF(scan.get_grad_dirs())  # depends on scan values, dims: (n_frames, )

    class Block:
        """ Object for a singel itaration data """

        def __init__(self, decays: Predictions, ydata: np.ndarray, vCSF: np.ndarray, i: int) -> None:
            """ Use decays to predict arrays for single voxel only """
            self.i = i
            self.ydata = ydata[i]  # scan signal, dims: (n_frames, )
            self.vCSF = vCSF  # dims: (n_frames, )
            self.vH = decays.get_hindered_slice(i)  # dims: (n_frames, )
            self.vR = decays.get_restricted_slice(i)  # dims: (n_frames, N_SPEC)
            self.prcsf_i = decays.get_csf_slice(i)  # float
            self.vH[self.vH > 1] = 0  # correct vH data

        def stack(self) -> np.ndarray:
            """ Put vR, vH, vCSF in matrix together """
            vR = np.nan_to_num(self.vR, nan=0)  # replace NaN with 0
            preds = np.column_stack((vR, self.vH, self.vCSF))  # dims: (n_frames, N_SPEC+2)
            return preds

    # The same as in DTI file
    def save_calc_files(self, scan: Scan, subj_folder: Path) -> None:
        """ save data of each measurement to file """
        files_dict = self.values.get_dict(KEYS)  # create dict of keys and data
        scan.save_files(files_dict, subj_folder)  # use scan object to save data in dict

    def main(self) -> None:
        """
        Calculate AxSI to each voxel separatelly
        Parallellism depends on hardcoded value of num_of_processed
        Save a struct with AxSI values (not a shared one in any case)
        """
        # Gurobi cannot operate with Python's multiprocessing when the method is set to 'fork' (which is required in
        # our program, as specified in the AxSI_main.py script). Therefore, within the
        # lin_least_squares_with_constraints_gurobi() function, we utilize Gurobi's internal multiprocessing
        # mechanism instead.
        if config.NUM_PROCESSES > 1 and config.LINEAR_LSQ_METHOD != 'gurobi':
            # run in parallel
            with MyManager() as manager:
                # Create a shared object
                self.values = manager.Calc_AxSI_Values(self.n_vols)
                parallel_batches(self.n_vols, self.perform_iteration, self.decays, self.data, self.vCSF)
                temp = self.values
                self.values = Calc_AxSI_Values(self.n_vols)
                # Copy values from the shared object to self
                Values.copy_values(temp, self.values, KEYS)
        elif config.NUM_THREADS > 1:
            parallel_batches(self.n_vols, self.perform_iteration, self.decays,
                             self.data, self.vCSF)
        else:
            for i in range(self.n_vols):
                self.perform_iteration(self.decays, self.data, self.vCSF, i)

        self.values.set('pcsf', self.decays.csf)

    def perform_iteration(self, decays: Predictions, ydata: np.ndarray, vCSF: np.ndarray,
                          i: int) -> None:
        """
        Calculate AxSI measurements for the i'th voxel

        :param decays:  Predictions object
        :param ydata:   smoothed MRI signal (n_vols, n_frames)
        :param vCSF:    dims: (n_frames, )
        :param i:       index of voxel
        """
        if i // 1000 == i / 1000:
            logger.info(f'calc axsi iter: {i}')

        # init an object with iteration data
        block = Calc_AxSI.Block(decays, ydata, vCSF, i)
        # nonlinear least squares with predictions and signal
        parameter_hat = self.least_squares_envelope(block)
        # linear least squares with predictions and signal
        x = self.solve_vdata(block, parameter_hat)
        # update iteration results in object
        self.set_values(x, parameter_hat, block)

    def least_squares_envelope(self, block: Block) -> np.ndarray:
        """
        'lsq-axsi' is an own version of lsq to achieve more similar results to MATLAB lsqnonlin

        :param block: object with iteration data
        :return: array with two parameters of the model
        """
        vRes = np.dot(block.vR, self.YD)  # YD is the same for all voxels, dims: (n_frames, )
        vRes = np.nan_to_num(vRes, nan=0)  # replace NaN with 0
        # run nonlinear least squares with regression function
        if config.NON_LINEAR_LSQ_METHOD == 'R-minpack':
            parameter_hat = least_squares_envelope_r(reg_func_r, config.X0, bounds=(config.MIN_VAL, config.MAX_VAL),
                                                     jac=jac_calc_r,
                                                     ftol=1e-6, xtol=1e-6, diff_step=1e-3, max_nfev=20000,
                                                     args=(block.ydata, block.vH, vRes, block.vCSF, block.prcsf_i))
        elif config.NON_LINEAR_LSQ_METHOD == 'scipy':
            parameter_hat = least_squares(reg_func, config.X0, jac=jac_calc, bounds=(config.MIN_VAL, config.MAX_VAL),
                                          method='trf',
                                          ftol=1e-6, xtol=1e-6, gtol=1e-06, x_scale=1.0, loss='linear', f_scale=1.0,
                                          diff_step=1e-3, tr_solver=None, tr_options={}, jac_sparsity=None,
                                          max_nfev=20000, verbose=0,
                                          args=(block.ydata, block.vH, vRes, block.vCSF, block.prcsf_i), kwargs={}).x
        elif config.NON_LINEAR_LSQ_METHOD == 'lsq-axsi':
            parameter_hat = nonlinear_least_squares(reg_func, config.X0, bounds=(config.MIN_VAL, config.MAX_VAL),
                                                    jac=jac_calc, ftol=1e-6,
                                                    xtol=1e-6, diff_step=1e-3, max_nfev=20000,
                                                    args=(block.ydata, block.vH, vRes, block.vCSF, block.prcsf_i)).x
        return parameter_hat

    def solve_vdata(self, block: Block, parameter_hat: np.ndarray) -> np.ndarray:
        """
        Linear least squares with predictions and signal
        Input: iteration data and results of lsqnonlin
        Output: np array of shape (N_SPEC+2,)
        """
        # divide signal with nonlinlsq result
        vdata = block.ydata / parameter_hat[1]  # dims: (n_frames, )
        # adjust variables to current voxels
        lower_bound_i = np.append(self.lower_bound, block.prcsf_i - 0.02)  # add a last element
        upper_bound_i = np.append(self.upper_bound, block.prcsf_i + 0.02)  # add a last element
        Xprim = np.concatenate((block.stack(), self.L_MATRIX))  # dims: (n_frames+N_SPEC, N_SPEC+2)
        Yprim = np.concatenate((vdata, np.zeros(config.N_SPEC)))  # dims: (n_frames+N_SPEC, )
        # run linear least squares
        if config.LINEAR_LSQ_METHOD == 'R-quadprog':
            x = lin_least_squares_with_constrains_r(Xprim, Yprim, lower_bound_i, upper_bound_i)
        elif config.LINEAR_LSQ_METHOD == 'gurobi':
            x = lin_least_squares_with_constraints_gurobi(Xprim, Yprim, lower_bound_i, upper_bound_i)
        elif config.LINEAR_LSQ_METHOD == 'scipy':
            x = lin_least_squares_with_constraints_scipy(Xprim, Yprim, lower_bound_i, upper_bound_i)
        elif config.LINEAR_LSQ_METHOD == 'cvxpy':
            x = lin_least_squares_with_constraints_cvxpy(Xprim, Yprim, lower_bound_i, upper_bound_i)
        return x  # dims: (N_SPEC+2, )

    def set_values(self, x: np.ndarray, parameter_hat: np.ndarray, block: Block) -> None:
        """ update object with current voxel values """
        i = block.i  # voxel index
        x[x < 0] = 0  # replace negative values with zeros
        nx = x[:130]  # take the first 130 element
        nx = nx / np.sum(nx)  # normalize
        pasi_i = np.sum(nx * config.ADD_VALS[:130])  # ADD_VALS is defined in toolbox
        self.values.set('ph', x[160], i)
        self.values.set('pfr', np.min([1 - block.prcsf_i - x[160], 0.]), i)
        self.values.set('pasi', pasi_i, i)
        self.values.set('paxsi', x[:160], i)
        self.values.set('CMDfh', parameter_hat, i)
        self.values.set('CMDfr', 1 - parameter_hat[0] - block.prcsf_i, i)


def calc_axsi(scan: Scan, dti1000: DTI_Values, shell_dti: list[DTI_Values], subj_folder: Path) -> None:
    """
    Envelope function for running AxSI.
    Use DTI1000 and shellDTI calculated for scan.

    :param scan:        scan object
    :param dti1000:     DTI_Values object for dti1000
    :param shell_dti:   list of DTI_Values objects for shell_dti
    :param subj_folder: data will be saved to files in path
    """
    tic = time.time()
    decays = predict(scan, dti1000, shell_dti)  # Calculate predictions decays for each environment
    toc = time.time()
    logger.debug("Computation time predictions = " + str((toc - tic)) + " sec.")
    # init AxSI object
    calc = Calc_AxSI(scan, decays)
    # run main function of AxSI analysis
    tic = time.time()
    calc.main()
    toc = time.time()
    logger.debug("Computation time Calc_AxSI.main " + ("with multi-processes/multi-threads "
                                                       if config.NUM_PROCESSES > 1 or config.NUM_THREADS > 1 else "") + "= " + str(
        (toc - tic)) + " sec.")
    logger.info("Saving calc files")
    calc.save_calc_files(scan, subj_folder)


def init_pixpredictCSF(grad_dirs: np.ndarray) -> np.ndarray:
    """
    A series of matrix multiplications

    :param grad_dirs: matrix of (n_frames, 3)
    :return: np array of shape (n_frames,)
    """
    D_mat = np.eye(grad_dirs.shape[1]) * 4  # dim: (3,3)
    pixpredictCSF = exp_vec_mat_multi(grad_dirs, D_mat, -4)
    return pixpredictCSF
