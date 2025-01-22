# This file is for running DTI for AxSI analysis
import logging

import numpy as np

from axsi.matrix_operations import array_to_sym, sym_mat_to_array
from axsi.mri_scan import Scan
from axsi.toolbox import Values, MyManager

logger = logging.getLogger(__name__)

EPS = np.finfo('float').eps  # 2.22X10-16
KEYS = ['dt', 'md', 'fa', 'eigval', 'eigvec']  # attributes of DTI calculations


class DTI_Values(Values):
    """
    An extended class for Values with relevant attributes for DTI
    """

    def __init__(self, n: int) -> None:
        """
        :param n: number of voxels
        """
        self.fa = np.zeros(n)
        self.md = np.zeros(n)
        self.dt = np.zeros((n, 3, 3))
        self.eigval = np.zeros((n, 3))
        self.eigvec = np.zeros((n, 3))


# register class to MyManager, used for multiprocessing
MyManager.register('DTI_Values', DTI_Values)


class AXSI_DTI:
    """
    The main object for performing DTI
    """

    def __init__(self, scan: Scan, bvalue: float, is1000: bool) -> None:
        """
        Use scan for initialization, don't keep it

        :param scan:    scan object
        :param bvalue:  bvalue for the DTI
        :param is1000:
        """
        self.num_of_vols = scan.get_num_of_voxels()  # number of brain voxels in mask
        self.values = DTI_Values(self.num_of_vols)  # init struct with zero values
        # compute in advance
        bvlocs = scan.get_locs_by_bvalues(bvalue)  # indices of specific bvalue
        self.bval_arr = self.compute_bval_arr(scan, bvlocs)  # dims: (len(bvlocs), 6)
        # signal log is computed differently for DTI1000 and shellDTI
        # dims: (n_vols, len(bvlocs)) - n_vols after masking
        self.signal_log = self.compute_signal_log(scan, bvlocs, is1000)

    def main(self) -> DTI_Values:
        """
        Calculate DTI to each voxel separatelly.

        :return: struct with dti values (not a shared one in any case)
        """
        # comment out for now. multiprocessing in this step is not efficient, instead we run each shell
        # with all voxels in parallel.
        if False:  # config.NUM_PROCESSES == 0:
            pass
            # logger.info('Run parallelism with num_of_processes={}'.format(config.NUM_PROCESSES))
            # # run parallelly
            # with MyManager() as manager:
            #     # create a shared object to use in mutliprocessing
            #     self.values = manager.DTI_Values(self.num_of_vols)
            #     parallel_batches(self.num_of_vols, self.volume_dti, False)
            #     self.extract_data()  # copy data from shared object to regular object
        else:
            # run sequentially
            for i in range(self.num_of_vols):
                self.volume_dti(i)
        return self.values

    def volume_dti(self, i: int) -> None:
        """
        Calculate dti measurements for the i'th voxel
        :param i: index of the voxel
        """
        self.volume_dt_calc(i)
        self.volume_eigen_calc(i)
        self.volume_md_calc(i)
        self.volume_fa_calc(i)

    def volume_dt_calc(self, i: int) -> None:
        """
        Calculate diffusion tensor for the i'th voxel over all len(bvlocs).
        Diffusion tensor is a (3,3) symmetric matrix.

        :param i: index of the voxel
        """
        bval_arr = self.bval_arr  # dims: (len(bvlocs), 6)
        signal_log_i = -1 * (self.signal_log[i, :])  # dims: (len(bvlocs), )
        dt_i = np.linalg.lstsq(bval_arr, signal_log_i, rcond=None)[0]  # dims: (6,)
        self.values.set('dt', array_to_sym(dt_i), i)  # dims: (3,3)

    def volume_md_calc(self, i: int) -> None:
        """
        Calculate mean diffusivity for the i'th voxel mean diffusivity is a float

        :param i: index of the voxel
        """
        md_i = np.mean(self.values.get('eigval', i))
        self.values.set('md', md_i, i)

    def volume_fa_calc(self, i: int) -> None:
        """
        Calculate fractional anisotropy for the i'th voxel

        :param i: index of the voxel
        """
        eigval_i = self.values.get('eigval', i)
        md_i = self.values.get('md', i)
        fa_i = np.sqrt(1.5) * np.linalg.norm(eigval_i - md_i) / np.linalg.norm(eigval_i)
        self.values.set('fa', fa_i, i)

    def volume_eigen_calc(self, i: int) -> None:
        """
        Calculate eigen values and eigen vectors of the diffusion tensor of the i'th voxel
        :param i: index of the voxel
        """
        dt_mat = self.values.get('dt', i)  # shape: (3, 3)
        eigen_vals, eigen_vecs = np.linalg.eig(dt_mat)  # compute eigen values and vectors
        index = np.argsort(eigen_vals)  # get indices of sorted eigen values
        eigen_vals = eigen_vals[index] * 1000
        self.volume_eigval_calc(eigen_vals, i)
        self.volume_eigvec_calc(eigen_vecs, eigen_vals, index, i)

    def volume_eigval_calc(self, eigen_vals: np.ndarray, i: int) -> None:
        """
        Change the eigen values to their absolute values (or epsilon) and set them sorted in increasing order in
        attribute self.values.eigval

        :param eigen_vals: sorted eigen values based on eigen values increasing order
        :param i: index of the voxel
        """
        if np.all(eigen_vals < 0):  # if all eigen values are negative:
            eigen_vals = np.abs(eigen_vals)  # change the sign of all values
        eigen_vals[eigen_vals <= 0] = EPS  # replace negative eigen values with EPS
        self.values.set('eigval', eigen_vals, i)

    def volume_eigvec_calc(self, eigen_vecs: np.ndarray, eigen_vals: np.ndarray, index: int, i: int) -> None:
        """
        Sort eigen vectors and keep (in self.values.eigvec attribute) the "last" one based on eigen values increasing
        order (multiplied by its eigen value)

        :param eigen_vecs: eigen vectors
        :param eigen_vals: sorted eigen values based on eigen values increasing order
        :param index: indexes of the eigen values according their sorted values
        :param i: index of the voxel
        """

        eigen_vecs = eigen_vecs[:, index]  # sort based in eigen values order
        eigen_vecs = eigen_vecs[:, -1] * eigen_vals[-1]  # take the last vector and multiply it
        self.values.set('eigvec', eigen_vecs, i)

    def extract_data(self) -> None:
        """
        Copy data from shared object to a new regular object set the new object instead of the shared one
        """
        shared_values = self.values
        output_values = DTI_Values(self.num_of_vols)  # init new object
        DTI_Values.copy_values(shared_values, output_values, KEYS)  # copy from shared to new
        self.values = output_values  # replace it

    def compute_bval_arr(self, scan: Scan, bvlocs: np.ndarray) -> np.ndarray:
        """
        Compute in advance for tensors calculation

        :param scan:    Scan object
        :param bvlocs:  locations of specific bvalue
        :return:        array of shape (len(bvlocs), 6)
        """
        n = len(bvlocs)
        bval_arr = np.zeros((n, 6))
        bval_real = scan.get_bval_data()[bvlocs]  # dims: (len(bvlocs), )
        norm_bvec = scan.get_bvec_norm()[bvlocs, :]  # dims: (len(bvlocs), 3)

        for i in range(n):
            bmat = bval_real[i] * np.outer(norm_bvec[i, :], norm_bvec[i, :])  # dims: (3,3)
            bval_arr[i, :] = sym_mat_to_array(bmat)  # dims: (6, )

        return bval_arr  # dims: (len(bvlocs), 6)

    def compute_signal_log(self, scan: Scan, bvlocs: np.ndarray, is1000: bool) -> np.ndarray:
        """
        :param scan:    Scan object
        :param bvlocs:  locations of specific bvalue
        :param is1000:  If `1`, the signal is divided by the signal at the initial time point where `bvals=0`;
                        otherwise, it is divided by the average of signals where `bvals=0`.
        :return:        matrix of dims: (n_vols, len(bvlocs)) - n_vols after masking
        """
        signal = scan.get_smoothed_data(bvalue_index=bvlocs)  # dims (len(n_vols), len(bvlocs))
        signal_0 = scan.get_signal_0(is1000)[np.newaxis].T  # dims (len(n_vols), 1)

        signal_log = np.log((signal / signal_0) + EPS)  # dims: (len(n_vols), len(bvlocs)))
        signal_log[np.isnan(signal_log)] = 0
        signal_log[np.isinf(signal_log)] = 0

        return signal_log

    def save_dti_files(self, scan: Scan) -> None:
        """
        Save data of each measurement to file
        :param scan:    Scan object
        :param subj_folder: folder where data is saved
        """
        files_dict = self.values.get_dict(KEYS)  # create dict of keys and data
        scan.save_files(files_dict)  # use scan object to save data in dict


def dti(scan: Scan, bvalue: float, is1000: bool = False, save_to_file: bool = False) -> DTI_Values:
    """
    Envelope function for running AxSI DTI of specific bvalue

    :param scan:        scan object
    :param bvalue:      bvalue to calculate envelope for
    :param is1000:      flag to distinguish DTI1000 and shellDTI
    :param subj_folder: data will be saved to files iff path is given
    :return:            object containing data calculated in DTI
    """
    logger.info(f"Run DTI on shell with b-value: {bvalue}")
    bvalue_dti = AXSI_DTI(scan, bvalue, is1000)  # init calc object
    # call main function of calc object
    values = bvalue_dti.main()  # return value is a "result" object
    if save_to_file:  # if path was given
        bvalue_dti.save_dti_files(scan)  # save files to path
    return values
