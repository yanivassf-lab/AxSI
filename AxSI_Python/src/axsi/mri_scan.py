# This file defines MRI scan object for AxSI analysis
# Scan obect holds various objects for Data, bvals, bvecs and mask

import logging
import os
from pathlib import Path

import nibabel as nb
import numpy as np
from dipy.io import read_bvals_bvecs

from axsi import config
from axsi.matrix_operations import normalize_matrix_rows
from axsi.toolbox import cart2sph_env, smooth4dim

logger = logging.getLogger(__name__)


class Scan_File:
    """ A basic class holds a file path and its data loaded """

    def __init__(self, filepath: Path, raw_data: np.ndarray) -> None:
        self.filepath = filepath
        self.raw_data = raw_data  # data is already loaded


class NIFTIFile(Scan_File):
    ''' A class holds nifti file such as scan data and mask'''

    def __init__(self, filepath: Path) -> None:
        self.diff_file = nb.load(filepath)
        # load data for parent class
        super().__init__(filepath, self.diff_file.get_fdata())


class Scan:
    """
    Main class for scan object use in the analysis
    holds relevant data from files and parameters
    provides get functions to retreive data
    """

    def __init__(self, file_names: dict, *params) -> None:
        """
        Load data of bvals and bvecs files

        :param file_names: filenames should contain path to four files by specific keys
        :param params:     params should be ordered in specific way
        """
        bvals, bvecs = read_bvals_bvecs(str(file_names['bval']), str(file_names['bvec']))
        self.bval = BVAL(file_names['bval'], bvals)  # init BVAL object
        self.bvec = BVEC(file_names['bvec'], bvecs, self.bval)  # init BVEC object
        self.data = DATA(file_names['data'], self.bval)  # init DATA object
        self.mask = MASK(file_names['mask'])  # init MASK object
        self.shape = self.mask.raw_data.shape  # based on scan resolution. For example: (128, 128, 88) in total 1,441,792
        self.num_of_vols = np.prod(self.shape)  # number of voxels in scan. For example: 342,625
        self.param_dict = self.build_param_dict(*params)

    def build_param_dict(self, small_delta: int, big_delta: int, gmax: float, gamma_val: float = 4257) -> dict:
        """
        create a dictionary with scan parameters

        :param small_delta: (float) Gradient duration in miliseconds.
        :param big_delta: (float) Time to scan (time interval) in milisecond.
        :param gmax: (float) Gradient maximum amplitude in G/cm (or 1/10 mT/m).
                             gmax calculation: sqrt(bval*100/(7.178e8*0.03^2*(0.06-0.01)))
        :param gamma_val: Gyromagnetic ratio.
        :return: dictionary with scan parameters
        """
        grad_dirs = self.bvec.grad_dirs
        # init dictionary
        scan_param = {'nb0': self.bval.locs[0], 'small_delta': small_delta, 'big_delta': big_delta, 'gmax': gmax}
        max_q = gamma_val * small_delta * gmax / 10e6
        # convert grad_dirs to spherical representation
        r_q, phi_q, theta_q = cart2sph_env(grad_dirs.T)  # shape of r_q, phi_q, theta_q: (n_frames,)
        r_q = r_q * max_q  # multiply by coefficient
        scan_param['q_dirs'] = grad_dirs * max_q  # multiply by coefficient
        # set spherical representations
        scan_param['r'] = r_q
        scan_param['phi'] = phi_q
        scan_param['theta'] = theta_q
        return scan_param

    def get_smoothed_data(self, voxel_index=None, bvalue_index=None) -> np.ndarray:
        """
        Get smoothed signal data of mri scan.
        Data is flattened: dim=0 are the voxels, dim=1 time

        Taking smoosed data of specific time (all voxels) or specific voxel (all bvalues)
        If voxel_index=None, bvalue_index=None returns the all smoothed data after masking.

        Note that voxel_index is always None in this program.

        :param voxel_index: for taking data of specific voxels
        :param bvalue_index: for taking data of specific bvalue
        :return: smoothed data
        """
        data = self.data.smoothed[self.mask.index]
        if voxel_index:  # for taking data of specific voxels
            data = data[voxel_index, :]
        if bvalue_index:  # for taking data of specific bvalue
            data = data[:, bvalue_index]
        return data

    def get_signal_0(self, take_first: bool = False) -> np.ndarray:
        """
        :param take_first: (bool) flag for DTI
                            for DTI1000 use only the first appearance of bvalue==0
                            for shellDTI use the mean of all bvalue==0
        :return: 1D array of signal when bvalue==0
        """
        if take_first:
            # first image when bvalue==0
            return self.data.signal_0[self.mask.index]
        else:
            # mean over bvalue==0
            return self.data.signal_0_mean[self.mask.index]

    def get_locs_by_bvalues(self, bvalue) -> list:
        """
        :return a list of indices to images taken with bvalue
        """
        return self.bval.locs[bvalue]

    def get_shape(self) -> np.ndarray:
        """
        :return: brain resolution (of the raw data) (4D)
        """
        return self.shape

    def get_num_of_voxels(self) -> int:
        """
        :return: number of brain voxels (after masking)
                 ** not as the attribute num_of_vol (of raw data) kept in scan **
        """
        return len(self.mask.index)

    def get_params(self) -> dict:
        """
        :return: get dictionary of scan parameters
        """
        return self.param_dict

    def get_bval_data(self) -> np.ndarray:
        """
        :return: array of bval real values
        """
        return self.bval.data * 1000

    def get_bval_length(self) -> int:
        """
        :return: number of timeframes (after filteration)
        """
        return self.bval.length  # n_frames

    def get_shell(self) -> np.ndarray:
        """
        :return: array with unique nonzero elements of bvalues
        """
        return self.bval.shell

    def get_max_bval(self) -> float:
        """
        :return: maximum value of bval
        """
        return self.bval.max_bval

    def get_first_shell(self) -> float:
        """
        :return: first value of bval after 0
        """
        return self.bval.first_shell

    def get_grad_dirs(self) -> np.ndarray:
        """
        :return: array with bvec grad_dirs data
        """
        return self.bvec.grad_dirs

    def get_bvec_norm(self) -> np.ndarray:
        """
        :return: 2D array with normalized bvec data
        """
        return self.bvec.norm_data

    def reshape_to_brain_size(self, data: np.ndarray) -> np.ndarray:
        """
        :param data: data can be n-dim, but assuming data.shape[0] == len(mask.index), "and `data[0]` contains the
                     data as an array (which may include additional data in other dimensions)"
        :return: matrix of original dimension as raw_data (but in addition can contain additional data in other dimensions)
        """
        exshape = data.shape[1:]  # shape without first dimension, could be empty tuple
        new_data = np.zeros((self.num_of_vols,) + exshape)
        new_data[self.mask.index] = data
        new_data = new_data.reshape(self.shape + exshape)
        return new_data

    def save_files(self, files: dict[str, np.ndarray]) -> None:
        """
        Save nifti files in subj_folder based on files dictionary

        :param files: dictionary of files to save. dict should hold strings as keys and np.ndarray as values.
        :param subj_folder: folder to save files
        """
        for key in files.keys():
            filename = build_file_name(key)  # build a full path to file
            data = self.reshape_to_brain_size(files[key])  # plant brain data in array of original image dimension
            save_nifti(filename, data, self.data.diff_file.affine)


def save_nifti(fname: Path, img: np.ndarray, affine: np.ndarray) -> None:
    """
    Save img as nifti file

    :param fname: full path to save nifti file
    :param img: data to be saved
    :param affine: affine matrix
    """
    file_img = nb.Nifti1Image(img, affine)
    nb.save(file_img, fname)


def build_file_name(file_name: str) -> Path:
    """
    Build path to save files in a specific manner
    """
    return Path(os.path.join(config.SUBJ_FOLDER, file_name + '.nii.gz'))


class BVAL(Scan_File):
    """ Object to store data of bvals file """

    def __init__(self, filepath: Path, raw_data: np.ndarray) -> None:
        # call parent constructor with path and raw_data
        super().__init__(filepath, raw_data)
        # map values to rounded integers
        data = self.compute_rounded_data()
        self.org_length = len(data)
        # Remove 0<bval<1000 from calculation
        self.low_locs = self.find_low_locs(data)
        # TODO: (Refael) The deletion of low bvals need to be before the roundness, since after the round there are
        #  or 0 or 1 not between them. But it doesn't affect the results with the new equipment.
        # Remove 0<bval<1000 from calculation
        self.data = np.delete(data, self.low_locs)
        self.length = len(self.data)
        # get unique values of bvalue
        shell = np.unique(self.data)

        self.shell = shell[shell > 0]  # keep only nonzero bvalues
        # store for each bvalue the indices of relevant images
        self.locs = {val: np.where(self.data == val)[0].tolist() for val in shell}
        self.max_bval = np.max(self.data)  # maximal bvalue
        self.first_shell = float(np.unique(np.sort(self.data))[1])  # Refael: minimum bvalue after 0
        self.norm = np.sqrt(self.data / self.max_bval)  # normalize bvalue data.

    def compute_rounded_data(self) -> np.ndarray:
        """
        Round bvalues to find bad values.
        for example: 1995 => 2, 1500 => 1.5
        :return: rounded_data
        """
        bval2 = 2 * np.asarray(self.raw_data)  # multiply by 2
        rounded_data = bval2.round(-2) / 2000  # round last two digits, divide by 2000
        return rounded_data

    def find_low_locs(self, bval) -> np.ndarray:
        """ get locations where 0 < elements < 1 """
        return np.where((bval > 0) & (bval < 1))[0]


class BVEC(Scan_File):
    """ Object to store data of bvecs file """

    def __init__(self, filepath: Path, raw_data: np.ndarray, bval: BVAL) -> None:
        # call parent constructor with path and loaded data
        super().__init__(filepath, raw_data)
        # use bval to reshape data if needed
        data = np.reshape(self.raw_data, [bval.org_length, -1])  # dims: (n_frames, 3)
        # use bval to omit bad values
        data = np.delete(data, bval.low_locs, 0)  # data: (n_frames, 3)
        # keep data multiplited by bvals coefficient
        self.grad_dirs = data * bval.norm[np.newaxis].T
        # normalize data
        self.norm_data = normalize_matrix_rows(data)
        self.norm_grad_dirs = normalize_matrix_rows(self.grad_dirs)


class DATA(NIFTIFile):
    """
    Object to store signal data of mri scan.
    Assuming preprocessed for now.
    """

    def __init__(self, filepath: Path, bval: BVAL) -> None:
        # call parent constructor with path
        super().__init__(filepath)
        data = np.asarray(self.raw_data, dtype='float64')
        # use bval to omit bad values
        data = np.delete(data, bval.low_locs, 3)
        data[data <= 0] = 0.0
        X, Y, Z, n = data.shape  # assume data has 4 dimensions
        self.corrected = data
        smoothed_data = smooth4dim(data)  # smooth every image separately
        self.smoothed = smoothed_data.reshape((X * Y * Z, n))  # flatten the brain to make data 2D
        # bval is always 0 in the first time.
        self.signal_0 = self.smoothed[:, bval.locs[0][0]]  # keep the signal when bvalue=0 for the first time
        self.signal_0_mean = np.nanmean(self.smoothed[:, bval.locs[0]], axis=1)  # mean across bvalue=0


class MASK(NIFTIFile):
    """
    Object to store brain mask file
    """

    def __init__(self, filepath) -> None:
        # call parent constructor with path
        super().__init__(filepath)
        self.flattened = self.raw_data.flatten()  # flatten the images to be 1D
        self.index = np.where(self.flattened > 0)[0]  # indices for voxels of brain
