import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def extract_times(input_bval_path, input_bvec_path, input_data_path, input_mask_path, output_dir, times_to_extract):
    # Define paths for the files
    os.makedirs(output_dir, exist_ok=True)

    times_str = 'test_vol_' + '_'.join(map(str, times_to_extract))
    # Output paths for the new files
    bval_out_path = os.path.join(output_dir, times_str + "_bval.txt")
    bvec_out_path = os.path.join(output_dir, times_str + "_bvec.txt")
    data_nii_out_path = os.path.join(output_dir, times_str + "_data.nii")
    mask_nii_out_path = os.path.join(output_dir, times_str + "_mask.nii")

    # Load the original data
    data_img = nib.load(input_data_path)  # 4D data
    mask_img = nib.load(input_mask_path)  # 3D mask

    data = data_img.get_fdata()  # Shape: (width, height, channels, times)
    mask = mask_img.get_fdata()  # Shape: (width, height, channels)

    # Selected times
    selected_times = times_to_extract
    new_data = data[..., selected_times]  # Shape: (width, height, channels, time)

    # Load and filter bvals and bvecs
    bvals = np.loadtxt(input_bval_path)  # Shape: (times,)
    bvecs = np.loadtxt(input_bvec_path)  # Shape: (3, times)

    new_bvals = bvals[selected_times]  # Shape: (times,)
    new_bvecs = bvecs[:, selected_times]  # Shape: (3, times)

    # Save the new data to a NIfTI file
    new_data_img = nib.Nifti1Image(new_data, data_img.affine)
    nib.save(new_data_img, data_nii_out_path)

    # Save the new mask (same as original)
    nib.save(mask_img, mask_nii_out_path)

    # Save the new bval and bvec files
    np.savetxt(bval_out_path, new_bvals, fmt='%d')
    np.savetxt(bvec_out_path, new_bvecs, fmt='%.54f')

    print("New files saved successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract a subset of times for testing purposes.')

    # Define command-line arguments
    parser.add_argument('--input-dir', type=Path, help='Path to the input folder', required=True)
    parser.add_argument('--bval', type=Path, help='bval file name', required=True)
    parser.add_argument('--bvec', type=Path, help='bvec file name', required=True)
    parser.add_argument('--data', type=Path, help='data file nmae', required=True)
    parser.add_argument('--mask', type=Path, help='mask file name', required=True)
    parser.add_argument('--output-dir', type=Path, help='Path to the output directory', required=True)
    parser.add_argument('--times', type=int, nargs='+', help='Specific times to extract')

    # Parse the command-line arguments
    args = parser.parse_args()

    input_files = [os.path.join(args.input_dir, file_path) for file_path in
                   [args.bval, args.bvec, args.data, args.mask]]

    # Check if each file exists
    for file_path in input_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    extract_times(*input_files, args.output_dir, args.times)
    # check_output(args.output_dir)

"""
Run command: (run from tests folder)
====================================
python extract_timeframes.py --input-dir '/Users/user/Library/CloudStorage/GoogleDrive-refaelkohen@mail.tau.ac.il/My Drive/TLV-U-drive/BrainWork/AxSI-pipeline/input-data/sub-CLMC10/ses-202407110849/dwi' --bval 'sub-CLMC10_ses-202407110849_space-dwi_desc-preproc_dwi.bval' --bvec 'sub-CLMC10_ses-202407110849_space-dwi_desc-preproc_dwi.bvec' --mask 'sub-CLMC10_ses-202407110849_space-dwi_desc-brain_mask.nii.gz' --data 'sub-CLMC10_ses-202407110849_space-dwi_desc-preproc_dwi.nii.gz' --output-dir '/Users/user/PycharmProjects/AxSI/AxSI_Python/AxSI_python/tests/test_data/part_times' --times 0 1 5 8 80 81
"""
