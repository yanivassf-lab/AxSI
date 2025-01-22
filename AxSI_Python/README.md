# Usage Guide

## Overview

This documentation provides details on how to use the AxSI parser for analyzing MRI data using various input parameters.

## Run from Command Line

To execute the program via the command line, use the following syntax:

```bash
axsi-main.py \
  --subj-folder /path/to/subject_folder \
  --run-name "run_001" \
  --data /path/to/data.nii.gz \
  --bval /path/to/bval.bval \
  --bvec /path/to/bvec.bvec \
  --mask /path/to/mask.nii.gz \
  --small-delta 20 \
  --big-delta 50 \
  --gmax 8.0 \
  --gamma-val 4258 \
  --num-processes-pred 35 \
  --num-threads-pred 1 \
  --num-processes-axsi 1 \
  --num-threads-axsi 10 \
  --linear-lsq-method R-quadprog \
  --nonlinear-lsq-method gurobi \
  --debug-mode
```

### Required Arguments

- **`--subj-folder`**: Path to the subject folder (must exist)
- **`--run-name`**: Specify the name for the run
- **`--data`**: Path to the data file
- **`--bval`**: Path to the bval file
- **`--bvec`**: Path to the bvec file
- **`--mask`**: Path to the mask file

### Optional Arguments

- **`--small-delta`** *(default: 15)*: Gradient duration in milliseconds.
- **`--big-delta`** *(default: 45)*: Time to scan (time interval) in milliseconds.
- **`--gmax`** *(default: 7.9)*: Gradient maximum amplitude in G/cm.
- **`--gamma-val`** *(default: 4257)*: Gyromagnetic ratio.
- **`--num-processes-pred`** *(default: 1)*: Number of processes to run in parallel in prediction step.
- **`--num-threads-pred`** *(default: 1)*: Number of threads to run in parallel in prediction step.
- **`--num-processes-axsi`** *(default: 1)*: Number of processes to run in parallel in AxSI step.
- **`--num-threads-axsi`** *(default: 1)*: Number of threads to run in parallel in AxSI step.
- **`--linear-lsq-method`** *(default: R-quadprog)*: Method for linear least squares. **Choices
  **: `R-quadprog`, `gurobi`, `scipy`, `cvxpy`
- **`--nonlinear-lsq-method`** *(default: R-minpack)*: Method for nonlinear least squares. **Choices
  **: `R-minpack`, `scipy`, `lsq-axsi`
- **`--debug-mode`**: Enable debug mode. If not provided, debug mode is disabled by default.


# NIfTI Viewer

## Overview

This is a Dash-based web application that allows users to interactively visualize slices of 3D or 4D NIfTI files (
commonly used in neuroimaging). Users can select slices along different axes, apply various color maps, and visualize
data dynamically using sliders for timepoints (in 4D data) and slice indices.

## Features

### Input File Handling

- The NIfTI file path is provided as a command-line argument using `argparse`.
- The script loads and processes the provided file using the `nibabel` library.

### Visualization

- Supports visualization along three axes: axial, sagittal, and coronal.
- Provides several color maps (e.g., gray, viridis, plasma) for the visualization.
- Interactive user interface for exploring slices.

### 4D Data Support

- If the input NIfTI file contains 4D data (e.g., time-series or multi-volume data), a slider lets users navigate
  through different timepoints.

### User Controls

- Dropdown menus to select the viewing axis and color map.
- Sliders for selecting specific slices and timepoints.

### Output

- The visualization is rendered as an interactive plot using Plotly.

## Getting Started

Run the script from the command line, providing the NIfTI file path as an argument:

```bash
nifti-viewer --nifti-file /path/to/your/file.nii.gz
```

### Example

```bash
nifti-viewer.py --nifti-file example_data/pasi.nii.gz
```

The app runs at [http://127.0.0.1:8050](http://127.0.0.1:8050) by default, displaying the interactive visualization.


# Installation Instructions

## Step 1: Install Miniconda

1. **Download the Miniconda installer**:
   Open your terminal and run the following command to download the Miniconda installer:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

2. **Make the installer executable**:
   Change the permissions of the downloaded script to make it executable:
   ```bash
   chmod +x Miniconda3-latest-Linux-x86_64.sh
   ```

3. **Run the installer**:
   Execute the installer script to install Miniconda:
   ```bash
   ./Miniconda3-latest-Linux-x86_64.sh
   ```
   Follow the on-screen instructions to complete the installation. You may need to agree to the license terms and specify the installation location.

## Step 2: Configure Conda

4. **Add the conda-forge channel**:
   After installing Miniconda, configure it to use the conda-forge channel, which provides additional packages:
   ```bash
   conda config --add channels conda-forge
   ```

## Step 3: Create a New Conda Environment

5. **Create a new conda environment**:
   Create a new environment named `axsi` with R version 4.4.2 and Python version 3.12:
   ```bash
   conda create --name axsi -c conda-forge r-base=4.4.2 python=3.12
   ```

6. **Activate the new environment**:
   Activate the newly created environment:
   ```bash
   conda activate axsi
   ```

## Step 4: Verify Installation

7. **Check the R installation**:
   Verify that R is installed correctly by running:
   ```bash
   which R
   ```

8. **Check the Python installation**:
   Verify that Python is installed correctly by running:
   ```bash
   which python
   ```

9. **Check Python version**:
   Confirm the installed version of Python:
   ```bash
   python --version
   ```

10. **Check R version**:
    Confirm the installed version of R:
    ```bash
    R --version
    ```

## Step 5: Install AxSI Packages

11. **Install the axsi package**:
    install the `axsi` package using pip:
    ```bash
    pip install axsi
    ```

12.  **Install AxSI-specific R components**:
    The `axsi` package provide a command to install necessary R packages: **'quadprog'** and **'minpack.lm'**.

    **Note**: R version 4.4.2 already installed on your system in the creation of the environment:

    ```bash
    axsi-install-r
    ```

13.  **Optional: Run AxSI tests**:
    AxSI comes with a test suite, you can run it using:
    ```bash
    axsi-run-tests
    ```

Execution
---------

The main script shipped with this project is **axsi-main**, see its options by running:

    axsi-main -h


This program implements the methodology from the paper:

Gast, H., Horowitz, A., Krupnik, R., Barazany, D., Lifshits, S., Ben-Amitay, S., & Assaf, Y.
*A Method for In-Vivo Mapping of Axonal Diameter Distributions in the Human Brain Using Diffusion-Based Axonal Spectrum Imaging (AxSI)*.

The Python code was originally developed by Hila Gast and improved by Refael Kohen <refael.kohen@gmail.com>, Yaniv Assaf Lab, Tel Aviv University.