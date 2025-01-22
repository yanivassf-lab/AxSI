NIfTI Viewer
=============

This is a Dash-based web application that allows users to interactively visualize slices of 3D or 4D NIfTI files (commonly used in neuroimaging). Users can select slices along different axes, apply various color maps, and visualize data dynamically using sliders for timepoints (in 4D data) and slice indices.

Features
--------
- **Input File Handling**:
  - The NIfTI file path is provided as a command-line argument using `argparse`.
  - The script loads and processes the provided file using the `nibabel` library.

- **Visualization**:
  - Supports visualization along three axes: axial, sagittal, and coronal.
  - Provides several color maps (e.g., gray, viridis, plasma) for the visualization.
  - Interactive user interface for exploring slices.

- **4D Data Support**:
  - If the input NIfTI file contains 4D data (e.g., time-series or multi-volume data), a slider lets users navigate through different timepoints.

- **User Controls**:
  - Dropdown menus to select the viewing axis and color map.
  - Sliders for selecting specific slices and timepoints.

- **Output**:
  - The visualization is rendered as an interactive plot using Plotly.

Getting Started
---------------

Run the script from the command line, providing the NIfTI file path as an argument:

.. code-block:: bash

  axsi-nifti-viewer --nifti-file /path/to/your/file.nii.gz

or if you use poetry:

.. code-block:: bash

  poetry run axsi-nifti-viewer --nifti-file /path/to/your/file.nii.gz

Example:

.. code-block:: bash

  axsi-nifti-viewer --nifti-file example_data/pasi.nii.gz

The app runs at http://127.0.0.1:8050 by default, displaying the interactive visualization.