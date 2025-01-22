Installation Instructions using Conda (Recommended)
===================================================

Step 1: Install Miniconda
---------------------------

1. **Download the Miniconda installer**:
   Open your terminal and run the following command to download the Miniconda installer:
   .. code-block:: bash

      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

2. **Make the installer executable**:
   Change the permissions of the downloaded script to make it executable:
   .. code-block:: bash

      chmod +x Miniconda3-latest-Linux-x86_64.sh

3. **Run the installer**:
   Execute the installer script to install Miniconda:
   .. code-block:: bash

      ./Miniconda3-latest-Linux-x86_64.sh

   Follow the on-screen instructions to complete the installation. You may need to agree to the license terms and specify the installation location.

Step 2: Configure Conda
------------------------

4. **Add the conda-forge channel**:
   After installing Miniconda, configure it to use the conda-forge channel, which provides additional packages:
   .. code-block:: bash

      conda config --add channels conda-forge

Step 3: Create a New Conda Environment
----------------------------------------

5. **Create a new conda environment**:
   Create a new environment named `axsi` with R version 4.4.2 and Python version 3.12:
   .. code-block:: bash

      conda create --name axsi -c conda-forge r-base=4.4.2 python=3.12

6. **Activate the new environment**:
   Activate the newly created environment:
   .. code-block:: bash

      conda activate axsi

Step 4: Verify Installation
----------------------------

7. **Check the R installation**:
   Verify that R is installed correctly by running:
   .. code-block:: bash

      which R

8. **Check the Python installation**:
   Verify that Python is installed correctly by running:
   .. code-block:: bash

      which python

9. **Check Python version**:
   Confirm the installed version of Python:
   .. code-block:: bash

      python --version

10. **Check R version**:
    Confirm the installed version of R:
    .. code-block:: bash

      R --version

Step 5: Install AxSI Packages
-------------------------------

11. **Install the axsi package**:
    Install the `axsi` package using pip:
    .. code-block:: bash

      pip install axsi

12. **Install AxSI-specific R components**:
    The `axsi` package provide a command to install necessary R packages: **'quadprog'** and **'minpack.lm'**.

    **Note**: R version 4.4.2 already installed on your system in the creation of the environment.

    .. code-block:: bash

       axsi-install-r

13. **Optional: Run AxSI tests**:
    AxSI comes with a test suite, you can run it using:

    .. code-block:: bash

       axsi-run-tests

Execution
---------

The main script shipped with this project is **axsi-main**, see its options by running:

.. code-block::

   axsi-main -h