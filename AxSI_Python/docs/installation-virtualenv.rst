Installation Instructions using Virtual Environment of Python
=============================================================

**Important Prerequisites**: Before proceeding, you must manually install R and Python on your system.
AxSI requires:

*   **R version 4.4.2**: Download and install R from the [CRAN (Comprehensive R Archive Network) website](https://cran.r-project.org/).
*   **Python version 3.12**: Download and install Python from the [official Python website](https://www.python.org/downloads/).

Ensure that both R and Python are correctly installed and added to your system's PATH environment variable so they can be invoked from the command line.

Step 1: Create a Project Directory (Recommended)
-------------------------------------------------

1.  **Create and navigate to a new directory for your AxSI project**:
    This helps keep your project files organized.

    .. code-block:: bash

       mkdir axsi_project
       cd axsi_project

Step 2: Create and Activate a Python Virtual Environment
----------------------------------------------------------

A Python virtual environment will isolate the AxSI installation and its dependencies from your global Python setup.

2.  **Create a new virtual environment**:
    We'll name it `axsi-env`. Make sure you have Python 3.12 installed and available in your PATH before running this.

    .. code-block:: bash

       python3.12 -m venv axsi-env
       # or on some systems, just 'python -m venv axsi-env' if python3.12 is the default python

3.  **Activate the new environment**:

    *   **On Linux or macOS**:
        .. code-block:: bash

           source axsi-env/bin/activate

    *   **On Windows (Command Prompt)**:
        .. code-block:: bat

           axsi-env\Scripts\activate.bat



    Your command prompt should now indicate that the `(axsi-env)` environment is active.

Step 3: Verify Installations
-----------------------------

4.  **Check the Python installation within the virtual environment**:
    Verify that Python is using the environment's interpreter.

    *   **On Linux or macOS**:
        .. code-block:: bash

           which python

        This should point to a Python interpreter inside your `axsi-env` directory.

    *   **On Windows**:
        .. code-block:: bat

           where python

        This should list the Python interpreter inside your `axsi-env` directory first.

5.  **Check Python version**:
    Confirm the installed version of Python.

    .. code-block:: bash

       python --version

    This should output `Python 3.12.x`.

6.  **Check the R installation**:
    Verify that R is accessible from your system PATH.

    *   **On Linux or macOS**:
        .. code-block:: bash

           which R

    *   **On Windows**:
        .. code-block:: bat

           where R

7.  **Check R version**:
    Confirm the installed version of R.

    .. code-block:: bash

       R --version

    This should output information including `R version 4.4.2`.

Step 4: Install AxSI Package
-----------------------------

8.  **Install the axsi package**:
    With the `axsi-env` virtual environment still active, install the `axsi` package using pip.

    .. code-block:: bash

       pip install axsi

9.  **Install AxSI-specific R components**:
    The `axsi` package provide a command to install necessary R packages: **'quadprog'** and **'minpack.lm'**.

    **Note**: R version 4.4.2 must already be installed on your system as per the prerequisites.

    .. code-block:: bash

       axsi-install-r

10. **Optional: Run AxSI tests**:
    AxSI comes with a test suite, you can run it using:

    .. code-block:: bash

       axsi-run-tests

Execution
---------

The main script shipped with this project is **axsi-main**, see its options by running:

.. code-block::

   axsi-main -h

Make sure your virtual environment (`axsi-env`) is activated before running the script.