Installation Instructions using Poetry
======================================

This guide explains how to install AxSI using Poetry for dependency management and virtual environment creation.

**Important Prerequisites**: Before proceeding, you must manually install R, Python, and Poetry on your system.

*   **R version 4.4.2**:
    Download and install R from the [CRAN (Comprehensive R Archive Network) website](https://cran.r-project.org/).
    Ensure R is added to your system's PATH.

*   **Python version 3.12** (or a compatible version >=3.10 if AxSI supports it):
    Download and install Python from the [official Python website](https://www.python.org/downloads/).
    Ensure Python is added to your system's PATH. We will specify Python 3.12 for Poetry to use.

*   **Poetry**:
    If you don't have Poetry installed, you can install it by following the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation). A common method is:

    .. code-block:: bash

       # On Linux, macOS, or WSL
       curl -sSL https://install.python-poetry.org | python3 -

       # Or using pip (ensure pip is for Python 3)
       # pip install poetry

    Verify your Poetry installation:
    .. code-block:: bash

       poetry --version

    Add Poetry's bin directory to your PATH if it wasn't done automatically by the installer.

Step 1: Create and Initialize Your AxSI Project
------------------------------------------------

1.  **Create a new directory for your AxSI project and navigate into it**:

    .. code-block:: bash

       mkdir axsi-poetry-project
       cd axsi-poetry-project

2.  **Initialize a new Poetry project**:
    This command creates a `pyproject.toml` file that will manage your project's dependencies. The `--no-interaction` flag accepts default values.

    .. code-block:: bash

       poetry init --no-interaction

3.  **Configure Poetry to use your desired Python version**:
    Ensure Python 3.12 (or the specific version you installed) is available on your system. Poetry will create or use a virtual environment with this Python version.

    .. code-block:: bash

       poetry env use python3.12
       # If 'python3.12' is not found, you might need to provide the full path
       # to the python3.12 executable, or ensure it's in your PATH.
       # Example: poetry env use /usr/bin/python3.12

    Poetry will automatically create and manage a virtual environment for this project.

Step 2: Add AxSI and Install Dependencies
------------------------------------------

4.  **Add the `axsi` package to your project**:
    Poetry will resolve `axsi` and its Python dependencies, and install them into the project's virtual environment.

    .. code-block:: bash

       poetry add axsi

5.  **Install AxSI-specific R components**:
    The `axsi` package provide a command to install necessary R packages: **'quadprog'** and **'minpack.lm'**.

    **Note**: R version 4.4.2 must already be installed on your system as per the prerequisites.

    .. code-block:: bash

       poetry run axsi-install-r

Step 3: Verify Installation
----------------------------

6.  **Check the Python version managed by Poetry**:

    .. code-block:: bash

       poetry run python --version

    This should output the Python version you configured (e.g., Python 3.12.x).

7.  **Check the R version (system-wide)**:

    .. code-block:: bash

       R --version

    This should output information including `R version 4.4.2`.

8.  **Optional: Run AxSI tests**:
    AxSI comes with a test suite, you can run it using:

    .. code-block:: bash

       poetry run axsi-run-tests

Execution
---------

To run the main AxSI script (e.g., **axsi-main.py**), use `poetry run`:

.. code-block:: bash

   poetry run axsi-main -h

Using `poetry run` ensures that the script is executed within the correct virtual environment where `axsi` and its dependencies are installed. You do not need to manually activate the virtual environment; Poetry handles this when you use `poetry run`.