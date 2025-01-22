import os
import subprocess
import sys

from setuptools import setup, find_packages
from setuptools.command.install import install

#
# class CustomInstallCommand(install):
#     """Custom installation command to check R and install R packages."""
#
#     def run(self):
#         # Check if R is installed
#         try:
#             script_path = os.path.join(os.path.dirname(__file__), 'src', 'axsi','cli', 'axsi_install_r.py')
#             process = subprocess.Popen(["python", script_path], stdout=sys.stdout, stderr=sys.stderr, text=True)
#             process.communicate()
#             if process.returncode != 0:
#                 print(f"Error: Process returned non-zero exit code {process.returncode}")
#                 sys.exit(process.returncode)
#         except Exception as e:
#             raise Exception(e)
#         install.run(self)

with open("README.md", "r") as fh:
    long_description = fh.read()

requires = [
    'numpy==1.26.4',
    'scipy==1.14.1',
    'scikit-learn==1.5.2',
    'psutil==6.1.1',
    'nibabel==5.3.0',
    'dipy==1.9.0',
    'lsqAxSI',  # Hila Gast's LSQ function
    'rpy2==3.5.11',
    'ecos==2.0.14',
    'cvxpy==1.5.3',
    'gurobipy==12.0.0',
    'dash==2.18.2',
    'plotly==5.24.1',
    'pandas==2.2.3',
    'pytest==8.0'
]

setup(
    # required
    name="AxSI",
    version="0.0.17",
    # optional
    author="Hila Gast",
    author_email="",
    description="AxSI for MRI data",
    long_description=long_description,  # This is shown on the package detail package on the Pypi
    long_description_content_type="text/markdown",  # text/plain, text/x-rst (for reStructuredText), text/markdown
    url="",  # github

    classifiers=[  # Meta-data. Examples: https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=requires,  # Install dependencies
    entry_points={
        "console_scripts": [
            "axsi-main=axsi.cli.axsi_main:main",
            "axsi-nifti-viewer=axsi.cli.axsi_nifti_viewer:main",
            "axsi-install-r=axsi.cli.axsi_install_r:main",
            "axsi-run-tests=axsi_tests.test_runner:run_tests",
        ],
    },

    # Explicitly list all packages (replace/add your real packages)
    packages=[
        "axsi",
        "axsi.cli",
        "axsi_tests"  # Include test package if you want tests installed
    ],

    package_dir={
        "axsi": "src/axsi",
        "axsi.cli": "src/axsi/cli",
        "axsi.tests": "axsi_tests"
    },

    # Include non-code files in packages as specified in MANIFEST.in
    # include_package_data=True,

    data_files=[('docs', ['docs/usage.rst']), ],
    include_package_data=True,
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.rst"],
    },
    # cmdclass={
    #     'install': CustomInstallCommand,  # Use the custom install command for R packages
    # },


    # # For sdist only:
    # # True: include only files from MANIFEST.in file (not from package_data).
    # # False: include files from MANIFEST.in file AND from package_data
    # include_package_data=False,
    #
    # tests_require=requires + ['nose2'], # Install requirements when you run: python setup.py test
    # test_suite='nose2.collector.collector',
)
