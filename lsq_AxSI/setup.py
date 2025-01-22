import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requires = ['numpy', 'scipy', 'scikit-learn']

setuptools.setup(
    # required
    name="lsqAxSI",
    version="0.0.6",
    # optional
    author="Hila Gast",
    author_email="",
    description="Nonlinear least-squares optimization for AxSI",
    long_description=long_description,  # This is shown on the package detail package on the Pypi
    long_description_content_type="text/markdown",  # text/plain, text/x-rst (for reStructuredText), text/markdown
    url="",  # github

    classifiers=[  # Meta-data. Examples: https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=requires,  # Install dependencies
    scripts=[
        'scripts/lsq_AxSI.py',
    ],

    packages=setuptools.find_packages(),

    data_files=[('docs', ['docs/usage.rst']), ],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.rst"],
    },
    #
    # # For sdist only:
    # # True: include only files from MANIFEST.in file (not from package_data).
    # # False: include files from MANIFEST.in file AND from package_data
    # include_package_data=False,
    #
    # tests_require=requires + ['nose2'], # Install requirements when you run: python setup.py test
    # test_suite='nose2.collector.collector',
)
