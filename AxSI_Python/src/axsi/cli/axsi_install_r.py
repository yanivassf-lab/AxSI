#!/usr/bin/env python

import os
import subprocess
import sys


def install_r_package(package_name):
    """Install an R package using the system R."""
    try:
        # Check if the package is already installed by running R's package test
        result = subprocess.run(
            ["R", "-e", f"if (!require('{package_name}', character.only = TRUE)) quit(status = 1)"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"Package {package_name} not found. Installing...")
            # Set CRAN mirror and install the package
            install_result = subprocess.run(
                ["R", "-e", f"install.packages('{package_name}', repos='https://cran.r-project.org')"],
                capture_output=True, text=True
            )

            if install_result.returncode != 0:
                print(f"Error installing package {package_name}.")
                print(f"Output: {install_result.stdout}")
                print(f"Error: {install_result.stderr}")
                sys.exit(1)
            else:
                print(f"Package {package_name} successfully installed.")
        else:
            print(f"R package {package_name} is already installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error during package installation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def is_r_installed():
    """Check if R is installed and accessible in PATH."""
    try:
        subprocess.run(["R", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except FileNotFoundError:
        return False


def get_os_specific_instructions():
    """Provide instructions for adding R to PATH based on the operating system."""
    os_name = os.name
    if os_name == "nt":  # Windows
        instructions = (
            "1. Locate the folder where R is installed (e.g., C:\\Program Files\\R\\R-x.x.x\\bin).\n"
            "2. Add this folder to the PATH environment variable:\n"
            "   - Search for 'Environment Variables' in the Start Menu.\n"
            "   - Click 'Environment Variables'.\n"
            "   - Edit the 'Path' variable and add the R folder path.\n"
            "3. Restart your terminal or IDE."
        )
    elif os_name == "posix":  # macOS or Linux
        instructions = (
            "1. Locate the folder where R is installed (use `which R` to find it, e.g., /usr/local/bin/R).\n"
            "2. Add this folder to the PATH environment variable:\n"
            "   - Edit your shell configuration file (e.g., ~/.bashrc, ~/.zshrc, or ~/.bash_profile).\n"
            "   - Add the line: export PATH=/path/to/R:$PATH\n"
            "   - Replace '/path/to/R' with the folder containing the R executable.\n"
            "3. Run `source ~/.bashrc` or restart your terminal."
        )
    else:
        instructions = "Unsupported operating system. Please consult your R installation documentation."
    return instructions


def ensure_r_installed():
    """Ensure R is installed and provide instructions if not."""
    if not is_r_installed():
        print("Error: R is not installed or not found in your system's PATH.")
        print("\nTo install R, visit: https://cran.r-project.org/\n")
        print("After installing R, follow these instructions to add R to your PATH:\n")
        print(get_os_specific_instructions())
        sys.exit(1)  # Exit to prevent further setup steps

def main():
    ensure_r_installed()
    install_r_package('quadprog')
    install_r_package('minpack.lm')
    # install_r_package('Rserve') # for Rserver - not in use, it is very slow


if __name__ == "__main__":
    main()
