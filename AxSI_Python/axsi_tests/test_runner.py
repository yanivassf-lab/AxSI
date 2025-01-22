#!/usr/bin/env python

import sys
import pytest
import os
import axsi_tests # your installed package

def run_tests():
    # Get the directory of the axsi_test package
    tests_dir = os.path.dirname(axsi_tests.__file__)

    if not os.path.exists(tests_dir):
        print(f"Tests directory not found at {tests_dir}")
        sys.exit(1)

    # Run pytest on that directory
    retcode = pytest.main(["-s", tests_dir])
    #     retcode = pytest.main(["--tb=short", "-q", "."])
    sys.exit(retcode)

if __name__ == "__main__":
    run_tests()
