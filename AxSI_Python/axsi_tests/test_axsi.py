import filecmp
from pathlib import Path

import pytest

from axsi.cli import axsi_main

BASE_DIR = Path(__file__).parent.resolve()


@pytest.mark.parametrize("test_args", [
    {
        # "subj_folder": Path(BASE_DIR, "test_data", "output_files"),
        "run_name": "test_run",
        "bval": Path(BASE_DIR, "test_data", "input_files", "voxels_50_52_bval.txt"),
        "bvec": Path(BASE_DIR, "test_data", "input_files", "voxels_50_52_bvec.txt"),
        "data": Path(BASE_DIR, "test_data", "input_files", "voxels_50_52_data.nii"),
        "mask": Path(BASE_DIR, "test_data", "input_files", "voxels_50_52_mask.nii"),
        # other args can be overridden if desired
    },
])
def test_axsi_run(tmp_path, test_args, capsys):

    test_args["subj_folder"] = tmp_path

    # Run main with test_args
    with capsys.disabled():
        axsi_main.main(test_args=test_args)

    # Now check outputs in tmp_path or args.subj_folder as needed
    # Example:
    expected_dir = Path(BASE_DIR, "test_data", "output_files", "expected_outputs")
    output_dir = Path(test_args["subj_folder"]) / test_args["run_name"]  # Assuming outputs go here

    for expected_file in expected_dir.glob("*"):
        output_file = output_dir / expected_file.name
        if expected_file.name != "run_info.log" and expected_dir.name != "__init__.py":
            assert output_file.exists(), f"Missing output file {output_file}"
            assert filecmp.cmp(output_file, expected_file, shallow=False), f"File {output_file} differs from expected."


# Run test:
# =========
# pip install -e .
# pytest