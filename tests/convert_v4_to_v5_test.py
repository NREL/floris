import os
from pathlib import Path

import floris
from floris import FlorisModel


CONVERT_FOLDER = Path(__file__).resolve().parent / "v4_to_v5_converter_test"
FLORIS_FOLDER = Path(floris.__file__).resolve().parent


def test_v4_to_v5_converter():
    # Note certain filenames
    filename_v4_floris = "gch.yaml"
    filename_v5_floris = "gch_v5.yaml"

    # Copy convert scripts from FLORIS_FOLDER to CONVERT_FOLDER
    os.system(f"cp {FLORIS_FOLDER / 'convert_floris_input_v4_to_v5.py'} {CONVERT_FOLDER}")

    # Change directory to the test folder
    os.chdir(CONVERT_FOLDER)

    # Run the converter on the floris file
    os.system(f"python convert_floris_input_v4_to_v5.py {filename_v4_floris}")

    # Now confirm that the converted file can be loaded by FLORIS
    fmodel = FlorisModel(filename_v5_floris)

    # Now confirm this model runs
    fmodel.run()

    # Delete the newly created files to clean up
    os.system(f"rm {filename_v5_floris}")
    os.system("rm convert_floris_input_v4_to_v5.py")
