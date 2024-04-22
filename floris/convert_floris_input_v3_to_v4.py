
import sys
from pathlib import Path

import yaml

from floris.utilities import load_yaml


"""
This script is intended to be called with an argument and converts a floris input
yaml file specified for FLORIS v3 to one specified for FLORIS v4.

Usage:
python convert_floris_input_v3_to_v4.py <path/to/floris_input>.yaml

The resulting floris input file is placed in the same directory as the original yaml,
and is appended _v4.
"""


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception(
            "Usage: python convert_floris_input_v3_to_v4.py <path/to/floris_input>.yaml"
        )

    input_yaml = sys.argv[1]

    # Handling the path and new filename
    input_path = Path(input_yaml)
    split_input = input_path.parts
    [filename_v3, extension] = split_input[-1].split(".")
    filename_v4 = filename_v3 + "_v4"
    split_output = list(split_input[:-1]) + [filename_v4+"."+extension]
    output_path = Path(*split_output)

    # Load existing v3 model
    v3_floris_input_dict = load_yaml(input_yaml)
    v4_floris_input_dict = v3_floris_input_dict.copy()

    # Change turbulence_intensity field to turbulence_intensities as list
    if "turbulence_intensities" in v3_floris_input_dict["flow_field"]:
        if "turbulence_intensity" in v3_floris_input_dict["flow_field"]:
            del v4_floris_input_dict["flow_field"]["turbulence_intensity"]
    elif "turbulence_intensity" in v3_floris_input_dict["flow_field"]:
        v4_floris_input_dict["flow_field"]["turbulence_intensities"] = (
            [v3_floris_input_dict["flow_field"]["turbulence_intensity"]]
        )
        del v4_floris_input_dict["flow_field"]["turbulence_intensity"]

    # Change multidim_cp_ct velocity model to gauss
    if v3_floris_input_dict["wake"]["model_strings"]["velocity_model"] == "multidim_cp_ct":
        print(
            "multidim_cp_ct velocity model specified. Changing to gauss, "
            + "but note that other velocity models are also compatible with multidimensional "
            + "turbines in FLORIS v4. "
            + "You will also need to convert your multidimensional turbine yaml files and their "
            + "corresponding power/thrust csv files to be compatible with FLORIS v4 and to reflect "
            + " the absolute power curve, rather than the power coefficient curve."
        )
        v4_floris_input_dict["wake"]["model_strings"]["velocity_model"] = "gauss"

    # Add enable_active_wake_mixing field
    v4_floris_input_dict["wake"]["enable_active_wake_mixing"] = False

    yaml.dump(
            v4_floris_input_dict,
            open(output_path, "w"),
            sort_keys=False
        )

    print(output_path, "created.")
