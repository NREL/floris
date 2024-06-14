import sys
from pathlib import Path

import yaml


"""
This script is intended to be called with an argument and converts a floris input
yaml file specified for FLORIS v3 to one specified for FLORIS v4.

Usage:
python convert_floris_input_v3_to_v4.py <path/to/floris_input>.yaml

The resulting floris input file is placed in the same directory as the original yaml,
and is appended _v4.
"""


def ignore_include(loader, node):
    # Parrot back the !include tag
    return node.tag + " " + node.value


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception(
            "Usage: python convert_floris_input_v3_to_v4.py <path/to/floris_input>.yaml"
        )

    # Set the yaml loader to ignore the !include tag
    yaml.SafeLoader.add_constructor("!include", ignore_include)

    input_yaml = sys.argv[1]

    # Handling the path and new filename
    input_path = Path(input_yaml)
    split_input = input_path.parts
    [filename_v3, extension] = split_input[-1].split(".")
    filename_v4 = filename_v3 + "_v4"
    split_output = list(split_input[:-1]) + [filename_v4 + "." + extension]
    output_path = Path(*split_output)

    # Load existing v3 model
    with open(input_yaml, "r") as file:
        v3_floris_input_dict = yaml.safe_load(file)
    v4_floris_input_dict = v3_floris_input_dict.copy()

    # Change turbulence_intensity field to turbulence_intensities as list
    if "turbulence_intensities" in v3_floris_input_dict["flow_field"]:
        if "turbulence_intensity" in v3_floris_input_dict["flow_field"]:
            del v4_floris_input_dict["flow_field"]["turbulence_intensity"]
    elif "turbulence_intensity" in v3_floris_input_dict["flow_field"]:
        v4_floris_input_dict["flow_field"]["turbulence_intensities"] = [
            v3_floris_input_dict["flow_field"]["turbulence_intensity"]
        ]
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

    # Write the new v4 model to a new file, note that the in order to ignore the !include tag
    # it is wrapped in single quotes by the ignore include/load/dump sequence and these will
    # need to be removed in the next block of code
    yaml.dump(v4_floris_input_dict, open(output_path, "w"), sort_keys=False)

    # Open the output file and loop through line by line
    # if a line contains the substring !include, then strip all
    # occurrences of ' from the line to remove the extra single quotes
    # added by the ignore include/load/dump sequence
    temp_output_path = output_path.with_name("temp.yaml")
    with open(temp_output_path, "w") as file:
        with open(output_path, "r") as f:
            for line in f:
                if "!include" in line:
                    line = line.replace("'", "")
                file.write(line)

    # Move the temp file to the output file
    temp_output_path.replace(output_path)

    print(output_path, "created.")
