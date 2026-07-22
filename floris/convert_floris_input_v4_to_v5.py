import sys
from pathlib import Path

import yaml


"""
This script is intended to be called with an argument and converts a floris input
yaml file specified for FLORIS v4 to one specified for FLORIS v5.

Usage:
python convert_floris_input_v4_to_v5.py <path/to/floris_input>.yaml

The resulting floris input file is placed in the same directory as the original yaml,
and is appended _v5.
"""


def ignore_include(loader, node):
    # Parrot back the !include tag
    return node.tag + " " + node.value

def check_wake_model_compatibility(wake_v4):

    """
    Checks if the wake model specified in the v4 input file is compatible with v5.
    If not, raises an exception.

    Args:
        wake_v4 (dict): The wake model dictionary from the v4 input file.
    """
    velocity_model = wake_v4["model_strings"]["velocity_model"]
    deflection_model = wake_v4["model_strings"]["deflection_model"]
    turbulence_model = wake_v4["model_strings"]["turbulence_model"]

    def deflection_model_warning(deflection_model, velocity_model, valid_deflection_model):
        return (
            f"Deflection model '{deflection_model}' is not compatible with velocity model "
            f"'{velocity_model}' in FLORIS v5. Only the '{valid_deflection_model}' deflection "
            f"model is compatible with the '{velocity_model}' velocity model."
        )

    def turbulence_model_warning(turbulence_model, velocity_model, valid_turbulence_model):
        return (
            f"Turbulence model '{turbulence_model}' is not compatible with velocity model "
            f"'{velocity_model}' in FLORIS v5. Only the '{valid_turbulence_model}' turbulence "
            f"model is compatible with the '{velocity_model}' velocity model."
        )

    if velocity_model == "turbopark":
        print("The original TurbOPark velocity model is not compatible with FLORIS v5. "
              "Please use the TurbOParkGauss velocity model instead.")
    elif velocity_model == "gauss":
        if deflection_model != "gauss":
            raise Exception(deflection_model_warning(deflection_model, velocity_model, "gauss"))
        if turbulence_model != "crespo_hernandez":
            raise Exception(
                turbulence_model_warning(turbulence_model, velocity_model, "crespo_hernandez")
            )
    elif velocity_model == "jensen":
        if deflection_model != "jimenez":
            raise Exception(deflection_model_warning(deflection_model, velocity_model, "jimenez"))
        if turbulence_model != "crespo_hernandez":
            raise Exception(
                    turbulence_model_warning(turbulence_model, velocity_model, "crespo_hernandez")
            )
    elif velocity_model == "empirical_gauss":
        if deflection_model != "empirical_gauss":
            raise Exception(
                deflection_model_warning(deflection_model, velocity_model, "empirical_gauss")
            )
        if turbulence_model != "wake_induced_mixing":
            raise Exception(
                turbulence_model_warning(turbulence_model, velocity_model, "wake_induced_mixing")
            )
    elif velocity_model == "cc":
        if deflection_model != "gauss":
            raise Exception(deflection_model_warning(deflection_model, velocity_model, "gauss"))
    elif velocity_model == "turboparkgauss":
        if deflection_model != "none":
            raise Exception(deflection_model_warning(deflection_model, velocity_model, "none"))
        if turbulence_model != "none":
            raise Exception(
                turbulence_model_warning(turbulence_model, velocity_model, "none")
            )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception(
            "Usage: python convert_floris_input_v4_to_v5.py <path/to/floris_input>.yaml"
        )

    # Set the yaml loader to ignore the !include tag
    yaml.SafeLoader.add_constructor("!include", ignore_include)

    input_yaml = sys.argv[1]

    # Handling the path and new filename
    input_path = Path(input_yaml)
    split_input = input_path.parts
    [filename_v4, extension] = split_input[-1].split(".")
    filename_v5 = filename_v4 + "_v5"
    split_output = list(split_input[:-1]) + [filename_v5 + "." + extension]
    output_path = Path(*split_output)

    # Load existing v4 model
    with open(input_yaml, "r") as file:
        floris_input_dict = yaml.safe_load(file)

    # Reorganize wake model and disallow combinations that are not supported in v5
    wake_v4 = floris_input_dict["wake"]
    check_wake_model_compatibility(wake_v4)
    velocity_model_parameters_v4 = (
        wake_v4["wake_velocity_parameters"]
        [wake_v4["model_strings"]["velocity_model"]]
    )
    deflection_model_parameters_v4 = (
        wake_v4["wake_deflection_parameters"]
        [wake_v4["model_strings"]["deflection_model"]]
    )
    turbulence_model_parameters_v4 = (
        wake_v4["wake_turbulence_parameters"]
        [wake_v4["model_strings"]["turbulence_model"]]
    )

    wake_v5 = {
        "model": wake_v4["model_strings"]["velocity_model"],
        "parameters": (
            turbulence_model_parameters_v4 |
            deflection_model_parameters_v4 |
            velocity_model_parameters_v4
        ),
        "combination_model": wake_v4["model_strings"]["combination_model"],
    }

    floris_input_dict["wake"] = wake_v5
    floris_input_dict["floris_version"] = "v5"

    with open(output_path, "w") as file:
        yaml.dump(floris_input_dict, file, sort_keys=False)

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
