
import argparse
import os
import json
from src.v1_0_0 import V1_0_0
from src.v2_0_0 import V2_0_0

VERSION_MAP = {
    "v1.0.0": V1_0_0,
    "v1.1.0": V1_0_0,
    "v1.1.1": V1_0_0,
    "v1.1.2": V1_0_0,
    "v1.1.3": V1_0_0,
    "v1.1.4": V1_0_0,
    "v1.1.5": V1_0_0,
    "v1.1.6": V1_0_0,
    "v1.1.7": V1_0_0,
    "v2.0.0": V2_0_0
}

def main():

    parser = argparse.ArgumentParser(
        description="Preprocessor for FLORIS input.",
        prog="preFLORIS"
    )
    parser.add_argument(
        "-i",
        dest="input_file",
        type=str,
        required=False,
        help="Existing input file to convert."
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=str,
        required=False,
        help="Name of the requested output file."
    )
    args = parser.parse_args()

    # If no input file, start with the defaults in v1.0.0
    if not args.input_file:
        starting_version = V1_0_0()

    elif args.input_file:
        with open(args.input_file) as jsonfile:
            data = json.load(jsonfile)
            if "floris_version" not in data:
                raise ValueError("Given input file does not contain a FLORIS.")
            elif data["floris_version"] >= "v2.0.0":
                raise ValueError("The given input file version is up to date with or newer than the latest supported verion.")
            elif data["floris_version"] not in VERSION_MAP:
                raise ValueError("Given FLORIS version is not currently supported.")
            else:
                starting_version = VERSION_MAP[data["floris_version"]](
                    turbine_dict = data.pop("turbine"),
                    wake_dict = data.pop("wake"),
                    farm_dict = data.pop("farm"),
                    meta_dict = data,
                )
    
    ending_version = V2_0_0(
        starting_version.meta_dict,
        starting_version.turbine_dict,
        starting_version.wake_dict,
        starting_version.farm_dict
    )
    if not args.output_file:
        output_filename = ending_version.version_string + ".json"
    else:
        output_filename = args.output_file
    ending_version.export(filename=output_filename)

if __name__=="__main__":

    main()
