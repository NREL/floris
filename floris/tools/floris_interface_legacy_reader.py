# Copyright 2021 NREL
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from floris.tools import FlorisInterface


class FlorisInterfaceLegacyV2(FlorisInterface):
    """
    FlorisInterface_legacy_v24 provides a wrapper around FlorisInterface
    which enables compatibility of the class with legacy floris v2.4 input
    files. The user can simply pass this class the path to a legacy v2.4
    floris input file to this class and it'll convert it to a v3.0-compatible
    input dictionary and load the floris v3.0 object.

    After successfully loading the v3.0 Floris object, you can export the
    input file using: fi.floris.to_file("converted_input_file_v3.yaml").
    An example of such a use case is demonstrated at the end of this file.

    If you would like to manually convert the input dictionary without first
    loading it in FLORIS, or if somehow the code fails to automatically
    convert the input file to v3, you should follow the following steps:
      1. Load the legacy v2.4 input floris JSON file as a dictionary
      2. Pass the v2.4 dictionary to `_convert_v24_dictionary_to_v3(...)`.
         That will return a v3.0-compatible input dictionary and a turbine
         dictionary.
      3. Save the converted configuration file to a YAML or JSON file.

      For example:

        import json, yaml
        from floris.tools.floris_interface_legacy_reader import (
            _convert_v24_dictionary_to_v3
        )

        with open(<path_to_legacy_v24_input_file.json>) as legacy_dict_file:
            configuration_v2 = json.load(legacy_dict_file)
        fi_dict, turb_dict = _convert_v24_dictionary_to_v3(configuration_v2)
        with open(r'fi_input_file_v3.yaml', 'w') as file:
            yaml.dump(fi_dict, file)
        with open(r'turbine_input_file_v3.yaml', 'w') as file:
            yaml.dump(turb_dict, file)

    Args:
        configuration (:py:obj:`dict`): The legacy v2.4 Floris configuration
            dictionary or the file path to the JSON file.
    """

    def __init__(self, configuration: dict | str | Path, het_map=None):

        if not isinstance(configuration, (str, Path, dict)):
            raise TypeError("The Floris `configuration` must of type 'dict', 'str', or 'Path'.")

        print("Importing and converting legacy floris v2.4 input file...")
        if isinstance(configuration, (str, Path)):
            with open(configuration) as legacy_dict_file:
                configuration = json.load(legacy_dict_file)

        dict_fi, dict_turbine = _convert_v24_dictionary_to_v3(configuration)
        super().__init__(dict_fi, het_map=het_map)  # Initialize full class

        # Now overwrite turbine types
        n_turbs = len(self.layout_x)
        self.reinitialize(turbine_type=[dict_turbine] * n_turbs)


def _convert_v24_dictionary_to_v3(dict_legacy):
    """
    Converts a v2.4 floris input dictionary file to a v3.0-compatible
    dictionary. See detailed instructions in the class
    FlorisInterface_legacy_v24.

    Args:
        dict_legacy (dict): Input dictionary in legacy floris v2.4 format.

    Returns:
        dict_floris (dict): Converted dictionary containing the floris input
        settings in v3.0-compatible format.
        dict_turbine (dict): A converted dictionary containing the turbine
        settings in v3.0-compatible format.
    """
    # Simple entries that can just be copied over
    dict_floris = {}  # Output dictionary
    dict_floris["name"] = dict_legacy["name"] + " (auto-converted to v3)"
    dict_floris["description"] = dict_legacy["description"]
    dict_floris["floris_version"] = "v3.0 (converted from legacy format v2)"
    dict_floris["logging"] = dict_legacy["logging"]

    dict_floris["solver"] = {
        "type": "turbine_grid",
        "turbine_grid_points": dict_legacy["turbine"]["properties"]["ngrid"],
    }

    fp = dict_legacy["farm"]["properties"]
    tp = dict_legacy["turbine"]["properties"]
    dict_floris["farm"] = {
        "layout_x": fp["layout_x"],
        "layout_y": fp["layout_y"],
        "turbine_type": ["nrel_5MW"]  # Placeholder
    }

    ref_height = fp["specified_wind_height"]
    if ref_height < 0:
        ref_height = tp["hub_height"]

    dict_floris["flow_field"] = {
        "air_density": fp["air_density"],
        "reference_wind_height": ref_height,
        "turbulence_intensity": fp["turbulence_intensity"][0],
        "wind_directions": [fp["wind_direction"]],
        "wind_shear": fp["wind_shear"],
        "wind_speeds": [fp["wind_speed"]],
        "wind_veer": fp["wind_veer"],
    }

    wp = dict_legacy["wake"]["properties"]
    velocity_model = wp["velocity_model"]
    velocity_model_str = velocity_model
    if velocity_model == "gauss_legacy":
        velocity_model_str = "gauss"
    deflection_model = wp["deflection_model"]
    turbulence_model = wp["turbulence_model"]
    wdp = wp["parameters"]["wake_deflection_parameters"][deflection_model]
    wvp = wp["parameters"]["wake_velocity_parameters"][velocity_model]
    wtp = wp["parameters"]["wake_turbulence_parameters"][turbulence_model]
    dict_floris["wake"] = {
        "model_strings": {
            "combination_model": wp["combination_model"],
            "deflection_model": deflection_model,
            "turbulence_model": turbulence_model,
            "velocity_model": velocity_model_str,
        },
        "enable_secondary_steering": wdp["use_secondary_steering"],
        "enable_yaw_added_recovery": wvp["use_yaw_added_recovery"],
        "enable_transverse_velocities": wvp["calculate_VW_velocities"],
    }

    # Copy over wake velocity parameters and remove unnecessary parameters
    velocity_subdict = copy.deepcopy(wvp)
    c = ["calculate_VW_velocities", "use_yaw_added_recovery", "eps_gain"]
    for ci in [ci for ci in c if ci in velocity_subdict.keys()]:
        velocity_subdict.pop(ci)

    # Copy over wake deflection parameters and remove unnecessary parameters
    deflection_subdict = copy.deepcopy(wdp)
    c = ["use_secondary_steering"]
    for ci in [ci for ci in c if ci in deflection_subdict.keys()]:
        deflection_subdict.pop(ci)

    # Copy over wake turbulence parameters and remove unnecessary parameters
    turbulence_subdict = copy.deepcopy(wtp)

    # Save parameter settings to wake dictionary
    dict_floris["wake"]["wake_velocity_parameters"] = {
        velocity_model_str: velocity_subdict
    }
    dict_floris["wake"]["wake_deflection_parameters"] = {
        deflection_model: deflection_subdict
    }
    dict_floris["wake"]["wake_turbulence_parameters"] = {
        turbulence_model: turbulence_subdict
    }

    # Finally add turbine information
    dict_turbine = {
        "turbine_type": dict_legacy["turbine"]["name"],
        "generator_efficiency": tp["generator_efficiency"],
        "hub_height": tp["hub_height"],
        "pP": tp["pP"],
        "pT": tp["pT"],
        "rotor_diameter": tp["rotor_diameter"],
        "TSR": tp["TSR"],
        "power_thrust_table": tp["power_thrust_table"],
        "ref_density_cp_ct": 1.225 # This was implicit in the former input file
    }

    return dict_floris, dict_turbine


if __name__ == "__main__":
    """
    When this file is ran as a script, it'll convert a legacy FLORIS v2.4
    legacy input file (.json) to a v3.0-compatible input file (.yaml).
    Please specify your input and output paths accordingly, and it will
    produce the necessary file.
    """
    import argparse

    # Parse the input arguments
    description = "Converts a FLORIS v2.4 input file to a FLORIS v3 compatible input file.\
        The file format is changed from JSON to YAML and all inputs are mapped, as needed."

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i",
                        "--input-file",
                        nargs=1,
                        required=True,
                        help="Path to the legacy input file")
    parser.add_argument("-o",
                        "--output-file",
                        nargs="?",
                        default=None,
                        help="Path to write the output file")
    args = parser.parse_args()

    # Specify paths
    legacy_json_path = Path(args.input_file[0])
    if args.output_file:
        floris_yaml_output_path = args.output_file
    else:
        floris_yaml_output_path = legacy_json_path.stem + ".yaml"

    # Load legacy input .json file into V3 object
    fi = FlorisInterfaceLegacyV2(legacy_json_path)

    # Create output directory and save converted input file
    fi.floris.to_file(floris_yaml_output_path)

    print(f"Converted file saved to: {floris_yaml_output_path}")
