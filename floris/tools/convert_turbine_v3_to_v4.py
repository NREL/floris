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

"""
This script is intended to be called with an argument and converts a turbine
yaml file specified for FLORIS v3 to one specified for FLORIS v4.

Usage:
python convert_turbine_yaml_v3_to_v4.py <path/to/turbine>.yaml

The resulting turbine is placed in the same directory as the original yaml,
and is appended _v4.
"""

import sys
from pathlib import Path

from floris.simulation.turbine import build_cosine_loss_turbine_dict, check_smooth_power_curve
from floris.utilities import load_yaml


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Usage: python convert_turbine_yaml_v3_to_v4.py <path/to/turbine>.yaml")

    input_yaml = sys.argv[1]

    # Handling the path and new filename
    input_path = Path(input_yaml)
    split_input = input_path.parts
    [filename_v3, extension] = split_input[-1].split(".")
    filename_v4 = filename_v3 + "_v4"
    split_output = list(split_input[:-1]) + [filename_v4+"."+extension]
    output_path = Path(*split_output)

    # Load existing v3 model
    v3_turbine_dict = load_yaml(input_yaml)

    # Split into components expected by build_turbine_dict
    power_thrust_table = v3_turbine_dict["power_thrust_table"]
    power_thrust_table["power_coefficient"] = power_thrust_table["power"]
    power_thrust_table["thrust_coefficient"] = power_thrust_table["thrust"]
    power_thrust_table.pop("power")
    power_thrust_table.pop("thrust")

    valid_properties = [
        "generator_efficiency",
        "hub_height",
        "pP",
        "pT",
        "rotor_diameter",
        "TSR",
        "ref_air_density",
        "ref_tilt"
    ]

    turbine_properties = {k:v for k,v in v3_turbine_dict.items() if k in valid_properties}
    turbine_properties["ref_air_density"] = v3_turbine_dict["ref_density_cp_ct"]
    if "ref_tilt_cp_ct" in v3_turbine_dict:
        turbine_properties["ref_tilt"] = v3_turbine_dict["ref_tilt_cp_ct"]

    # Convert to v4 and print new yaml
    v4_turbine_dict = build_cosine_loss_turbine_dict(
        power_thrust_table,
        v3_turbine_dict["turbine_type"],
        output_path,
        **turbine_properties
    )

    if not check_smooth_power_curve(
        v4_turbine_dict["power_thrust_table"]["power"],
        tolerance=0.001
    ):
        print(
            "Non-smoothness detected in output power curve. ",
            "Check above-rated power in generated v4 yaml file."
        )
