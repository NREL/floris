# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import json

import numpy as np


class InputReader:
    """
    InputReader parses JSON input files into inputs for the
    :py:class:`~.floris.Floris` class. It handles input validation regarding
    input type, but does not enforce value checking.
    """

    def __init__(self):
        """
        Initializes the parameter types expected in the input data.
        """

        self._valid_objects = ["turbine", "wake", "farm"]

        self._turbine_properties = {
            "rotor_diameter": float,
            "hub_height": float,
            "blade_count": int,
            "pP": float,
            "pT": float,
            "generator_efficiency": float,
            "power_thrust_table": dict,
            "yaw_angle": float,
            "tilt_angle": float,
            "TSR": float,
        }

        self._wake_properties = {
            "velocity_model": str,
            "turbulence_model": str,
            "deflection_model": str,
            "combination_model": str,
            "parameters": dict,
        }

        self._farm_properties = {
            "wind_speed": list,
            "wind_direction": list,
            "turbulence_intensity": list,
            "wind_shear": float,
            "wind_veer": float,
            "air_density": float,
            "layout_x": list,
            "layout_y": list,
            "wind_x": list,
            "wind_y": list,
        }

    def _parseJSON(self, filename):
        """
        Opens the input JSON file and parses the contents into a Python
        dict.

        Args:
            filename (str): Path to the JSON input file.

        Returns:
            dict: The data parsed from the input file.
        """
        with open(filename) as jsonfile:
            data = json.load(jsonfile)
        return data

    def _validate_dict(self, input_dict, type_map):
        """
        Verifies that the expected fields exist in the JSON input file
        and validates the type of the input data by casting the fields
        to appropriate values based on the predefined type maps.

        Args:
            input_dict (dict): Input data.
            type_map (dict): Predefined type-map for type checking inputs;
                structured as {"property": type}.

        Raises:
            KeyError: Missing/invalid key.
            ValueError: Invalid value type.

        Returns:
            dict: Validated and correctly typed input data.
        """
        validated = {}

        # validate the object type
        if "type" not in input_dict:
            raise KeyError("'type' key is required in input")
        if input_dict["type"] not in self._valid_objects:
            raise ValueError(
                "'type' must be one of {}".format(", ".join(self._valid_objects))
            )
        validated["type"] = input_dict["type"]

        # validate the name
        if "name" not in input_dict:
            raise KeyError("'name' key is required in input")
        validated["name"] = input_dict["name"]

        # validate the properties dictionary
        if "properties" not in input_dict:
            raise KeyError("'properties' key is required in input")

        # check every attribute in the predefined type dictionary for existence
        # and proper type in the given inputs
        prop_dict = {}
        properties = input_dict["properties"]
        for element in type_map:
            if element not in properties:
                raise KeyError(
                    "'{}' is required for object type '{}'".format(
                        element, validated["type"]
                    )
                )
            prop_dict[element] = self._cast(type_map[element], properties[element])
        validated["properties"] = prop_dict

        for element in properties:
            if element not in type_map:
                raise KeyError(
                    "'{}' is given but not required for object type '{}'".format(
                        element, validated["type"]
                    )
                )

        return validated

    def _cast(self, typecast, value):
        """
        Casts the input to the given type in `typecast`.

        Args:
            typecast (type): Type class to use for casting.
            value (str): Input to cast to 'typecast'.

        Returns:
            typecast
        """
        return typecast(value)

    def validate_turbine(self, input_dict):
        """
        Checks for the required values and types of input in the given input
        dictionary as required by the :py:obj:`~.turbine.Turbine` object.

        Args:
            input_dict (dict): Input dictionary describing a turbine model.

        Returns:
            dict: A validated dictionary.
        """
        return self._validate_dict(input_dict, self._turbine_properties)

    def validate_wake(self, input_dict):
        """
        Checks for the required values and types of input in the given input
        dictionary as required by the :py:obj:`~.wake.Wake` object.

        Args:
            input_dict (dict): Input dictionary describing a wake model.

        Returns:
            dict: A validated dictionary.
        """
        return self._validate_dict(input_dict, self._wake_properties)

    def validate_farm(self, input_dict):
        """
        Checks for the required values and types of input in the given input
        dictionary as required by the :py:obj:`~.farm.Farm` object.

        Args:
            input_dict (dict): Input dictionary describing a farm model.

        Returns:
            dict: A validated dictionary.
        """
        return self._validate_dict(input_dict, self._farm_properties)

    def read(self, input_file=None, input_dict=None):
        """
        Parses a given input file or input dictionary and validates the
        contents.

        Args:
            input_file (str, optional): A string path to the JSON input file.
                Defaults to None.
            input_dict (dict, optional): A Python dictionary of inputs.
                Defaults to None.

        Raises:
            ValueError: Input file or dictionary must be provided.

        Returns:
            dict, dict, dict, dict:

                - Meta data from the input dictionary.
                - Validated "turbine" section from the input dictionary.
                - Validated "wake" section from the input dictionary.
                - Validated "farm" section from the input dictionary.
        """
        if input_file is not None:
            input_dict = self._parseJSON(input_file)
        elif input_dict is not None:
            input_dict = input_dict.copy()
        else:
            raise ValueError("Input file or dictionary must be provided")

        turbine_dict = input_dict.pop("turbine")
        wake_dict = input_dict.pop("wake")
        farm_dict = input_dict.pop("farm")
        meta_dict = input_dict
        return meta_dict, turbine_dict, wake_dict, farm_dict
