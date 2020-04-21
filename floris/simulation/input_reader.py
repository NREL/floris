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
from ..utilities import setup_logger


class InputReader():
    """
    InputReader parses json input files into inputs for FLORIS objects.

    InputReader is a helper class which parses json input files and
    provides an interface to instantiate model objects in FLORIS. This
    class handles input validation regarding input type, but does not
    enforce value checking. It is designed to function as a singleton
    object, but that is not enforced or required.

    Returns:
        InputReader:  An instantiated InputReader object
    """

    def __init__(self):

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
            "TSR": float
        }

        self._wake_properties = {
            "velocity_model": str,
            "turbulence_model": str,
            "deflection_model": str,
            "combination_model": str,
            "parameters": dict
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
            "wind_y": list
        }

    def _parseJSON(self, filename):
        """
        Opens the input json file and parses the contents into a python
        dict.

        Args:
            filename:  A string that is the path to the json input file.

        Returns:
            dict:  A dictionary *data* that contains the json input
            file.
        """
        with open(filename) as jsonfile:
            data = json.load(jsonfile)
        return data

    def _validate_dict(self, input_dict, type_map):
        """
        Verifies that the expected fields exist in the json input file
        and validates the type of the input data by casting the fields
        to appropriate values based on the predefined type maps in.

        Args:
            input_dict: Input dictionary with all elements of type str.
            type_map: Predefined type map dictionary for type checking 
                inputs structured as {"property": type}.

        Returns:
            dict: Validated and correctly typed input property
            dictionary.
        """

        validated = {}

        # validate the object type
        if "type" not in input_dict:
            raise KeyError("'type' key is required in input")
        if input_dict["type"] not in self._valid_objects:
            raise ValueError("'type' must be one of {}".format(", ".join(self._valid_objects)))
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
                raise KeyError("'{}' is required for object type '{}'".format(element, validated["type"]))
            prop_dict[element] = self._cast(type_map[element], properties[element])
        validated["properties"] = prop_dict

        for element in properties:
            if element not in type_map:
                raise KeyError("'{}' is given but not required for object type '{}'".format(element, validated["type"]))
        
        return validated

    def _cast(self, typecast, value):
        """
        Casts the string input to the type in typecast.

        Args:
            typcast: type - the type class to use on value.
            value: str - the input string to cast to 'typecast'.

        Returns:
            type or None: position 0 - the casted value.
            None or Error: position 1 - the caught error.
        """
        return typecast(value)

    def validate_turbine(self, input_dict):
        """
        Checks for the required values and types of input in the
        given input dictionary as required by the
        :py:obj:`floris.simulation.turbine` object.

        Args:
            input_dict: Input dictionary describing a turbine model.

        Returns:
            dict: A validated dictionary
        """
        return self._validate_dict(input_dict, self._turbine_properties)

    def validate_wake(self, input_dict):
        """
        Checks for the required values and types of input in the
        given input dictionary as required by the
        :py:obj:`floris.simulation.wake` object.

        Args:
            input_dict: dict - Input dictionary describing a wake model.

        Returns:
            dict: A validated dictionary
        """
        return self._validate_dict(input_dict, self._wake_properties)

    def validate_farm(self, input_dict):
        """
        Checks for the required values and types of input in the
        given input dictionary as required by the
        :py:obj:`floris.simulation.farm` object.

        Args:
            input_dict: Input dictionary describing a farm model.

        Returns:
            dict: A validated dictionary
        """
        return self._validate_dict(input_dict, self._farm_properties)

    def read(self, input_file=None, input_dict=None):
        """
        Parses a given input file or input dictionary and validated the
        contents.

        Args:
            input_file: A string path to the json input file.
            input_dict: A Python dictionary of inputs

        Returns:
            input_dict: A validated dictionary
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
