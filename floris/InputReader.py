"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import json
from Turbine import Turbine
from Wake import Wake
from Farm import Farm


class InputReader():
    """
        Describe InputReader here
    """

    def __init__(self):

        self._validObjects = ["turbine", "wake", "farm"]

        self._turbineProperties = {
            "rotorDiameter": float,
            "hubHeight": float,
            "numBlades": int,
            "pP": float,
            "pT": float,
            "generatorEfficiency": float,
            "eta": float,
            "power_thrust_table": dict,
            "bladePitch": float,
            "yawAngle": float,
            "tilt": float,
            "TSR": float
        }

        # self._wakeProperties = {
        #     "": type
        # }

        # self._farmProperties = {
        #     "": type
        # }

    def _parseJSON(self, filename):
        """
        Opens the input json file and parses the contents into a python dict
        inputs:
            filename: str - path to the json input file
        outputs:
            data: dict - contents of the json input file

        """
        with open(filename) as jsonfile:
            data = json.load(jsonfile)
        return data

    def _validateJSON(self, jsonDict, typeMap):
        """
        Verifies that the expected fields exist in the json input file and
        validates the type of the input data by casting the fields to
        appropriate values based on the predefined type maps in
        _turbineProperties
        _wakeProperties
        _farmProperties

        inputs:
            jsonDict: dict - Input dictionary with all elements of type str
            typeMap: dict - Predefined type map for type checking inputs
                             structured as {"property": type}
        outputs:
            validated: dict - Validated and correctly typed input property
                              dictionary
        """

        validated = {}

        # validate the object type
        if "type" not in jsonDict:
            raise KeyError("'type' key is required")

        if jsonDict["type"] not in self._validObjects:
            raise ValueError("'type' must be one of {}".format(", ".join(self._validObjects)))

        validated["type"] = jsonDict["type"]

        # validate the description
        if "description" not in jsonDict:
            raise KeyError("'description' key is required")

        validated["description"] = jsonDict["description"]

        # validate the properties dictionary
        if "properties" not in jsonDict:
            raise KeyError("'properties' key is required")
        # check every attribute in the predefined type dictionary for existence
        # and proper type in the given inputs
        propDict = {}
        properties = jsonDict["properties"]
        for element in typeMap:
            if element not in properties:
                raise KeyError("'{}' is required for object type '{}'".format(element, validated["type"]))

            value,error = self._cast_to_type(typeMap[element], properties[element])
            if error is not None:
                raise error("'{}' must be of type '{}'".format(element, typeMap[element]))

            propDict[element] = value

        validated["properties"] = propDict

        return validated

    def _cast_to_type(self, typecast, value):
        """
        Casts the string input to the type in typecast
        inputs:
            typcast: type - the type class to use on value
            value: str - the input string to cast to 'typecast'
        outputs:
            position 0: type or None - the casted value
            position 1: None or Error - the caught error
        """
        try:
            return typecast(value), None
        except ValueError:
            return None, ValueError

    # Public methods

    def build_turbine(self, jsonDict):
        """
        Creates a turbine object from a given input file
        inputs:
            inputFile: str - path to the json input file
        outputs:
            turbine: Turbine - instantiated Turbine object
        """
        # jsonDict = self._parseJSON(inputFile)
        propertyDict = self._validateJSON(jsonDict, self._turbineProperties)
        turbine = Turbine()
        turbine.init_with_dict(propertyDict)
        return turbine

    def input_reader(self, inputFile):
        """
        Parses main input file
        inputs:
            inputFile: str - path to the json input file
        outputs:
            farm: instantiated FLORIS model of wind farm
        """
        jsonDict = self._parseJSON(inputFile)

        # Build the turbine model
        return self.build_turbine(jsonDict['turbine'])


    # def buildWake(self, inputFile):

    # def buildFarm(self, inputFile):
