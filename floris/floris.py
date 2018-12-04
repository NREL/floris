# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from .input_reader import InputReader

class Floris():
    """
    Floris is the highest level class of the Floris package. Import this class
    and instantiate it with a path to an input file to begin running Floris. Use
    the ``farm`` attribute to access other objects within the model.

    inputs:
        input_file: str - path to the json input file
        input_dict: dict - dictionary of appropriate inputs

    outputs:
        self: Floris - an instantiated Floris object
    """

    def __init__(self, input_file=None, input_dict=None):
        self.input_reader = InputReader()
        self.input_file = input_file
        self.input_dict = input_dict
        self._farm = []
        self.add_farm(
                input_file=self.input_file,
                input_dict=self.input_dict
        )

    @property
    def farm(self):
        if len(self._farm) == 1:
            return self._farm[0]
        else:
            return self._farm

    @farm.setter
    def farm(self, value):
        if not hasattr(self._farm):
            self._farm = value

    def add_farm(self, input_file=None, input_dict=None):
        """
        Adds a farm with user-defined input file to the FLORIS object
        """
        self._farm.append(self.input_reader.read(input_file=input_file,
                                                input_dict=input_dict))

    def list_farms(self):
        """
        Lists the farms and relevant farm details stored in FLORIS object
        """
        for num, farm in enumerate(self._farm):
            print('Farm', num)
            print('\tDescription:', farm.description)
            print('\tWake Model:', farm.flow_field.wake.velocity_model.type_string)                                      
            print('\tDeflection Model:', farm.flow_field.wake.deflection_model.type_string)
