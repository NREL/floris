# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import pickle
from .input_reader import InputReader


class Floris():
    """
    Top-level class that contains a Floris model.

    Floris is the highest level class of the Floris package. Import 
    this class and instantiate it with a path to an input file to 
    create a Floris model. Use the ``farm`` attribute to access other 
    objects within the model.

    Args:
        input_file: A string that is the path to the json input file.
        input_dict: A dictionary as generated from the input_reader.

    Returns:
        Floris: An instantiated Floris object.
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
        """
        Property of the Floris object that returns the farm(s) 
        contained within the object.

        Returns:
            Farm: A Farm object, or if multiple farms, a list of Farm 
            objects.

        Examples:
            To get the Farm object(s) stored in a Floris object:

            >>> farms = floris.farm()
        """
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
        A method that adds a farm with user-defined input file to the 
        Floris object.

        Returns:
            *None* -- The :py:class:`floris.simulation.floris` object 
            is updated directly.

        Examples:
            To add a farm to a Floris object using a specific input 
            file:

            >>> floris.add_farm(input_file='input.json')

            To add a farm to a Floris object using the stored input 
            file:

            >>> floris.add_farm()
        """
        self._farm.append(self.input_reader.read(input_file=input_file,
                                                 input_dict=input_dict))

    def list_farms(self):
        """
        A method that lists the farms and relevant farm details stored 
        in Floris object.

        Returns:
            *None* -- The farm infomation is printed to the console.

        Examples:
            To list the current farms in Floris object:

            >>> floris.list_farms()
        """
        for num, farm in enumerate(self._farm):
            print('Farm {}'.format(num))
            print(farm)

    def export_pickle(self, pickle_file):
        """
        A method that exports a farm to a pickle file.

        Returns:
            *None* -- Creates a pickle file.

        Examples:
            To export a farm to a pickle file:

            >>> floris.export_pickle('saved_farm.p')
        """
        pickle.dump(self.farm, open(pickle_file, "wb"))

    def import_pickle(self, pickle_file):
        """
        A method that imports a farm from a pickle file.

        Returns:
            *None* - Loads the farm into the 
            :py:class:`floris.simulation.floris.farm` object.

        Examples:
            To load a pickled farm:

            >>> floris.import_pickle('saved_farm.p')
        """
        self.farm = pickle.load(open(pickle_file, "rb"))
