# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See read the https://floris.readthedocs.io for documentation

from .turbine import Turbine
from .wake import Wake
from .farm import Farm
import pickle
from .input_reader import InputReader
from ..utilities import setup_logger, LogClass


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
        # Parse the input into dictionaries
        input_reader = InputReader()
        self.meta_dict, turbine_dict, wake_dict, farm_dict \
            = input_reader.read(input_file, input_dict)

        # Configure logging
        logging_dict = self.meta_dict["logging"]
        self.logger = setup_logger(name=__name__, logging_dict=logging_dict)

        # Initialize the simulation objects
        turbine = Turbine(turbine_dict)
        wake = Wake(wake_dict)
        self.farm = Farm(farm_dict, turbine, wake)

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
