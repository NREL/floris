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


import pickle

import floris.logging_manager as logging_manager

from .farm import Farm
from .wake import Wake
from .turbine import Turbine
from .input_reader import InputReader


class Floris(logging_manager.LoggerBase):
    """
    Top-level class that describes a Floris model and initializes the
    simulation. Use the :py:class:`~.simulation.farm.Farm` attribute to
    access other objects within the model.
    """

    def __init__(self, input_file=None, input_dict=None):
        """
        Import this class with one of the two optional input arguments
        to create a Floris model. The `input_dict` and `input_file`
        should both conform to the same data format.

        Args:
            input_file (str, optional): Path to the input file which will
                be parsed and converted to a Python dict.
            input_dict (dict, optional): Python dict given directly.
        """
        # Parse the input into dictionaries
        input_reader = InputReader()
        self.meta_dict, turbine_dict, wake_dict, farm_dict = input_reader.read(
            input_file, input_dict
        )

        # Configure logging
        self.meta_dict["logging"]
        logging_manager.configure_console_log(
            self.meta_dict["logging"]["console"]["enable"],
            self.meta_dict["logging"]["console"]["level"],
        )
        logging_manager.configure_file_log(
            self.meta_dict["logging"]["file"]["enable"],
            self.meta_dict["logging"]["file"]["level"],
        )

        # Initialize the simulation objects
        turbine = Turbine(turbine_dict)
        wake = Wake(wake_dict)
        self.farm = Farm(farm_dict, turbine, wake)

    def export_pickle(self, pickle_file):
        """
        Exports the :py:class:`~.farm.Farm` object to a
        Pickle binary file.

        Args:
            pickle_file (str): Name of the resulting output file.
        """
        pickle.dump(self.farm, open(pickle_file, "wb"))

    def import_pickle(self, pickle_file):
        """
        Imports the :py:class:`~.farm.Farm` object from a
        Pickle binary file.

        Args:
            pickle_file (str): Name of the Pickle file to load.
        """
        self.farm = pickle.load(open(pickle_file, "rb"))
