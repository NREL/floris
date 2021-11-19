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


import json

import floris.logging_manager as logging_manager

from floris.simulation import Farm
from floris.simulation import TurbineGrid
from floris.simulation import sequential_solver

# from .wake import Wake
from .turbine import Turbine
from .flow_field import FlowField
from .wake_velocity.jensen import JensenVelocityDeficit


class Floris(logging_manager.LoggerBase):
    """
    Top-level class that describes a Floris model and initializes the
    simulation. Use the :py:class:`~.simulation.farm.Farm` attribute to
    access other objects within the model.
    """

    def __init__(self, input_file_path=None, input_dict=None):
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
        if input_file_path is not None:
            input_file = open(input_file_path)
            input_dict = json.load(input_file)
        elif input_dict is not None:
            input_dict = input_dict.copy()
        else:
            raise ValueError("Floris: No input file or dictionary given.")

        turbine_dict = input_dict.pop("turbine")
        # wake_dict = input_dict.pop("wake")
        farm_dict = input_dict.pop("farm")
        meta_dict = input_dict

        # Configure logging
        logging_manager.configure_console_log(
            meta_dict["logging"]["console"]["enable"],
            meta_dict["logging"]["console"]["level"],
        )
        logging_manager.configure_file_log(
            meta_dict["logging"]["file"]["enable"],
            meta_dict["logging"]["file"]["level"],
        )

        # Initialize the simulation objects
        self.turbine = Turbine(**turbine_dict)
        # self.wake = Wake(wake_dict)

        wind_directions = farm_dict["wind_directions"]
        wind_speeds = farm_dict["wind_speeds"]
        layout_x = farm_dict["layout_x"]
        layout_y = farm_dict["layout_y"]
        wtg_id = [f"WTG_{str(i).zfill(3)}" for i in range(len(layout_x))]
        turbine_id = ["t1"] * len(layout_x)
        turbine_map = dict(t1=turbine_dict)
        self.farm = Farm(
            turbine_id=turbine_id,
            turbine_map=turbine_map,
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            layout_x=layout_x,
            layout_y=layout_y,
            wtg_id=wtg_id,
        )
        self.flow_field = FlowField.from_dict(farm_dict)

    def annual_energy_production(self, wind_rose):
        # self.steady_state_atmospheric_condition()
        pass

    def steady_state_atmospheric_condition(self):

        # <<interface>>
        # Initialize grid and field quanitities
        grid = TurbineGrid(
            turbine_coordinates=self.farm.coordinates,
            reference_turbine_diameter=self.farm.reference_turbine_diameter,
            wind_directions=self.flow_field.wind_directions,
            wind_speeds=self.flow_field.wind_speeds,
            grid_resolution=5,
        )
        # TODO: where do we pass in grid_resolution? Hardcoded to 5 above.

        self.flow_field.initialize_velocity_field(grid)

        # <<interface>>
        # JensenVelocityDeficit.solver(self.farm, self.flow_field)
        sequential_solver(self.farm, self.flow_field, grid)

        grid.finalize()
        self.flow_field.finalize(grid.unsorted_indices)

    # Utility functions

    def set_wake_model(self, wake_model):
        """
        Sets the velocity deficit model to use as given, and determines the
        wake deflection model based on the selected velocity deficit model.

        Args:
            wake_model (str): The desired wake model.

        Raises:
            Exception: Invalid wake model.
        """
        valid_wake_models = [
            "jensen",
            "turbopark",
            "multizone",
            "gauss",
            "gauss_legacy",
            "blondel",
            "ishihara_qian",
            "curl",
        ]
        if wake_model not in valid_wake_models:
            # TODO: logging
            raise Exception("Invalid wake model. Valid options include: {}.".format(", ".join(valid_wake_models)))

        self.flow_field.wake.velocity_model = wake_model
        if wake_model == "jensen" or wake_model == "multizone" or wake_model == "turbopark":
            self.flow_field.wake.deflection_model = "jimenez"
        elif wake_model == "blondel" or wake_model == "ishihara_qian" or "gauss" in wake_model:
            self.flow_field.wake.deflection_model = "gauss"
        else:
            self.flow_field.wake.deflection_model = wake_model

        self.flow_field.reinitialize_flow_field(
            with_resolution=self.flow_field.wake.velocity_model.model_grid_resolution
        )

        self.reinitialize_turbines()

    def update_hub_heights(self):
        """
        Triggers a rebuild of the internal Python dictionary. This may be
        used to update the z-component of the turbine coordinates if
        the hub height has changed.
        """
        self.turbine_map_dict = self._build_internal_dict(self.coords, self.turbines)
