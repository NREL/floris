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

import json
from pathlib import Path
import yaml

import floris.logging_manager as logging_manager
from floris.type_dec import FromDictMixin
from floris.simulation import (
    Farm,
    WakeModelManager,
    FlowField,
    TurbineGrid,
    FlowFieldGrid,
    sequential_solver,
    full_flow_sequential_solver
)
from attrs import define, field


@define
class Floris(logging_manager.LoggerBase, FromDictMixin):
    """
    Top-level class that describes a Floris model and initializes the
    simulation. Use the :py:class:`~.simulation.farm.Farm` attribute to
    access other objects within the model.
    """

    logging: dict = field(converter=dict)
    solver: dict = field(converter=dict)
    flow_field: FlowField = field(converter=FlowField.from_dict)
    wake: WakeModelManager = field(converter=WakeModelManager.from_dict)
    farm: Farm = field(converter=Farm.from_dict)

    def __attrs_post_init__(self) -> None:
        if self.solver["type"] == "turbine_grid":
            self.grid = TurbineGrid(
                turbine_coordinates=self.farm.coordinates,
                reference_turbine_diameter=self.farm.rotor_diameter,
                wind_directions=self.flow_field.wind_directions,
                wind_speeds=self.flow_field.wind_speeds,
                grid_resolution=self.solver["turbine_grid_points"],
            )
        elif self.solver["type"] == "flow_field_grid":
            self.grid = FlowFieldGrid(
                turbine_coordinates=self.farm.coordinates,
                reference_turbine_diameter=self.farm.rotor_diameter,
                wind_directions=self.flow_field.wind_directions,
                wind_speeds=self.flow_field.wind_speeds,
                grid_resolution=self.solver["flow_field_grid_points"],
            )
        else:
            raise ValueError(
                f"Supported solver types are [turbine_grid, flow_field_grid], but type given was {self.solver['type']}"
            )

        # Initialize farm quanitities
        self.farm.set_yaw_angles(self.flow_field.n_wind_directions, self.flow_field.n_wind_speeds)

        # Configure logging
        logging_manager.configure_console_log(
            self.logging["console"]["enable"],
            self.logging["console"]["level"],
        )
        logging_manager.configure_file_log(
            self.logging["file"]["enable"],
            self.logging["file"]["level"],
        )
    

    # @profile
    def steady_state_atmospheric_condition(self):

        # Initialize field quanitities; doing this immediately prior to doing
        # the calculation step allows for manipulating inputs in a script
        # without changing the data structures
        self.flow_field.initialize_velocity_field(self.grid)

        # <<interface>>
        # start = time.time()
        elapsed_time = sequential_solver(self.farm, self.flow_field, self.grid, self.wake)
        # end = time.time()
        # elapsed_time = end - start

        self.grid.finalize()
        self.flow_field.finalize(self.grid.unsorted_indices)
        return elapsed_time

    def solve_for_viz(self):
        # Do the calculation with the TurbineGrid for a single wind speed
        # and wind direction and 1 point on the grid. Then, use the result
        # to construct the full flow field grid.
        # This function call should be for a single wind direction and wind speed
        # since the memory consumption is very large.

        # self.steady_state_atmospheric_condition()

        # turbine_based_floris = copy.deepcopy(self)
        turbine_grid = TurbineGrid(
            turbine_coordinates=self.farm.coordinates,
            reference_turbine_diameter=self.farm.rotor_diameter,
            wind_directions=self.flow_field.wind_directions,
            wind_speeds=self.flow_field.wind_speeds,
            grid_resolution=5,
        )

        self.flow_field.initialize_velocity_field(self.grid)

        full_flow_sequential_solver(self.farm, self.flow_field, self.grid, turbine_grid, self.wake)


    ## I/O

    @classmethod
    def from_json(cls, input_file_path: str | Path) -> Floris:
        """Creates a `Floris` instance from a JSON file.

        Args:
            input_file_path (str): The relative or absolute file path and name to the
                JSON input file.

        Returns:
            Floris: The class object instance.
        """
        input_file_path = Path(input_file_path).resolve()
        with open(input_file_path) as json_file:
            input_dict = json.load(json_file)
        return Floris.from_dict(input_dict)

    @classmethod
    def from_yaml(cls, input_file_path: str | Path) -> Floris:
        """Creates a `Floris` instance from a YAML file.

        Args:
            input_file_path (str): The relative or absolute file path and name to the
                YAML input file.

        Returns:
            Floris: The class object instance
        """
        input_file_path = Path(input_file_path).resolve()
        input_dict = yaml.load(open(input_file_path, "r"), Loader=yaml.SafeLoader)
        return Floris.from_dict(input_dict)

    def to_json(self, output_file_path: str) -> None:
        """Converts the `Floris` object to an input-ready JSON file at `output_file_path`.

        Args:
            output_file_path (str): The full path and filename for where to save the JSON file.
        """
        output_dict = self._prepare_for_save()
        with open(output_file_path, "w+") as f:
            json.dump(output_dict, f, indent=2, sort_keys=False)

    def to_yaml(self, output_file_path: str) -> None:
        """Converts the `Floris` object to an input-ready YAML file at `output_file_path`.

        Args:
            output_file_path (str): The full path and filename for where to save the YAML file.
        """
        output_dict = self._prepare_for_save()
        with open(output_file_path, "w+") as f:
            yaml.dump(output_dict, f, default_flow_style=False)