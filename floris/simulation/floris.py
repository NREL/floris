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
from floris.utilities import load_yaml

import floris.logging_manager as logging_manager
from floris.type_dec import FromDictMixin
from floris.simulation import (
    Farm,
    WakeModelManager,
    FlowField,
    Turbine,
    Grid,
    TurbineGrid,
    FlowFieldGrid,
    FlowFieldPlanarGrid,
    sequential_solver,
    cc_solver,
    full_flow_sequential_solver,
    full_flow_cc_solver
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
    wake: WakeModelManager = field(converter=WakeModelManager.from_dict)
    farm: Farm = field(converter=Farm.from_dict)
    flow_field: FlowField = field(converter=FlowField.from_dict)

    # These fields are included to appease the requirement that all inputs must
    # be mapped to a field in the class. They are not used in FLORIS.
    name: str  = field(converter=str)
    description: str = field(converter=str)
    floris_version: str = field(converter=str)

    grid: Grid = field(init=False)

    def __attrs_post_init__(self) -> None:

        # Initialize farm quanitities that depend on other objects
        self.farm.construct_turbine_map()
        self.farm.construct_turbine_fCts()
        self.farm.construct_turbine_fCps()
        self.farm.construct_turbine_power_interps()
        self.farm.construct_hub_heights()
        self.farm.construct_rotor_diameters()
        self.farm.construct_turbine_TSRs()
        self.farm.construc_turbine_pPs()
        self.farm.construct_coordinates()
        self.farm.set_yaw_angles(self.flow_field.n_wind_directions, self.flow_field.n_wind_speeds)

        if self.solver["type"] == "turbine_grid":
            self.grid = TurbineGrid(
                turbine_coordinates=self.farm.coordinates,
                reference_turbine_diameter=self.farm.rotor_diameters,
                wind_directions=self.flow_field.wind_directions,
                wind_speeds=self.flow_field.wind_speeds,
                grid_resolution=self.solver["turbine_grid_points"],
            )
        elif self.solver["type"] == "flow_field_grid":
            self.grid = FlowFieldGrid(
                turbine_coordinates=self.farm.coordinates,
                reference_turbine_diameter=self.farm.rotor_diameters,
                wind_directions=self.flow_field.wind_directions,
                wind_speeds=self.flow_field.wind_speeds,
                grid_resolution=self.solver["flow_field_grid_points"],
            )
        elif self.solver["type"] == "flow_field_planar_grid":
            self.grid = FlowFieldPlanarGrid(
                turbine_coordinates=self.farm.coordinates,
                reference_turbine_diameter=self.farm.rotor_diameters,
                wind_directions=self.flow_field.wind_directions,
                wind_speeds=self.flow_field.wind_speeds,
                normal_vector=self.solver["normal_vector"],
                planar_coordinate=self.solver["planar_coordinate"],
                grid_resolution=self.solver["flow_field_grid_points"],
                x1_bounds=self.solver["flow_field_bounds"][0],
                x2_bounds=self.solver["flow_field_bounds"][1],
            )
        else:
            raise ValueError(
                f"Supported solver types are [turbine_grid, flow_field_grid], but type given was {self.solver['type']}"
            )

        if type(self.grid) == TurbineGrid:
            self.farm.expand_farm_properties(
                self.flow_field.n_wind_directions, self.flow_field.n_wind_speeds, self.grid.sorted_coord_indices
            )

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

        # Initialize farm quantities
        self.farm.initialize(self.grid.sorted_indices)

        vel_model = self.wake.model_strings["velocity_model"]

        # <<interface>>
        # start = time.time()

        if vel_model=="cc":
            elapsed_time = cc_solver(
                self.farm,
                self.flow_field,
                self.grid,
                self.wake
            )
        else:
            elapsed_time = sequential_solver(
                self.farm,
                self.flow_field,
                self.grid,
                self.wake
            )
        # end = time.time()
        # elapsed_time = end - start

        self.grid.finalize()
        self.flow_field.finalize(self.grid.unsorted_indices)
        self.farm.finalize(self.grid.unsorted_indices)
        return elapsed_time

    def solve_for_viz(self):
        # Do the calculation with the TurbineGrid for a single wind speed
        # and wind direction and 1 point on the grid. Then, use the result
        # to construct the full flow field grid.
        # This function call should be for a single wind direction and wind speed
        # since the memory consumption is very large.

        self.flow_field.initialize_velocity_field(self.grid)

        vel_model = self.wake.model_strings["velocity_model"]

        if vel_model=="cc":
            full_flow_cc_solver(self.farm, self.flow_field, self.grid, self.wake)
        else:
            full_flow_sequential_solver(self.farm, self.flow_field, self.grid, self.wake)


    ## I/O

    @classmethod
    def from_file(cls, input_file_path: str | Path, filetype: str = None) -> Floris:
        """Creates a `Floris` instance from an input file. Must be filetype
        JSON or YAML.

        Args:
            input_file_path (str): The relative or absolute file path and name to the
                input file.
            filetype (str): The type to export: [YAML | JSON]

        Returns:
            Floris: The class object instance.
        """
        input_file_path = Path(input_file_path).resolve()
        if filetype is None:
            filetype = input_file_path.suffix.strip(".")

        with open(input_file_path) as input_file:
            if filetype.lower() in ("yml", "yaml"):
                input_dict = load_yaml(input_file_path)
            elif filetype.lower() == "json":
                input_dict = json.load(input_file)

                # TODO: This is a temporary hack to put the turbine definition into the farm.
                # Long term, we need a strategy for handling this. The YAML file format supports
                # pointers to other data, for example.
                # input_dict["farm"]["turbine"] = input_dict["turbine"]
                # input_dict.pop("turbine")
            else:
                raise ValueError("Supported import filetypes are JSON and YAML")
        return Floris.from_dict(input_dict)

    def to_file(self, output_file_path: str, filetype: str="YAML") -> None:
        """Converts the `Floris` object to an input-ready JSON or YAML file at `output_file_path`.

        Args:
            output_file_path (str): The full path and filename for where to save the file.
            filetype (str): The type to export: [YAML | JSON]
        """
        with open(output_file_path, "w+") as f:
            if filetype.lower() == "yaml":
                yaml.dump(self.as_dict(), f, default_flow_style=False)
            elif filetype.lower() == "json":
                json.dump(self.as_dict(), f, indent=2, sort_keys=False)
            else:
                raise ValueError("Supported export filetypes are JSON and YAML")
