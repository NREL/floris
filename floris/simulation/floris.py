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

from pathlib import Path

import yaml
from attrs import define, field

from floris import logging_manager
from floris.simulation import (
    BaseClass,
    cc_solver,
    empirical_gauss_solver,
    Farm,
    FlowField,
    FlowFieldGrid,
    FlowFieldPlanarGrid,
    full_flow_cc_solver,
    full_flow_empirical_gauss_solver,
    full_flow_sequential_solver,
    full_flow_turbopark_solver,
    Grid,
    PointsGrid,
    sequential_solver,
    State,
    TurbineCubatureGrid,
    TurbineGrid,
    turbopark_solver,
    WakeModelManager,
)
from floris.utilities import load_yaml


@define
class Floris(BaseClass):
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

        self.check_deprecated_inputs()

        # Initialize farm quanitities that depend on other objects
        self.farm.construct_turbine_map()
        self.farm.construct_turbine_fCts()
        self.farm.construct_turbine_power_interps()
        self.farm.construct_hub_heights()
        self.farm.construct_rotor_diameters()
        self.farm.construct_turbine_TSRs()
        self.farm.construct_turbine_pPs()
        self.farm.construct_turbine_pTs()
        self.farm.construct_turbine_ref_density_cp_cts()
        self.farm.construct_turbine_ref_tilt_cp_cts()
        self.farm.construct_turbine_fTilts()
        self.farm.construct_turbine_correct_cp_ct_for_tilt()
        self.farm.construct_coordinates()
        self.farm.set_yaw_angles(self.flow_field.n_wind_directions, self.flow_field.n_wind_speeds)
        self.farm.set_tilt_to_ref_tilt(
            self.flow_field.n_wind_directions,
            self.flow_field.n_wind_speeds,
        )

        if self.solver["type"] == "turbine_grid":
            self.grid = TurbineGrid(
                turbine_coordinates=self.farm.coordinates,
                reference_turbine_diameter=self.farm.rotor_diameters,
                wind_directions=self.flow_field.wind_directions,
                wind_speeds=self.flow_field.wind_speeds,
                grid_resolution=self.solver["turbine_grid_points"],
                time_series=self.flow_field.time_series,
            )
        elif self.solver["type"] == "turbine_cubature_grid":
            self.grid = TurbineCubatureGrid(
                turbine_coordinates=self.farm.coordinates,
                reference_turbine_diameter=self.farm.rotor_diameters,
                wind_directions=self.flow_field.wind_directions,
                wind_speeds=self.flow_field.wind_speeds,
                time_series=self.flow_field.time_series,
                grid_resolution=self.solver["turbine_grid_points"],
            )
        elif self.solver["type"] == "flow_field_grid":
            self.grid = FlowFieldGrid(
                turbine_coordinates=self.farm.coordinates,
                reference_turbine_diameter=self.farm.rotor_diameters,
                wind_directions=self.flow_field.wind_directions,
                wind_speeds=self.flow_field.wind_speeds,
                grid_resolution=self.solver["flow_field_grid_points"],
                time_series=self.flow_field.time_series,
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
                time_series=self.flow_field.time_series,
                x1_bounds=self.solver["flow_field_bounds"][0],
                x2_bounds=self.solver["flow_field_bounds"][1],
            )
        else:
            raise ValueError(
                "Supported solver types are "
                "[turbine_grid, turbine_cubature_grid, flow_field_grid, flow_field_planar_grid], "
                f"but type given was {self.solver['type']}"
            )

        if isinstance(self.grid, (TurbineGrid, TurbineCubatureGrid)):
            self.farm.expand_farm_properties(
                self.flow_field.n_wind_directions,
                self.flow_field.n_wind_speeds,
                self.grid.sorted_coord_indices
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

    def check_deprecated_inputs(self):
        """
        This function should used when the FLORIS input file changes in order to provide
        an informative error and suggest a fix.
        """

        error_messages = []
        # Check for missing values add in version 3.2 and 3.4
        for turbine in self.farm.turbine_definitions:

            if "ref_density_cp_ct" not in turbine.keys():
                error_messages.append(
                    "From FLORIS v3.2, the turbine definition must include 'ref_density_cp_ct'. "
                    "This value represents the air density at which the provided Cp and Ct "
                    "curves are defined. Previously, this was assumed to be 1.225 kg/m^3, "
                    "and other air density values applied were assumed to be a deviation "
                    "from the defined level. FLORIS now requires the user to explicitly "
                    "define the reference density. Add 'ref_density_cp_ct' to your "
                    "turbine definition and try again. For a description of the turbine inputs, "
                    "see https://nrel.github.io/floris/input_reference_turbine.html."
                )

            if "ref_tilt_cp_ct" not in turbine.keys():
                error_messages.append(
                    "From FLORIS v3.4, the turbine definition must include 'ref_tilt_cp_ct'. "
                    "This value represents the tilt angle at which the provided Cp and Ct "
                    "curves are defined. Add 'ref_tilt_cp_ct' to your turbine definition and "
                    "try again. For a description of the turbine inputs, "
                    "see https://nrel.github.io/floris/input_reference_turbine.html."
                )

            if len(error_messages) > 0:
                raise ValueError(
                    f"{turbine['turbine_type']} turbine model\n" +
                    "\n\n".join(error_messages)
                )

    # @profile
    def initialize_domain(self):
        """Initialize solution space prior to wake calculations"""

        # Initialize field quanitities; doing this immediately prior to doing
        # the calculation step allows for manipulating inputs in a script
        # without changing the data structures
        self.flow_field.initialize_velocity_field(self.grid)

        # Initialize farm quantities
        self.farm.initialize(self.grid.sorted_indices)

        self.state.INITIALIZED

    def steady_state_atmospheric_condition(self):
        """Perform the steady-state wind farm wake calculations. Note that
        initialize_domain() is required to be called before this function."""

        vel_model = self.wake.model_strings["velocity_model"]

        # <<interface>>
        # start = time.time()

        if vel_model in ["gauss", "cc", "turbopark", "jensen"] and \
            self.farm.correct_cp_ct_for_tilt.any():
            self.logger.warn(
                "The current model does not account for vertical wake deflection due to " +
                "tilt. Corrections to Cp and Ct can be included, but no vertical wake " +
                "deflection will occur."
            )

        if vel_model=="cc":
            cc_solver(
                self.farm,
                self.flow_field,
                self.grid,
                self.wake
            )
        elif vel_model=="turbopark":
            turbopark_solver(
                self.farm,
                self.flow_field,
                self.grid,
                self.wake
            )
        elif vel_model=="empirical_gauss":
            empirical_gauss_solver(
                self.farm,
                self.flow_field,
                self.grid,
                self.wake
            )
        else:
            sequential_solver(
                self.farm,
                self.flow_field,
                self.grid,
                self.wake
            )
        # end = time.time()
        # elapsed_time = end - start

        self.finalize()
        # return elapsed_time

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
        elif vel_model=="turbopark":
            full_flow_turbopark_solver(self.farm, self.flow_field, self.grid, self.wake)
        elif vel_model=="empirical_gauss":
            full_flow_empirical_gauss_solver(self.farm, self.flow_field, self.grid, self.wake)
        else:
            full_flow_sequential_solver(self.farm, self.flow_field, self.grid, self.wake)

    def solve_for_points(self, x, y, z):
        # Do the calculation with the TurbineGrid for a single wind speed
        # and wind direction and a 3x3 rotor grid. Then, use the result
        # to construct the full flow field grid.
        # This function call should be for a single wind direction and wind speed
        # since the memory consumption is very large.

        # Instantiate the flow_grid
        field_grid = PointsGrid(
            points_x=x,
            points_y=y,
            points_z=z,
            turbine_coordinates=self.farm.coordinates,
            reference_turbine_diameter=self.farm.rotor_diameters,
            wind_directions=self.flow_field.wind_directions,
            wind_speeds=self.flow_field.wind_speeds,
            grid_resolution=1,
            time_series=self.flow_field.time_series,
            x_center_of_rotation=self.grid.x_center_of_rotation,
            y_center_of_rotation=self.grid.y_center_of_rotation
        )

        self.flow_field.initialize_velocity_field(field_grid)

        vel_model = self.wake.model_strings["velocity_model"]

        if vel_model == "cc" or vel_model == "turbopark":
            raise NotImplementedError(
                "solve_for_points is currently only available with the "+\
                "gauss, jensen, and empirical_guass models."
            )
        elif vel_model == "empirical_gauss":
            full_flow_empirical_gauss_solver(self.farm, self.flow_field, field_grid, self.wake)
        else:
            full_flow_sequential_solver(self.farm, self.flow_field, field_grid, self.wake)

        return self.flow_field.u_sorted[:,:,:,0,0] # Remove turbine grid dimensions

    def finalize(self):
        # Once the wake calculation is finished, unsort the values to match
        # the user-supplied order of things.
        self.flow_field.finalize(self.grid.unsorted_indices)
        self.farm.finalize(self.grid.unsorted_indices)
        self.state = State.USED

    ## I/O

    @classmethod
    def from_file(cls, input_file_path: str | Path) -> Floris:
        """Creates a `Floris` instance from an input file. Must be filetype YAML.

        Args:
            input_file_path (str): The relative or absolute file path and name to the
                input file.

        Returns:
            Floris: The class object instance.
        """
        input_dict = load_yaml(Path(input_file_path).resolve())
        return Floris.from_dict(input_dict)

    def to_file(self, output_file_path: str) -> None:
        """Converts the `Floris` object to an input-ready YAML file at `output_file_path`.

        Args:
            output_file_path (str): The full path and filename for where to save the file.
        """
        with open(output_file_path, "w+") as f:
            yaml.dump(
                self.as_dict(),
                f,
                sort_keys=False,
                default_flow_style=False
            )
