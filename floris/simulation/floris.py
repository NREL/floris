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

import numpy as np
import pandas as pd
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
from floris.type_dec import NDArrayFloat
from floris.utilities import (
    load_yaml,
    reverse_rotate_coordinates_rel_west,
)


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

        # Configure logging
        logging_manager.configure_console_log(
            self.logging["console"]["enable"],
            self.logging["console"]["level"],
        )
        logging_manager.configure_file_log(
            self.logging["file"]["enable"],
            self.logging["file"]["level"],
        )

        # Initialize farm quantities that depend on other objects
        self.farm.construct_turbine_map()
        self.farm.construct_turbine_thrust_coefficient_functions()
        self.farm.construct_turbine_axial_induction_functions()
        self.farm.construct_turbine_power_functions()
        self.farm.construct_turbine_power_thrust_tables()
        self.farm.construct_hub_heights()
        self.farm.construct_rotor_diameters()
        self.farm.construct_turbine_TSRs()
        self.farm.construct_turbine_ref_tilts()
        self.farm.construct_turbine_tilt_interps()
        self.farm.construct_turbine_correct_cp_ct_for_tilt()
        self.farm.set_yaw_angles(self.flow_field.n_findex)
        self.farm.set_tilt_to_ref_tilt(self.flow_field.n_findex)

        if self.solver["type"] == "turbine_grid":
            self.grid = TurbineGrid(
                turbine_coordinates=self.farm.coordinates,
                turbine_diameters=self.farm.rotor_diameters,
                wind_directions=self.flow_field.wind_directions,
                grid_resolution=self.solver["turbine_grid_points"],
                time_series=self.flow_field.time_series,
            )
        elif self.solver["type"] == "turbine_cubature_grid":
            self.grid = TurbineCubatureGrid(
                turbine_coordinates=self.farm.coordinates,
                turbine_diameters=self.farm.rotor_diameters,
                wind_directions=self.flow_field.wind_directions,
                time_series=self.flow_field.time_series,
                grid_resolution=self.solver["turbine_grid_points"],
            )
        elif self.solver["type"] == "flow_field_grid":
            self.grid = FlowFieldGrid(
                turbine_coordinates=self.farm.coordinates,
                turbine_diameters=self.farm.rotor_diameters,
                wind_directions=self.flow_field.wind_directions,
                grid_resolution=self.solver["flow_field_grid_points"],
                time_series=self.flow_field.time_series,
            )
        elif self.solver["type"] == "flow_field_planar_grid":
            self.grid = FlowFieldPlanarGrid(
                turbine_coordinates=self.farm.coordinates,
                turbine_diameters=self.farm.rotor_diameters,
                wind_directions=self.flow_field.wind_directions,
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
                self.flow_field.n_findex,
                self.grid.sorted_coord_indices
            )

    def initialize_domain(self):
        """Initialize solution space prior to wake calculations"""

        # Initialize field quantities; doing this immediately prior to doing
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

        if vel_model in ["gauss", "cc", "turbopark", "jensen"] and \
            self.farm.correct_cp_ct_for_tilt.any():
            self.logger.warning(
                "The current model does not account for vertical wake deflection due to " +
                "tilt. Corrections to power and thrust coefficient can be included, but no " +
                "vertical wake deflection will occur."
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

        self.finalize()

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
            turbine_diameters=self.farm.rotor_diameters,
            wind_directions=self.flow_field.wind_directions,
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

        return self.flow_field.u_sorted[:,:,0,0] # Remove turbine grid dimensions

    def solve_for_velocity_deficit_profiles(
        self,
        direction: str,
        downstream_dists: NDArrayFloat | list,
        profile_range: NDArrayFloat | list,
        resolution: int,
        homogeneous_wind_speed: float,
        ref_rotor_diameter: float,
        x_start: float,
        y_start: float,
        reference_height: float,
    ) -> list[pd.DataFrame]:
        """
        Extract velocity deficit profiles. See
        :py:meth:`~floris.tools.floris_interface.FlorisInterface.sample_velocity_deficit_profiles`
        for more details.
        """

        # Create a grid that contains coordinates for all the sample points in all profiles.
        # Effectively, this is a grid of parallel lines.
        n_lines = len(downstream_dists)

        # Coordinate system (x1, x2, x3) is used to define the sample points. The origin is at
        # (x_start, y_start, reference_height) and x1 is in the streamwise direction.
        # The x1-coordinate is fixed for every line (every row in  `x1`).
        x1 = np.atleast_2d(downstream_dists).T * np.ones((n_lines, resolution))

        if resolution == 1:
            single_line = [0.0]
        else:
            single_line = np.linspace(profile_range[0], profile_range[1], resolution)

        if direction == 'cross-stream':
            x2 = single_line * np.ones((n_lines, resolution))
            x3 = np.zeros((n_lines, resolution))
        elif direction == 'vertical':
            x3 = single_line * np.ones((n_lines, resolution))
            x2 = np.zeros((n_lines, resolution))

        # Find the coordinates of the sample points in the inertial frame (x, y, z). This is done
        # through one rotation and one translation.
        x, y, z = reverse_rotate_coordinates_rel_west(
            self.flow_field.wind_directions,
            x1[None, :, :],
            x2[None, :, :],
            x3[None, :, :],
            x_center_of_rotation=0.0,
            y_center_of_rotation=0.0,
        )
        x = np.squeeze(x, axis=0) + x_start
        y = np.squeeze(y, axis=0) + y_start
        z = np.squeeze(z, axis=0) + reference_height

        u = self.solve_for_points(x.flatten(), y.flatten(), z.flatten())
        u = np.reshape(u[0, :], (n_lines, resolution))
        velocity_deficit = (homogeneous_wind_speed - u) / homogeneous_wind_speed

        velocity_deficit_profiles = []

        for i in range(n_lines):
            df = pd.DataFrame(
                {
                    'x': x[i],
                    'y': y[i],
                    'z': z[i],
                    'x1/D': x1[i]/ref_rotor_diameter,
                    'x2/D': x2[i]/ref_rotor_diameter,
                    'x3/D': x3[i]/ref_rotor_diameter,
                    'velocity_deficit': velocity_deficit[i],
                }
            )
            velocity_deficit_profiles.append(df)

        return velocity_deficit_profiles

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
