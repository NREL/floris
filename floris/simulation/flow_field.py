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

import attrs
import matplotlib.path as mpltPath
import numpy as np
from attrs import define, field
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

from floris.simulation import (
    BaseClass,
    Grid,
)
from floris.type_dec import (
    floris_array_converter,
    NDArrayFloat,
)


@define
class FlowField(BaseClass):
    wind_speeds: NDArrayFloat = field(converter=floris_array_converter)
    wind_directions: NDArrayFloat = field(converter=floris_array_converter)
    wind_veer: float = field(converter=float)
    wind_shear: float = field(converter=float)
    air_density: float = field(converter=float)
    turbulence_intensity: float = field(converter=float)
    reference_wind_height: float = field(converter=float)
    time_series : bool = field(default=False)
    heterogenous_inflow_config: dict = field(default=None)

    n_wind_speeds: int = field(init=False)
    n_wind_directions: int = field(init=False)

    u_initial_sorted: NDArrayFloat = field(init=False, default=np.array([]))
    v_initial_sorted: NDArrayFloat = field(init=False, default=np.array([]))
    w_initial_sorted: NDArrayFloat = field(init=False, default=np.array([]))
    u_sorted: NDArrayFloat = field(init=False, default=np.array([]))
    v_sorted: NDArrayFloat = field(init=False, default=np.array([]))
    w_sorted: NDArrayFloat = field(init=False, default=np.array([]))
    u: NDArrayFloat = field(init=False, default=np.array([]))
    v: NDArrayFloat = field(init=False, default=np.array([]))
    w: NDArrayFloat = field(init=False, default=np.array([]))
    het_map: list = field(init=False, default=None)
    dudz_initial_sorted: NDArrayFloat = field(init=False, default=np.array([]))

    turbulence_intensity_field: NDArrayFloat = field(init=False, default=np.array([]))
    turbulence_intensity_field_sorted: NDArrayFloat = field(init=False, default=np.array([]))
    turbulence_intensity_field_sorted_avg: NDArrayFloat = field(init=False, default=np.array([]))

    @wind_speeds.validator
    def wind_speeds_validator(self, instance: attrs.Attribute, value: NDArrayFloat) -> None:
        """Using the validator method to keep the `n_wind_speeds` attribute up to date."""
        if self.time_series:
            self.n_wind_speeds = 1
        else:
            self.n_wind_speeds = value.size

    @wind_directions.validator
    def wind_directions_validator(self, instance: attrs.Attribute, value: NDArrayFloat) -> None:
        """Using the validator method to keep the `n_wind_directions` attribute up to date."""
        self.n_wind_directions = value.size

    @heterogenous_inflow_config.validator
    def heterogenous_config_validator(self, instance: attrs.Attribute, value: dict | None) -> None:
        """Using the validator method to check that the heterogenous_inflow_config dictionary has
        the correct key-value pairs.
        """
        if value is None:
            return

        # Check that the correct keys are supplied for the heterogenous_inflow_config dict
        for k in ["speed_multipliers", "x", "y"]:
            if k not in value.keys():
                raise ValueError(
                    "heterogenous_inflow_config must contain entries for 'speed_multipliers',"
                    f"'x', and 'y', with 'z' optional. Missing '{k}'."
                )
        if "z" not in value:
            # If only a 2D case, add "None" for the z locations
            value["z"] = None

    @het_map.validator
    def het_map_validator(self, instance: attrs.Attribute, value: list | None) -> None:
        """Using this validator to make sure that the het_map has an interpolant defined for
        each wind direction.
        """
        if value is None:
            return

        if self.n_wind_directions!= np.array(value).shape[0]:
            raise ValueError(
                "The het_map's wind direction dimension not equal to number of wind directions."
            )


    def __attrs_post_init__(self) -> None:
        if self.heterogenous_inflow_config is not None:
            self.generate_heterogeneous_wind_map()


    def initialize_velocity_field(self, grid: Grid) -> None:

        # Create an initial wind profile as a function of height. The values here will
        # be multiplied with the wind speeds to give the initial wind field.
        # Since we use grid.z, this is a vertical plane for each turbine
        # Here, the profile is of shape (# turbines, N grid points, M grid points)
        # This velocity profile is 1.0 at the reference wind height and then follows wind
        # shear as an exponent.
        # NOTE: the convention of which dimension on the TurbineGrid is vertical and horizontal is
        # determined by this line. Since the right-most dimension on grid.z is storing the values
        # for height, using it here to apply the shear law makes that dimension store the vertical
        # wind profile.
        wind_profile_plane = (grid.z_sorted / self.reference_wind_height) ** self.wind_shear
        dwind_profile_plane = (
            self.wind_shear
            * (1 / self.reference_wind_height) ** self.wind_shear
            * (grid.z_sorted) ** (self.wind_shear - 1)
        )

        # If no hetergeneous inflow defined, then set all speeds ups to 1.0
        if self.het_map is None:
            speed_ups = 1.0

        # If heterogeneous flow data is given, the speed ups at the defined
        # grid locations are determined in either 2 or 3 dimensions.
        else:
            bounds = np.array(list(zip(
                self.heterogenous_inflow_config['x'],
                self.heterogenous_inflow_config['y']
            )))
            hull = ConvexHull(bounds)
            polygon = Polygon(bounds[hull.vertices])
            path = mpltPath.Path(polygon.boundary.coords)
            points = np.column_stack(
                (
                    grid.x_sorted_inertial_frame.flatten(),
                    grid.y_sorted_inertial_frame.flatten(),
                )
            )
            inside = path.contains_points(points)
            if not np.all(inside):
                self.logger.warning(
                    "The calculated flow field contains points outside of the the user-defined "
                    "heterogeneous inflow bounds. For these points, the interpolated value has "
                    "been filled with the freestream wind speed. If this is not the desired "
                    "behavior, the user will need to expand the heterogeneous inflow bounds to "
                    "fully cover the calculated flow field area."
                )

            if len(self.het_map[0].points[0]) == 2:
                speed_ups = self.calculate_speed_ups(
                    self.het_map,
                    grid.x_sorted_inertial_frame,
                    grid.y_sorted_inertial_frame
                )
            elif len(self.het_map[0].points[0]) == 3:
                speed_ups = self.calculate_speed_ups(
                    self.het_map,
                    grid.x_sorted_inertial_frame,
                    grid.y_sorted_inertial_frame,
                    grid.z_sorted
                )

        # Create the sheer-law wind profile
        # This array is of shape (# wind directions, # wind speeds, grid.template_array)
        # Since generally grid.template_array may be many different shapes, we use transposes
        # here to do broadcasting from left to right (transposed), and then transpose back.
        # The result is an array the wind speed and wind direction dimensions on the left side
        # of the shape and the grid.template array on the right
        if self.time_series:
            self.u_initial_sorted = (
                (self.wind_speeds[:].T * wind_profile_plane.T).T
                * speed_ups
            )
            self.dudz_initial_sorted = (
                (self.wind_speeds[:].T * dwind_profile_plane.T).T
                * speed_ups
            )
        else:
            self.u_initial_sorted = (
                (self.wind_speeds[None, :].T * wind_profile_plane.T).T
                * speed_ups
            )
            self.dudz_initial_sorted = (
                (self.wind_speeds[None, :].T * dwind_profile_plane.T).T
                * speed_ups
            )
        self.v_initial_sorted = np.zeros(
            np.shape(self.u_initial_sorted),
            dtype=self.u_initial_sorted.dtype
        )
        self.w_initial_sorted = np.zeros(
            np.shape(self.u_initial_sorted),
            dtype=self.u_initial_sorted.dtype
        )

        self.u_sorted = self.u_initial_sorted.copy()
        self.v_sorted = self.v_initial_sorted.copy()
        self.w_sorted = self.w_initial_sorted.copy()

        self.turbulence_intensity_field = self.turbulence_intensity * np.ones(
            (
                self.n_wind_directions,
                self.n_wind_speeds,
                grid.n_turbines,
                1,
                1,
            )
        )
        self.turbulence_intensity_field_sorted = self.turbulence_intensity_field.copy()

    def finalize(self, unsorted_indices):
        self.u = np.take_along_axis(self.u_sorted, unsorted_indices, axis=2)
        self.v = np.take_along_axis(self.v_sorted, unsorted_indices, axis=2)
        self.w = np.take_along_axis(self.w_sorted, unsorted_indices, axis=2)

        self.turbulence_intensity_field = np.mean(
            np.take_along_axis(
                self.turbulence_intensity_field_sorted,
                unsorted_indices,
                axis=2
            ),
            axis=(3,4)
        )

    def calculate_speed_ups(self, het_map, x, y, z=None):
        if z is not None:
            # Calculate the 3-dimensional speed ups; squeeze is needed as the generator
            # adds an extra dimension
            speed_ups = np.squeeze(
                [het_map[i](x[i:i+1], y[i:i+1], z[i:i+1]) for i in range( len(het_map))],
                axis=1,
            )

        else:
            # Calculate the 2-dimensional speed ups; squeeze is needed as the generator
            # adds an extra dimension
            speed_ups = np.squeeze(
                [het_map[i](x[i:i+1], y[i:i+1]) for i in range(len(het_map))],
                axis=1,
            )

        return speed_ups

    def generate_heterogeneous_wind_map(self):
        """This function creates the heterogeneous interpolant used to calculate heterogeneous
        inflows. The interpolant is for computing wind speed based on an x and y location in the
        flow field. This is computed using SciPy's LinearNDInterpolator and uses a fill value
        equal to the freestream for interpolated values outside of the user-defined heterogeneous
        map bounds.

        Args:
            heterogenous_inflow_config (dict): The heterogeneous inflow configuration dictionary.
            The configuration should have the following inputs specified.
                - **speed_multipliers** (list): A list of speed up factors that will multiply
                    the specified freestream wind speed. This 2-dimensional array should have an
                    array of multiplicative factors defined for each wind direction.
                - **x** (list): A list of x locations at which the speed up factors are defined.
                - **y**: A list of y locations at which the speed up factors are defined.
                - **z** (optional): A list of z locations at which the speed up factors are defined.
        """
        speed_multipliers = self.heterogenous_inflow_config['speed_multipliers']
        x = self.heterogenous_inflow_config['x']
        y = self.heterogenous_inflow_config['y']
        z = self.heterogenous_inflow_config['z']

        if z is not None:
            # Compute the 3-dimensional interpolants for each wind direction
            # Linear interpolation is used for points within the user-defined area of values,
            # while the freestream wind speed is used for points outside that region
            in_region = [
                LinearNDInterpolator(list(zip(x, y, z)), multiplier, fill_value=1.0)
                for multiplier in speed_multipliers
            ]
        else:
            # Compute the 2-dimensional interpolants for each wind direction
            # Linear interpolation is used for points within the user-defined area of values,
            # while the freestream wind speed is used for points outside that region
            in_region = [
                LinearNDInterpolator(list(zip(x, y)), multiplier, fill_value=1.0)
                for multiplier in speed_multipliers
            ]

        self.het_map = in_region
