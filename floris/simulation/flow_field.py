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
import numpy as np
from attrs import define, field

from floris.simulation import Grid
from floris.type_dec import (
    floris_array_converter,
    FromDictMixin,
    NDArrayFloat,
)


@define
class FlowField(FromDictMixin):
    wind_speeds: NDArrayFloat = field(converter=floris_array_converter)
    wind_directions: NDArrayFloat = field(converter=floris_array_converter)
    wind_veer: float = field(converter=float)
    wind_shear: float = field(converter=float)
    air_density: float = field(converter=float)
    turbulence_intensity: float = field(converter=float)
    reference_wind_height: float = field(converter=float)
    time_series : bool = field(default=False)

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
            if len(self.het_map[0][0].points[0]) == 2:
                speed_ups = self.calculate_speed_ups(
                    self.het_map,
                    grid.x_sorted,
                    grid.y_sorted
                )
            elif len(self.het_map[0][0].points[0]) == 3:
                speed_ups = self.calculate_speed_ups(
                    self.het_map,
                    grid.x_sorted,
                    grid.y_sorted,
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

        # Check that the het maps wd dimension matches
        if self.n_wind_directions!= np.array(het_map).shape[1]:
            raise ValueError(
                "het_map's wind direction dimension not equal to number of wind directions"
            )

        if z is not None:
            # Calculate the 3-dimensional speed ups; reshape is needed as the generator
            # adds an extra dimension
            speed_ups = np.reshape(
                [
                    het_map[0][i](x[i:i+1,:,:,:,:], y[i:i+1,:,:,:,:], z[i:i+1,:,:,:,:])
                    for i in range(len(het_map[0]))
                ],
                np.shape(x)
            )

            # If there are any points requested outside the user-defined area, use the
            # nearest-neighbor interplonat to determine those speed up values
            if np.isnan(speed_ups).any():
                idx_nan = np.where(np.isnan(speed_ups))
                speed_ups_out_of_region = np.reshape(
                    [
                        het_map[1][i](x[i:i+1,:,:,:,:], y[i:i+1,:,:,:,:], z[i:i+1,:,:,:,:])
                        for i in range(len(het_map[1]))
                    ],
                    np.shape(x)
                )

                speed_ups[idx_nan] = speed_ups_out_of_region[idx_nan]

        else:
            # Calculate the 2-dimensional speed ups; reshape is needed as the generator
            # adds an extra dimension
            speed_ups = np.reshape(
                [
                    het_map[0][i](x[i:i+1,:,:,:,:], y[i:i+1,:,:,:,:])
                    for i in range(len(het_map[0]))
                ],
                np.shape(x)
            )

            # If there are any points requested outside the user-defined area, use the
            # nearest-neighbor interplonat to determine those speed up values
            if np.isnan(speed_ups).any():
                idx_nan = np.where(np.isnan(speed_ups))
                speed_ups_out_of_region = np.reshape(
                    [
                        het_map[1][i](x[i:i+1,:,:,:,:], y[i:i+1,:,:,:,:])
                        for i in range(len(het_map[1]))
                    ],
                    np.shape(x)
                )

                speed_ups[idx_nan] = speed_ups_out_of_region[idx_nan]

        return speed_ups
