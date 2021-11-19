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

import attr
import numpy as np
import numpy.typing as npt

from floris.utilities import FromDictMixin, int_attrib, float_attrib, attrs_array_converter

from floris.simulation import Grid


NDArrayFloat = npt.NDArray[np.float64]


@attr.s(auto_attribs=True)
class FlowField(FromDictMixin):
    wind_shear: float = float_attrib()
    wind_veer: float = float_attrib()
    wind_speeds: NDArrayFloat = attr.ib(converter=attrs_array_converter)
    wind_directions: NDArrayFloat = attr.ib(converter=attrs_array_converter)
    reference_wind_height: int = int_attrib()
    air_density: float = float_attrib()

    n_wind_speeds: int = attr.ib(init=False)
    n_wind_directions: int = attr.ib(init=False)

    u_initial: NDArrayFloat = attr.ib(init=False)
    v_initial: NDArrayFloat = attr.ib(init=False)
    w_initial: NDArrayFloat = attr.ib(init=False)

    u: NDArrayFloat = attr.ib(init=False)
    v: NDArrayFloat = attr.ib(init=False)
    w: NDArrayFloat = attr.ib(init=False)

    @wind_speeds.validator
    def wind_speeds_validator(self, instance: attr.Attribute, value: NDArrayFloat) -> None:
        """Using the validator method to keep the `n_wind_speeds` attribute up to date."""
        self.n_wind_speeds = value.size

    @wind_directions.validator
    def wind_directionss_validator(self, instance: attr.Attribute, value: NDArrayFloat) -> None:
        """Using the validator method to keep the `n_wind_directions` attribute up to date."""
        self.n_wind_directions = value.size

    def initialize_velocity_field(self, grid: Grid) -> None:

        # Create an initial wind profile as a function of height. The values here will
        # be multiplied with the wind speeds to give the initial wind field.
        # Since we use grid.z, this is a vertical plane for each turbine
        # Here, the profile is of shape (# turbines, N grid points, M grid points)
        # This velocity profile is 1.0 at the reference wind height and then follows wind shear as an exponent.
        wind_profile_plane = (grid.z / self.reference_wind_height) ** self.wind_shear

        # Create the sheer-law wind profile
        # This array is of shape (# wind directions, # wind speeds, # turbines, N grid points, M grid points)
        self.u_initial = self.wind_speeds[None, :, None, None, None] * wind_profile_plane
        self.v_initial = np.zeros(np.shape(self.u_initial))
        self.w_initial = np.zeros(np.shape(self.u_initial))

        self.u = self.u_initial.copy()
        self.v = self.v_initial.copy()
        self.w = self.w_initial.copy()

    def finalize(self, unsorted_indices):
        self.u = np.take_along_axis(self.u, unsorted_indices, axis=2)
        self.v = np.take_along_axis(self.v, unsorted_indices, axis=2)
        self.w = np.take_along_axis(self.w, unsorted_indices, axis=2)
