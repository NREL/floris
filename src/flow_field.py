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


import numpy as np

from .grid import Grid


class FlowField:
    def __init__(self, input_dictionary):
        self.wind_shear = input_dictionary["wind_shear"]
        self.wind_veer = input_dictionary["wind_veer"]
        self.wind_speeds = np.array(input_dictionary["wind_speeds"])
        self.wind_directions = np.array(input_dictionary["wind_directions"])
        self.reference_wind_height = input_dictionary["reference_wind_height"]
        self.air_density = input_dictionary["air_density"]

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

    def finalize(self, unsorted_indeces):
        n_wind_directions = np.shape(self.u)[0]
        n_turbines = np.shape(self.u)[2]

        _u = np.zeros_like(self.u)
        _v = np.zeros_like(self.v)
        _w = np.zeros_like(self.w)
        for i in range(n_wind_directions):
            for j in range(n_turbines):
                _u[i, :, j] = self.u[i, :, unsorted_indeces[i, j]]
                _v[i, :, j] = self.v[i, :, unsorted_indeces[i, j]]
                _w[i, :, j] = self.w[i, :, unsorted_indeces[i, j]]
        self.u = _u
        self.v = _v
        self.w = _w

    @property
    def n_wind_speeds(self) -> int:
        return len(self.wind_speeds)

    @property
    def n_wind_directions(self) -> int:
        return len(self.wind_directions)
