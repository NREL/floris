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
        self.reference_turbine_diameter = input_dictionary["reference_turbine_diameter"]
        self.air_density = input_dictionary["air_density"]


    def initialize_velocity_field(self, grid: Grid) -> None:

        # Create an initial wind profile as a function of height. The values here will
        # be multiplied with the wind speeds to give the initial wind field.
        # Since we use grid.z, this is a vertical plane for each turbine
        # Here, the profile is of shape (# turbines, N grid points, M grid points)
        # This velocity profile is 1.0 at the reference wind height and then follows wind shear as an exponent.
        wind_profile_plane = ( grid.z / self.reference_wind_height) ** self.wind_shear

        # Add a dimension for each wind speed by broadcasting
        # Here, the profile is of shape (# wind speeds, # turbines, N grid points, M grid points)
        wind_profile_plane = np.array(self.n_wind_speeds * [wind_profile_plane]) # broadcast

        # Create the array containing the initial uniform wind profile
        # This is also of shape (# wind speeds, # turbines, N grid points, M grid points)
        n_elements = np.prod([d for d in np.shape(grid.z)])                     # find the total number of elements in lower dimensions for each wind speed
        _wind_speeds = np.array( n_elements * [self.wind_speeds]).T             # broadcast the input wind speeds to an array of this size
        _wind_speeds = np.reshape(_wind_speeds, np.shape(wind_profile_plane))   # reshape based on the wind profile array

        # Create the sheer-law wind profile
        # This array is of shape (# wind speeds, # turbines, N grid points, M grid points)
        self.u_initial = _wind_speeds * wind_profile_plane

        # The broadcast and matrix multiplication above is equivalent to this:
        # self.u_initial = np.zeros((grid.n_turbines, self.n_wind_speeds, 5, 5))
        # for i in range(self.n_wind_speeds):
        #     for j in range(grid.n_turbines):
        #         self.u_initial[j, i, :, :] = self.wind_speeds[i] * wind_profile_plane[i]

        self.v_initial = np.zeros(np.shape(self.u_initial))
        self.w_initial = np.zeros(np.shape(self.u_initial))

        self.u = self.u_initial.copy()
        self.v = self.v_initial.copy()
        self.w = self.w_initial.copy()


    @property
    def n_wind_speeds(self) -> int:
        return len(self.wind_speeds)
