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


from typing import List

import numpy as np
from numpy import newaxis

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

    # Public methods
    def initialize_velocity_field(self, grid: Grid) -> List[float]:
        """
        calculate initial values at these points.

        2) Initializing a gridded curl model (using a call to _discretize_gridded_domain)
        3) Appending points to a non-curl model, this could either be for adding additional points to calculate
            for use in visualization, or else to enable calculation of additional points.  Note this assumes
            the flow has previously been discritized in a prior call to _compute_initialized_domain /
            _discretize_turbine_domain

        Args:
            points: An array that contains the x, y, and z coordinates of
                user-specified points, at which the flow field velocity
                is recorded.
            with_resolution: Vec3
        """
        # Create an initial wind profile as a function of height
        # Since we use grid.z, this is a plane for each turbine
        wind_profile_plane = (grid.z / self.reference_wind_height) ** self.wind_shear
        # Add a dimension for each wind speed
        wind_profile_plane = np.reshape(
            np.array([wind_profile_plane] * self.n_wind_speeds),  # broadcast
            (
                grid.n_turbines,
                len(self.wind_speeds),
                grid.grid_resolution,
                grid.grid_resolution,
            ),  # reshape
        )

        # Initialize the grid with the wind profile
        _wind_speeds = np.reshape(
            np.array([self.wind_speeds] * 25).T,  # broadcast
            (
                len(self.wind_speeds),
                grid.grid_resolution,
                grid.grid_resolution,
            ),  # reshape
        )
        self.u_initial = _wind_speeds * wind_profile_plane

        # The broadcast and matrix multiplication above is equivalent to this:
        # self.u_initial = np.zeros((grid.n_turbines, self.n_wind_speeds, 5, 5))
        # for i in range(self.n_wind_speeds):
        #     for j in range(grid.n_turbines):
        #         self.u_initial[j, i, :, :] = self.wind_speeds[i] * wind_profile_plane[i]

        # u = [
        #     n turbines,
        #     n wind directions,
        #     n wind speeds,
        #     y,
        #     z
        # ]
        #     x, is this dimension needed? it might be implicit since the turbines are always oriented in x

        self.v_initial = np.zeros(np.shape(self.u_initial))
        self.w_initial = np.zeros(np.shape(self.u_initial))

        self.u = self.u_initial.copy()
        self.v = self.v_initial.copy()
        self.w = self.w_initial.copy()

    @property
    def n_wind_speeds(self) -> int:
        return len(self.wind_speeds)
