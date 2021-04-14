# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np

from ...utilities import cosd, sind
from .base_wake_solver import WakeSolver


class Cumulative(WakeSolver):
    """
    The cumulative solver does some things.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: jvm-
    """

    def __init__(self):
        super().__init__()
        self.solver_string = "cumulative"

    def function(
        self,
        flow_field,
        sorted_map,
        rx,
        ry,
        initial_rotated_x,
        initial_rotated_y,
        rotated_z,
        u_wake,
        center_of_rotation,
        no_wake,
        track_n_upstream_wakes,
    ):
        Ctmp = []
        for n, (coord, turbine) in enumerate(sorted_map):
            xloc, yloc = np.array(rx == coord.x1), np.array(ry == coord.x2)
            idx = int(np.where(np.logical_and(yloc, xloc))[0])

            if np.unique(flow_field.wind_map.grid_wind_direction).size == 1:
                # only rotate grid once for homogeneous wind direction
                rotated_x, rotated_y = initial_rotated_x, initial_rotated_y

            else:
                # adjust grid rotation with respect to current turbine for
                # heterogeneous wind direction
                wd = (
                    flow_field.wind_map.turbine_wind_direction[idx]
                    - flow_field.wind_map.grid_wind_direction
                )

                # for straight wakes, change rx[idx] to initial_rotated_x
                xoffset = center_of_rotation.x1 - rx[idx]
                # for straight wakes, change ry[idx] to initial_rotated_y
                yoffset = center_of_rotation.x2 - ry[idx]
                y_grid_offset = xoffset * sind(wd) + yoffset * cosd(wd) - yoffset
                rotated_y = initial_rotated_y - y_grid_offset

                xoffset = center_of_rotation.x1 - initial_rotated_x
                yoffset = center_of_rotation.x2 - initial_rotated_y
                x_grid_offset = xoffset * cosd(wd) - yoffset * sind(wd) - xoffset
                rotated_x = initial_rotated_x - x_grid_offset

            # update the turbine based on the velocity at its hub
            turbine.update_velocities(
                u_wake, coord, flow_field, rotated_x, rotated_y, rotated_z
            )

            # get the wake deflection field
            deflection = flow_field._compute_turbine_wake_deflection(
                rotated_x, rotated_y, rotated_z, turbine, coord, flow_field
            )

            # get the velocity deficit accounting for the deflection
            (
                u_wake,
                v_wake,
                w_wake,
                Ctmp,
            ) = flow_field._compute_turbine_velocity_deficit(
                rotated_x,
                rotated_y,
                rotated_z,
                turbine,
                coord,
                deflection,
                flow_field,
                n=n,
                sorted_map=sorted_map,
                u_wake=u_wake,
                Ctmp=Ctmp,
            )

            flow_field.WAT_downstream(
                sorted_map,
                rx,
                ry,
                turbine,
                coord,
                rotated_x,
                rotated_y,
                rotated_z,
                u_wake,
                track_n_upstream_wakes,
            )

        # combine this turbine's wake into the full wake field
        if not no_wake:
            # u_wake = self.wake.combination_function(u_wake, turb_u_wake)
            flow_field.u = flow_field.u_initial - u_wake
            flow_field.v = flow_field.v + v_wake
            flow_field.w = flow_field.w + w_wake

        return u_wake
