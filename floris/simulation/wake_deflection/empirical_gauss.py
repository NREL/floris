# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from typing import Any, Dict

import numpy as np
from attrs import define, field

from floris.simulation import (
    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)
from floris.utilities import cosd, sind

@define
class EmpiricalGaussVelocityDeflection(BaseModel):

    horizontal_deflection_gain_D: float = field(default=3.0)
    vertical_deflection_gain_D: float = field(default=-1)
    deflection_rate: float = field(default=15)
    mixing_gain_deflection: float = field(default=0.0)
    yaw_added_mixing_gain: float = field(default=0.0)

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = dict(
            x=grid.x_sorted,
            y=grid.y_sorted,
            z=grid.z_sorted,
            freestream_velocity=flow_field.u_initial_sorted,
            wind_veer=flow_field.wind_veer,
        )
        return kwargs

    # @profile
    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        yaw_i: np.ndarray,
        tilt_i: np.ndarray,
        mixing_i: np.ndarray,
        ct_i: np.ndarray,
        rotor_diameter_i: float,
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        freestream_velocity: np.ndarray,
        wind_veer: float,
    ):
        """
        Calculates the deflection field of the wake. See
        :cite:`gdm-bastankhah2016experimental` and :cite:`gdm-King2019Controls`
        for details on the methods used.

        Args:
            x_locations (np.array): An array of floats that contains the
                streamwise direction grid coordinates of the flow field
                domain (m).
            y_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the direction normal to
                x and parallel to the ground (m).
            z_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the vertical
                direction (m).
            turbine (:py:obj:`floris.simulation.turbine`): Object that
                represents the turbine creating the wake.
            coord (:py:obj:`floris.utilities.Vec3`): Object containing
                the coordinate of the turbine creating the wake (m).
            flow_field (:py:class:`floris.simulation.flow_field`): Object
                containing the flow field information for the wind farm.

        Returns:
            np.array: Deflection field for the wake.
        """
        # ==============================================================

        deflection_gain_y = self.horizontal_deflection_gain_D*rotor_diameter_i
        if self.vertical_deflection_gain_D == -1:
            deflection_gain_z = deflection_gain_y
        else:
            deflection_gain_z = self.vertical_deflection_gain_D * \
                rotor_diameter_i
        
        # Convert to radians, CW yaw for consistency with other models
        yaw_r = np.pi/180 * -yaw_i
        tilt_r = np.pi/180 * tilt_i

        A_y = (deflection_gain_y*ct_i*yaw_r)/\
              (1+self.mixing_gain_deflection*mixing_i)
        
        A_z = (deflection_gain_z*ct_i*tilt_r)/\
              (1+self.mixing_gain_deflection*mixing_i)
            
        # Apply downstream mask in the process
        x_normalized = ((x - x_i)*np.array(x > x_i + 0.1))/rotor_diameter_i
        
        log_term = np.log((x_normalized - self.deflection_rate) \
                          /(x_normalized + self.deflection_rate) + 2)

        deflection_y = A_y * log_term
        deflection_z = A_z * log_term

        return deflection_y, deflection_z

def yaw_added_wake_mixing(
    axial_induction_i,
    yaw_angle_i,
    downstream_distance_D_i,
    yaw_added_mixing_gain
):
    return axial_induction_i[:,:,:,0,0] * yaw_added_mixing_gain * \
        (1 - cosd(yaw_angle_i[:,:,:,0,0]))\
        / downstream_distance_D_i**2
