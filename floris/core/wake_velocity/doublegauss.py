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

import numexpr as ne
import numpy as np
from scipy import special as sp
from attrs import define, field

from floris.core import (
    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)
from floris.utilities import (
    cosd,
    sind,
    tand,
)


@define
class DoublegaussVelocityDeficit(BaseModel):

    # These two parameters should be inputs
    # A: float = field(default=0.04)
    # sigma_max_rel: float = field(default=4.0)
        
    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
            "u_initial": flow_field.u_initial_sorted,
            "wind_veer": flow_field.wind_veer
        }
        return kwargs

    # @profile
    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        axial_induction_i: np.ndarray,
        deflection_field_i: np.ndarray,
        yaw_angle_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        hub_height_i: float,
        rotor_diameter_i: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u_initial: np.ndarray,
        wind_veer: float,
    ) -> None:

        # Initialize the velocity deficit array
        velocity_deficit = np.zeros_like(u_initial)

        n = 1/3

        ks = 0.2966*ct_i**0.7345
        epsilon = 4.6e-2*ct_i**-1.06
        c_ = 0.44*ct_i**0.22

        X_D = (x - x_i)/rotor_diameter_i
        X_D[X_D<0] = 0
        R_D = np.hypot(y-y_i,z-z_i)/rotor_diameter_i

        sigma_D = (ks*X_D**n+epsilon)
        sigma_D[X_D<=0] = 1e-16

        r0_D = 3/8

        sigma_r0 = sigma_D/r0_D + 1e-16

        M = sigma_r0*(2*sigma_r0*np.exp(-1/2./sigma_r0**2) + np.sqrt(2*np.pi)*(sp.erfc(1./(np.sqrt(2)*sigma_r0))-1))
        N = sigma_r0*(  sigma_r0*np.exp(-sigma_r0**-2)     + np.sqrt(np.pi)/2*(sp.erfc(1./sigma_r0)-1))

        C_ = ( M-np.sqrt(M**2-1/2*N*ct_i/r0_D**2) ) / (2*N)
        C_[X_D<=0]=0

        Dp = -1/2*( (R_D + r0_D)/sigma_D )**2
        Dm = -1/2*( (R_D - r0_D)/sigma_D )**2

        F = (np.exp(Dp)+np.exp(Dm))/2

        Udef_Uinf = c_*C_*F
        Udef_Uinf[X_D<0] = 0

        velocity_deficit = Udef_Uinf

        return velocity_deficit




def mask_upstream_wake(mesh_y_rotated, x_coord_rotated, y_coord_rotated, turbine_yaw):
    yR = mesh_y_rotated - y_coord_rotated
    xR = yR * tand(turbine_yaw) + x_coord_rotated
    return xR, yR
