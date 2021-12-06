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

import attr
import numpy as np

from floris.simulation import TurbineGrid
from floris.utilities import float_attrib, model_attrib, cosd, sind, tand
from floris.simulation import BaseModel
from floris.simulation import Farm
from floris.simulation import FlowField


@attr.s(auto_attribs=True)
class GaussVelocityDeficit(BaseModel):

    alpha: float = float_attrib(default=0.58)
    beta: float = float_attrib(default=0.077)
    ka: float = float_attrib(default=0.38)
    kb: float = float_attrib(default=0.004)
    model_string: str = model_attrib(default="gauss")

    def prepare_function(
        self,
        grid: TurbineGrid,
        farm: Farm,
        flow_field: FlowField
    ) -> Dict[str, Any]:

        reference_rotor_diameter = farm.reference_turbine_diameter * np.ones(
            (
                flow_field.n_wind_directions,
                flow_field.n_wind_speeds,
                grid.n_turbines,
                1,
                1
            )
        )
        reference_hub_height = farm.hub_height[0, 0, 0] * np.ones(
            (
                flow_field.n_wind_directions,
                flow_field.n_wind_speeds,
                grid.n_turbines,
                1,
                1
            )
        )

        kwargs = dict(
            x=grid.x,
            y=grid.y,
            z=grid.z,
            reference_hub_height=reference_hub_height,
            reference_rotor_diameter=reference_rotor_diameter,
            yaw_angle=farm.farm_controller.yaw_angles,
            u_initial=flow_field.u_initial,
            wind_veer=flow_field.wind_veer
        )
        return kwargs

    def function(
        self,
        i: int,
        deflection_field: np.ndarray,
        turbine_ai: np.ndarray,
        turbulence_intensity: np.ndarray,
        Ct: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        reference_hub_height: float,
        reference_rotor_diameter: np.ndarray,
        yaw_angle: np.ndarray,
        u_initial: np.ndarray,
        wind_veer: float
    ) -> None:

        # yaw_angle is all turbine yaw angles for each wind speed
        # Extract and broadcast only the current turbine yaw setting
        # for all wind speeds
        yaw_angle = -1 * yaw_angle[:, :, i:i+1, None, None]  # Opposite sign convention in this model

        # Ct is given for only the current turbine, so broadcast
        # this to the grid dimesions
        Ct = Ct[:, :, :, None, None] * np.ones((1,1,1,5,5))

        # Construct arrays for the current turbine's location
        x_i = np.mean(x[:, :, i:i+1], axis=(3,4))
        x_i = x_i[:, :, :, None, None]
        y_i = np.mean(y[:, :, i:i+1], axis=(3,4))
        y_i = y_i[:, :, :, None, None]
        z_i = np.mean(z[:, :, i:i+1], axis=(3,4))
        z_i = z_i[:, :, :, None, None]


        # Initialize the velocity deficit
        uR = u_initial * Ct / ( 2.0 * (1 - np.sqrt(1 - Ct) ) )
        u0 = u_initial * np.sqrt(1 - Ct)

        # Initial lateral bounds
        sigma_z0 = reference_rotor_diameter * 0.5 * np.sqrt(uR / (u_initial + u0))
        sigma_y0 = sigma_z0 * cosd(yaw_angle) * cosd(wind_veer)


        # Compute the bounds of the near and far wake regions and a mask

        # Start of the near wake
        xR = x_i

        # Start of the far wake
        x0 = reference_rotor_diameter * cosd(yaw_angle) * (1 + np.sqrt(1 - Ct) )
        x0 /= np.sqrt(2) * (4 * self.alpha * turbulence_intensity + 2 * self.beta * (1 - np.sqrt(1 - Ct) ) )
        x0 += x_i

        # Masks
        near_wake_mask = np.array(x > xR) * np.array(x < x0)  # This mask defines the near wake; keeps the areas downstream of xR and upstream of x0
        far_wake_mask = np.array(x >= x0)


        # Compute the velocity deficit in the NEAR WAKE region
        # TODO: for the turbinegrid, do we need to do this near wake calculation at all?
        #       same question for any grid with a resolution larger than the near wake region

        # Calculate the wake expansion
        near_wake_ramp_up = (x - xR) / (x0 - xR)  # This is a linear ramp from 0 to 1 from the start of the near wake to the start of the far wake.
        near_wake_ramp_down = (x0 - x) / (x0 - xR)  # Another linear ramp, but positive upstream of the far wake and negative in the far wake; 0 at the start of the far wake
        # near_wake_ramp_down = -1 * (near_wake_ramp_up - 1)  # TODO: this is equivalent, right?

        sigma_y = near_wake_ramp_down * 0.501 * reference_rotor_diameter * np.sqrt(Ct / 2.0) + near_wake_ramp_up * sigma_y0
        sigma_y = sigma_y * np.array(x >= xR) + np.ones_like(sigma_y) * np.array(x < xR) * 0.5 * reference_rotor_diameter
        
        sigma_z = near_wake_ramp_down * 0.501 * reference_rotor_diameter * np.sqrt(Ct / 2.0) + near_wake_ramp_up * sigma_z0
        sigma_z = sigma_z * np.array(x >= xR) + np.ones_like(sigma_z) * np.array(x < xR) * 0.5 * reference_rotor_diameter

        r, C = rC(
            wind_veer,
            sigma_y,
            sigma_z,
            y,
            y_i,
            deflection_field,
            z,
            reference_hub_height,
            Ct,
            yaw_angle,
            reference_rotor_diameter
        )

        near_wake_deficit = gaussian_function(C, r, 1, np.sqrt(0.5))
        near_wake_deficit *= near_wake_mask


        # Compute the velocity deficit in the FAR WAKE region

        # Wake expansion in the lateral (y) and the vertical (z)
        ky = self.ka * turbulence_intensity + self.kb  # wake expansion parameters
        kz = self.ka * turbulence_intensity + self.kb  # wake expansion parameters
        sigma_y = (ky * (x - x0) + sigma_y0) * far_wake_mask + sigma_y0 * np.array(x < x0)
        sigma_z = (kz * (x - x0) + sigma_z0) * far_wake_mask + sigma_z0 * np.array(x < x0)

        r, C = rC(
            wind_veer,
            sigma_y,
            sigma_z,
            y,
            y_i,
            deflection_field,
            z,
            reference_hub_height,
            Ct,
            yaw_angle,
            reference_rotor_diameter
        )

        far_wake_deficit = gaussian_function(C, r, 1, np.sqrt(0.5))
        far_wake_deficit *= far_wake_mask


        # Combine the near and far wake regions
        velocity_deficit = np.sqrt(near_wake_deficit ** 2 + far_wake_deficit ** 2)

        return velocity_deficit


def rC(wind_veer, sigma_y, sigma_z, y, y_i, delta, z, HH, Ct, yaw, D):
    a = cosd(wind_veer) ** 2 / (2 * sigma_y ** 2) + sind(wind_veer) ** 2 / (2 * sigma_z ** 2)
    b = -sind(2 * wind_veer) / (4 * sigma_y ** 2) + sind(2 * wind_veer) / (4 * sigma_z ** 2)
    c = sind(wind_veer) ** 2 / (2 * sigma_y ** 2) + cosd(wind_veer) ** 2 / (2 * sigma_z ** 2)
    r = (
        a * (y - y_i - delta) ** 2
        - 2 * b * (y - y_i - delta) * (z - HH)
        + c * (z - HH) ** 2
    )
    C = 1 - np.sqrt( np.clip(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / D ** 2)), 0.0, 1.0) )
    return r, C

def mask_upstream_wake(mesh_y_rotated, x_coord_rotated, y_coord_rotated, turbine_yaw):
    yR = mesh_y_rotated - y_coord_rotated
    xR = yR * tand(turbine_yaw) + x_coord_rotated
    return xR, yR

def gaussian_function(C, r, n, sigma):
    return C * np.exp(-1 * r ** n / (2 * sigma ** 2))
