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
class TurboparkgaussVelocityDeficit(BaseModel):

    A: float = field(default=0.04)
    sigma_max_rel: float = field(default=4.0)

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

        downstream_mask = (x - x_i >= self.NUM_EPS)
        x_dist = (x - x_i) * downstream_mask / rotor_diameter_i

        # Characteristic wake widths from all turbines relative to turbine i
        sigma = characteristic_wake_width(
            x_dist, rotor_diameter_i, turbulence_intensity_i, ct_i, self.A
        )

        # Peak wake deficits
        val = np.clip(1 - ct_i / (8 * (sigma / rotor_diameter_i) ** 2), 0.0, 1.0)
        C = 1 - np.sqrt(val)

        r_dist = np.sqrt((y - y_i) ** 2 + (z - z_i) ** 2)
        r_dist_image = np.sqrt((y - y_i) ** 2 + (z - 3*z_i) ** 2)

        # Compute deficit for all turbines and mask to keep upstream and overlapping turbines
        # NOTE self.sigma_max_rel * sigma is an effective wake width
        is_overlapping = (self.sigma_max_rel * sigma) / 2 + rotor_diameter_i / 2 > r_dist
        wtg_overlapping = (x_dist > 0) * is_overlapping

        delta_real = np.empty(np.shape(u_initial)) * np.nan
        delta_image = np.empty(np.shape(u_initial)) * np.nan

        # Compute deficits for real turbines and for mirrored (image) turbines
        delta_real  = wtg_overlapping * gaussian_function(C, r_dist, 2, sigma)
        delta_image = wtg_overlapping * gaussian_function(C, r_dist_image, 2, sigma)

        # delta = delta_real # np.concatenate((delta_real, delta_image), axis=2)# No mirror turbines
        delta = np.hypot(delta_real,delta_image) # incl mirror turbines

        velocity_deficit = np.nan_to_num(delta)

        return velocity_deficit


def characteristic_wake_width(x_dist, D, TI, Cts, A):
    # Parameter values taken from S. T. Frandsen, “Risø-R-1188(EN) Turbulence
    # and turbulence generated structural loading in wind turbine clusters”
    # Risø, Roskilde, Denmark, 2007.
    c1 = 1.5
    c2 = 0.8

    alpha = TI * c1
    beta = c2 * TI / np.sqrt(Cts)

    dw = A * TI / beta * (
        np.sqrt((alpha + beta * x_dist) ** 2 + 1)
        - np.sqrt(1 + alpha ** 2)
        - np.log(
            ((np.sqrt((alpha + beta * x_dist) ** 2 + 1) + 1) * alpha)
            / ((np.sqrt(1 + alpha ** 2) + 1) * (alpha + beta * x_dist))
        )
    )

    epsilon = 0.25 * np.sqrt( np.min( 0.5 * (1 + np.sqrt(1 - Cts)) / np.sqrt(1 - Cts) ) )
    sigma = D * (epsilon + dw)

    return sigma


def mask_upstream_wake(mesh_y_rotated, x_coord_rotated, y_coord_rotated, turbine_yaw):
    yR = mesh_y_rotated - y_coord_rotated
    xR = yR * tand(turbine_yaw) + x_coord_rotated
    return xR, yR


def gaussian_function(C, r, n, sigma):
    result = ne.evaluate("C * exp(-1 * r ** n / (2 * sigma ** 2))")
    return result
