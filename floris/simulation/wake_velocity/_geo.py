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

from floris.simulation import (
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
class _GaussGeometricVelocityDeficit(BaseModel):

    wake_expansion_rates: list = field(default=[0.0, 0.012])
    breakpoints_D: list = field(default=[5])
    sigma_y0_D: float = field(default=0.28)
    smoothing_length_D: float = field(default=2.0)
    mixing_gain_velocity: float = field(default=2.5) 

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = dict(
            x=grid.x_sorted,
            y=grid.y_sorted,
            z=grid.z_sorted,
            u_initial=flow_field.u_initial_sorted,
            wind_veer=flow_field.wind_veer
        )
        return kwargs

    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        axial_induction_i: np.ndarray,
        deflection_field_y_i: np.ndarray,
        deflection_field_z_i: np.ndarray,
        yaw_angle_i: np.ndarray,
        tilt_angle_i: np.ndarray,
        mixing_i: np.ndarray,
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
        wind_veer: float
    ) -> None:

        include_mirror_wake = True # Could add this as a user preference.

        # Only symmetric terms using yaw, but keep for consistency
        yaw_angle = -1 * yaw_angle_i

        # Initial wake widths
        sigma_y0 = self.sigma_y0_D * rotor_diameter_i * cosd(yaw_angle)
        sigma_z0 = self.sigma_y0_D * rotor_diameter_i * cosd(tilt_angle_i)

        # No specific near, far wakes in this model
        downstream_mask = np.array(x > x_i + 0.1)
        upstream_mask = np.array(x < x_i - 0.1)

        # Wake expansion in the lateral (y) and the vertical (z)
        # TODO: could compute shared components in sigma_z, sigma_y 
        # with one function call.
        sigma_y = _geometric_model_wake_width(
            x-x_i, 
            self.wake_expansion_rates, 
            [b*rotor_diameter_i for b in self.breakpoints_D], # .flatten()[0]
            sigma_y0, 
            self.smoothing_length_D*rotor_diameter_i,
            self.mixing_gain_velocity*mixing_i,
        )
        sigma_y[upstream_mask] = \
            np.tile(sigma_y0, np.shape(sigma_y)[2:])[upstream_mask]
        
        sigma_z = _geometric_model_wake_width(
            x-x_i, 
            self.wake_expansion_rates, 
            [b*rotor_diameter_i for b in self.breakpoints_D], # .flatten()[0]
            sigma_z0, 
            self.smoothing_length_D*rotor_diameter_i,
            self.mixing_gain_velocity*mixing_i,
        )
        sigma_z[upstream_mask] = \
            np.tile(sigma_z0, np.shape(sigma_z)[2:])[upstream_mask]
        
        # 'Standard' wake component
        r, C = rCalt(
            wind_veer,
            sigma_y,
            sigma_z,
            y,
            y_i,
            deflection_field_y_i,
            deflection_field_z_i,
            z,
            hub_height_i,
            ct_i,
            yaw_angle,
            tilt_angle_i,
            rotor_diameter_i,
            sigma_y0,
            sigma_z0
        )
        # Normalize to match end of acuator disk model tube
        C = C / (8*(self.sigma_y0_D**2))
        
        wake_deficit = gaussian_function(C, r, 1, np.sqrt(0.5))
        
        if include_mirror_wake:
            # TODO: speed up this option by calculating various elements in 
            #       rCalt only once.
            # Mirror component
            r_mirr, C_mirr = rCalt(
                wind_veer, # TODO: Is veer OK with mirror wakes?
                sigma_y,
                sigma_z,
                y,
                y_i,
                deflection_field_y_i,
                deflection_field_z_i,
                z,
                -hub_height_i, # Turbine at negative hub height location
                ct_i,
                yaw_angle,
                tilt_angle_i,
                rotor_diameter_i,
                sigma_y0,
                sigma_z0
            )
            # Normalize to match end of acuator disk model tube
            C_mirr = C_mirr / (8*(self.sigma_y0_D**2))
            
            # ASSUME sum-of-squares superposition for the real and mirror wakes
            wake_deficit = np.sqrt(
                wake_deficit**2 + 
                gaussian_function(C_mirr, r_mirr, 1, np.sqrt(0.5))**2
            )

        velocity_deficit = wake_deficit * downstream_mask

        return velocity_deficit

def rCalt(wind_veer, sigma_y, sigma_z, y, y_i, delta_y, delta_z, z, HH, Ct, yaw, tilt, D, sigma_y0, sigma_z0):

    ## Numexpr
    wind_veer = np.deg2rad(wind_veer)
    a = ne.evaluate("cos(wind_veer) ** 2 / (2 * sigma_y ** 2) + sin(wind_veer) ** 2 / (2 * sigma_z ** 2)")
    b = ne.evaluate("-sin(2 * wind_veer) / (4 * sigma_y ** 2) + sin(2 * wind_veer) / (4 * sigma_z ** 2)")
    c = ne.evaluate("sin(wind_veer) ** 2 / (2 * sigma_y ** 2) + cos(wind_veer) ** 2 / (2 * sigma_z ** 2)")
    r = ne.evaluate("a * ( (y - y_i - delta_y) ** 2) - 2 * b * (y - y_i - delta_y) * (z - HH - delta_z) + c * ((z - HH - delta_z) ** 2)")
    d = 1 - Ct * (sigma_y0 * sigma_z0)/(sigma_y * sigma_z) * cosd(yaw) * cosd(tilt)
    C = ne.evaluate("1 - sqrt(d)")
    return r, C

def gaussian_function(C, r, n, sigma):
    return C * np.exp(-1 * r ** n / (2 * sigma ** 2))

def sigmoid_integral(x, center=0, width=1):
    w = width/(2*np.log(0.95/0.05))

    # np.exp causes numerical issues; simply return the limit value if x large
    y1 = np.zeros_like(x)
    eval_sig_int = ((x-center/w) <= 10000)
    y1[eval_sig_int] = (w*np.log(np.exp((x[eval_sig_int]-center)/w) + 1)).flatten()
    y1[~eval_sig_int] = (x[~eval_sig_int]-center).flatten()

    return y1

def _geometric_model_wake_width(
    x, 
    wake_expansion_rates, 
    breakpoints, 
    sigma_0, 
    smoothing_length,
    mixing_final,
    ):
    assert len(wake_expansion_rates) == len(breakpoints) + 1, \
        "Invalid combination of wake_expansion_rates and breakpoints."

    sigma = (wake_expansion_rates[0] + mixing_final) * x + sigma_0
    for ib, b in enumerate(breakpoints):
        sigma += (wake_expansion_rates[ib+1]-wake_expansion_rates[ib]) * \
            sigmoid_integral(x, center=b, width=smoothing_length) 

    return sigma