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
from scipy.special import gamma

from floris.simulation import (
    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)


@define
class SuperGaussianVAWTVelocityDeficit(BaseModel):
    """
    This is a super-Gaussian wake velocity model for vertical-axis wind turbines (VAWTs).
    The model is based on :cite:`ouro2021theoretical` and allows the wake to have
    different characteristics in the cross-stream (y) and vertical direction (z). The initial
    wake shape is closely related to the turbine cross section, which is:
        rotor diameter * length of the vertical turbine blades.

    Parameters:
        wake_expansion_coeff_y: The wake expands linearly in y with a rate of
            wake_expansion_coeff_y * turbulence_intensity_i.
        wake_expansion_coeff_z: The wake expands linearly in z with a rate of
            wake_expansion_coeff_z * turbulence_intensity_i.
        ay, by, cy: Parameters that control how the shape function exponent `ny` evolves
            in the streamwise direction.
        az, bz, cz: Parameters that control how the shape function exponent `nz` evolves
            in the streamwise direction.
    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
    """
    wake_expansion_coeff_y: float = field(default=0.50)
    wake_expansion_coeff_z: float = field(default=0.50)
    ay: float = field(default=0.95)
    az: float = field(default=4.5)
    by: float = field(default=0.35)
    bz: float = field(default=0.70)
    cy: float = field(default=2.4)
    cz: float = field(default=2.4)

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
            "wind_veer": flow_field.wind_veer
        }
        return kwargs

    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        hub_height_i: float,
        rotor_diameter_i: np.ndarray,
        vawt_blade_length_i: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        wind_veer: float
    ) -> None:

        # Model parameters need to be unpacked for use in Numexpr
        ay, by, cy = self.ay, self.by, self.cy
        az, bz, cz = self.az, self.bz, self.cz

        # No specific near or far wakes in this model
        downstream_mask = (x > x_i + 0.1)

        # Nondimensional coordinates. Is z_i always equal to hub_height_i?
        x_tilde =   ne.evaluate("downstream_mask * (x - x_i) / rotor_diameter_i")
        x_tilde_H = ne.evaluate("downstream_mask * (x - x_i) / vawt_blade_length_i")
        y_tilde =   ne.evaluate("downstream_mask * (y - y_i) / rotor_diameter_i")
        z_tilde =   ne.evaluate("downstream_mask * (z - z_i) / vawt_blade_length_i")

        # Wake expansion rates
        ky_star = self.wake_expansion_coeff_y * turbulence_intensity_i
        kz_star = self.wake_expansion_coeff_z * turbulence_intensity_i

        # `beta` is the initial wake expansion relative to the turbine cross-section
        fac = np.sqrt(1 - ct_i)
        beta = 0.5 * (1 + fac) / fac

        # `epsilon` is the characteristic nondimensional wake width at x - x_i = 0
        ny_0 = ay + cy
        nz_0 = az + cz
        eta_0 = 1/ny_0 + 1/nz_0
        epsilon = beta * ny_0 * nz_0 / (2 ** (2 * eta_0 + 2) * gamma(1 / ny_0) * gamma(1 / nz_0))
        epsilon **= 1 / (2 * eta_0)

        # Characteristic nondimensional wake widths grow linearly in the streamwise direction
        sigma_y_tilde = ne.evaluate("ky_star * x_tilde + epsilon")
        sigma_z_tilde = ne.evaluate("kz_star * x_tilde_H + epsilon")

        # Exponents which determine the wake shape
        ny = ne.evaluate("ay * exp(-by * x_tilde) + cy")
        nz = ne.evaluate("az * exp(-bz * x_tilde_H) + cz")

        # At any streamwise location x behind the turbine, the velocity deficit in the
        # yz-plane is given by multiplying the maximum deficit C = C(x) with two super-Gaussian
        # shape functions fy = fy(y) and fz = fz(z)
        eta = ne.evaluate("1 / ny + 1 / nz")
        ny_inv = ne.evaluate("1 / ny")
        nz_inv = ne.evaluate("1 / nz")
        gamma_y, gamma_z = gamma(ny_inv), gamma(nz_inv)
        fac =  ne.evaluate(
            "8 * sigma_y_tilde ** (2 / ny) * sigma_z_tilde ** (2 / nz) * gamma_y * gamma_z"
        )
        C = ne.evaluate(
            "2 ** (eta - 1) - sqrt(2 ** (2 * eta - 2) - ct_i * ny * nz / fac)"
        )

        fy = ne.evaluate("exp(-abs(y_tilde) ** ny / (2 * sigma_y_tilde ** 2))")
        fz = ne.evaluate("exp(-abs(z_tilde) ** nz / (2 * sigma_z_tilde ** 2))")

        return ne.evaluate("C * fy * fz * downstream_mask")
