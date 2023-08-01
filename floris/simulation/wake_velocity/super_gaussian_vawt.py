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
    The Empirical Gauss velocity model has a Gaussian profile
    (see :cite:`bastankhah2016experimental` and
    :cite:`King2019Controls`) throughout and expands in a (smoothed)
    piecewise linear fashion.

    parameter_dictionary (dict): Model-specific parameters.
        Default values are used when a parameter is not included
        in `parameter_dictionary`. Possible key-value pairs include:

            -   **wake_expansion_rates** (*list*): List of expansion
                rates for the Gaussian wake width. Must be of length 1
                or greater.
            -   **breakpoints_D** (*list*): List of downstream
                locations, specified in terms of rotor diameters, where
                the expansion rates go into effect. Must be one element
                shorter than wake_expansion_rates. May be empty.
            -   **sigma_0_D** (*float*): Initial width of the Gaussian
                wake at the turbine location, specified as a multiplier
                of the rotor diameter.
            -   **smoothing_length_D** (*float*): Distance over which
                the corners in the piece-wise linear wake expansion rate
                are smoothed (specified as a multiplier of the rotor
                diameter).
            -   **mixing_gain_deflection** (*float*): Gain to set the
                increase in wake expansion due to wake-induced mixing.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
    """
    wake_expansion_coeff_y: float = field(default=0.45)
    wake_expansion_coeff_z: float = field(default=0.45)
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
        """
        Calculates the velocity deficits in the wake.

        Args:
            x_i (np.array): Streamwise direction grid coordinates of
                the ith turbine (m).
            y_i (np.array): Cross stream direction grid coordinates of
                the ith turbine (m).
            z_i (np.array): Vertical direction grid coordinates of
                the ith turbine (m) [not used].
            axial_induction_i (np.array): Axial induction factor of the
                ith turbine (-) [not used].
            yaw_angle_i (np.array): Yaw angle of the ith turbine (deg).
            ct_i (np.array): Thrust coefficient for the ith turbine (-).
            hub_height_i (float): Hub height for the ith turbine (m).
            rotor_diameter_i (np.array): Rotor diameter for the ith
                turbine (m).

            x (np.array): Streamwise direction grid coordinates of the
                flow field domain (m).
            y (np.array): Cross stream direction grid coordinates of the
                flow field domain (m).
            z (np.array): Vertical direction grid coordinates of the
                flow field domain (m).
            wind_veer (np.array): Wind veer (deg).

        Returns:
            np.array: Velocity deficits (-).
        """

        # Model parameters need to be unpacked for use in Numexpr
        ay, by, cy = self.ay, self.by, self.cy
        az, bz, cz = self.az, self.bz, self.cz

        # No specific near or far wakes in this model
        downstream_mask = np.array(x > x_i + 0.1)

        # Nondimensional coordinates. Is z_i always equal to hub_height_i?
        x_tilde =   ne.evaluate("downstream_mask * (x - x_i) / rotor_diameter_i")
        x_tilde_H = ne.evaluate("downstream_mask * (x - x_i) / vawt_blade_length_i")
        y_tilde =   ne.evaluate("downstream_mask * (y - y_i) / rotor_diameter_i")
        z_tilde =   ne.evaluate("downstream_mask * (z - z_i) / vawt_blade_length_i")

        # Wake expansion rates
        ky_star = self.wake_expansion_coeff_y * turbulence_intensity_i
        kz_star = self.wake_expansion_coeff_z * turbulence_intensity_i

        # Relative initial wake expansion
        fac = np.sqrt(1 - ct_i)
        beta = 0.5 * (1 + fac) / fac

        # Initial wake width
        ny_0 = ay + cy
        nz_0 = az + cz
        eta_0 = 1/ny_0 + 1/nz_0
        epsilon = beta * ny_0 * nz_0 / (2 ** (2 * eta_0 + 2) * gamma(1 / ny_0) * gamma(1 / nz_0))
        epsilon **= 1 / (2 * eta_0)
        epsilon_y, epsilon_z = epsilon, epsilon

        # Characteristic wake widths grow linearly in the streamwise direction
        sigma_y_tilde = ne.evaluate("ky_star * x_tilde + epsilon_y")
        sigma_z_tilde = ne.evaluate("kz_star * x_tilde_H + epsilon_z")

        # Gaussian exponents which determine the wake shape
        ny = ne.evaluate("ay * exp(-by * x_tilde) + cy")
        nz = ne.evaluate("az * exp(-bz * x_tilde_H) + cz")

        # At any streamwise location x behind the turbine, the velocity deficit in the
        # y-z plane is given by multiplying the maximum deficit C = C(x) with two Gaussian
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
