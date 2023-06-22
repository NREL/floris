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
        axial_induction_i: np.ndarray,
        deflection_field_i: np.ndarray,
        yaw_angle_i: np.ndarray,
        tilt_angle_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        hub_height_i: float,
        rotor_diameter_i: np.ndarray,
        #blade_length_i: np.ndarray,
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
            tilt_angle_i (np.array): Tilt angle of the ith turbine
                (deg).
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

        # Initial wake widths

        # No specific near, far wakes in this model
        downstream_mask = np.array(x > x_i + 0.1)

        # Wake expansion in the lateral (y) and the vertical (z)
        # TODO: could compute shared components in sigma_z, sigma_y
        # with one function call.
        wake_deficit = 0.5

        velocity_deficit = wake_deficit * downstream_mask

        return velocity_deficit
