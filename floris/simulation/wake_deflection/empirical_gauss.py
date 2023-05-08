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
    """
    The Empirical Gauss deflection model is based on the form of previous the
    Guass deflection model (see :cite:`bastankhah2016experimental` and
    :cite:`King2019Controls`) but simplifies the formulation for simpler
    tuning and more independence from the velocity deficit model.

    parameter_dictionary (dict): Model-specific parameters.
        Default values are used when a parameter is not included
        in `parameter_dictionary`. Possible key-value pairs include:

            -   **horizontal_deflection_gain_D** (*float*): Gain for the
                maximum (y-direction) deflection acheived far downstream
                of a yawed turbine.
            -   **vertical_deflection_gain_D** (*float*): Gain for the
                maximum vertical (z-direction) deflection acheived at a
                far downstream location due to rotor tilt. Specifying as
                -1 will mean that vertical deflections due to tilt match
                horizontal deflections due to yaw.
            -   **deflection_rate** (*float*): Rate at which the
                deflected wake center approaches its maximum deflection.
            -   **mixing_gain_deflection** (*float*): Gain to set the
                reduction in deflection due to wake-induced mixing.
            -   **yaw_added_mixing_gain** (*float*): Sets the
                contribution of turbine yaw misalignment to the mixing
                in that turbine's wake (similar to yaw-added recovery).

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
    """
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

        kwargs = {
            "x": grid.x_sorted,
        }
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
    ):
        """
        Calculates the deflection field of the wake.

        Args:
            x_i (np.array): Streamwise direction grid coordinates of
                the ith turbine (m).
            y_i (np.array): Cross stream direction grid coordinates of
                the ith turbine (m) [not used].
            yaw_i (np.array): Yaw angle of the ith turbine (deg).
            tilt_i (np.array): Tilt angle of the ith turbine (deg).
            mixing_i (np.array): The wake-induced mixing term for the
                ith turbine.
            ct_i (np.array): Thrust coefficient for the ith turbine (-).
            rotor_diameter_i (np.array): Rotor diamter for the ith
                turbine (m).

            x (np.array): Streamwise direction grid coordinates of the
                flow field domain (m).

        Returns:
            np.array: Deflection field for the wake.
        """
        # ==============================================================

        deflection_gain_y = self.horizontal_deflection_gain_D * rotor_diameter_i
        if self.vertical_deflection_gain_D == -1:
            deflection_gain_z = deflection_gain_y
        else:
            deflection_gain_z = self.vertical_deflection_gain_D * rotor_diameter_i

        # Convert to radians, CW yaw for consistency with other models
        yaw_r = np.pi/180 * -yaw_i
        tilt_r = np.pi/180 * tilt_i

        A_y = (deflection_gain_y * ct_i * yaw_r) / (1 + self.mixing_gain_deflection * mixing_i)
        A_z = (deflection_gain_z * ct_i * tilt_r) / (1 + self.mixing_gain_deflection * mixing_i)

        # Apply downstream mask in the process
        x_normalized = (x - x_i) * np.array(x > x_i + 0.1) / rotor_diameter_i

        log_term = np.log(
            (x_normalized - self.deflection_rate) / (x_normalized + self.deflection_rate)
            + 2
        )

        deflection_y = A_y * log_term
        deflection_z = A_z * log_term

        return deflection_y, deflection_z

def yaw_added_wake_mixing(
    axial_induction_i,
    yaw_angle_i,
    downstream_distance_D_i,
    yaw_added_mixing_gain
):
    return (
        axial_induction_i[:,:,:,0,0]
        * yaw_added_mixing_gain
        * (1 - cosd(yaw_angle_i[:,:,:,0,0]))
        / downstream_distance_D_i**2
    )
