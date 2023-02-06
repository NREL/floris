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
from floris.utilities import cosd, sind


@define
class JimenezVelocityDeflection(BaseModel):
    """
    Jiménez wake deflection model, dervied from
    :cite:`jdm-jimenez2010application`.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: jdm-
    """

    kd: float = field(default=0.05)
    ad: float = field(default=0.0)
    bd: float = field(default=0.0)

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
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        rotor_diameter_i: np.ndarray,
        *,
        x: np.ndarray,
    ):
        """
        Calcualtes the deflection field of the wake in relation to the yaw of
        the turbine. This is coded as defined in [1].

        Args:
            x_locations (np.array): streamwise locations in wake
            y_locations (np.array): spanwise locations in wake
            z_locations (np.array): vertical locations in wake
                (not used in Jiménez)
            turbine (:py:class:`floris.simulation.turbine.Turbine`):
                Turbine object
            coord
                (:py:meth:`floris.simulation.turbine_map.TurbineMap.coords`):
                Spatial coordinates of wind turbine.
            flow_field
                (:py:class:`floris.simulation.flow_field.FlowField`):
                Flow field object.

        Returns:
            deflection (np.array): Deflected wake centerline.


        This function calculates the deflection of the entire flow field
        given the yaw angle and Ct of the current turbine
        """

        # NOTE: Its important to remember the rules of broadcasting here.
        # An operation between two np.arrays of different sizes involves
        # broadcasting. First, the rank and then the dimensions are compared.
        # If the ranks are different, new dimensions of size 1 are added to
        # the missing dimensions. Then, arrays can be combined (arithmetic)
        # if corresponding dimensions are either the same size or 1.
        # https://numpy.org/doc/stable/user/basics.broadcasting.html
        # Here, many dimensions are 1, but these are essentially treated
        # as a scalar value for that dimension.

        # angle of deflection
        xi_init = cosd(yaw_i) * sind(yaw_i) * ct_i / 2.0

        """
        delta_x = x - x_i

        # yaw displacement
        A = 15 * (2 * self.kd * delta_x / rotor_diameter_i + 1) ** 4.0 + xi_init ** 2.0
        B = (30 * self.kd / rotor_diameter_i)
        B *= ( 2 * self.kd * delta_x / rotor_diameter_i + 1 ) ** 5.0
        C = xi_init * rotor_diameter_i * (15 + xi_init ** 2.0)
        D = 30 * self.kd

        yYaw_init = (xi_init * A / B) - (C / D)

        # corrected yaw displacement with lateral offset
        # This has the same shape as the grid

        deflection = yYaw_init + self.ad + self.bd * delta_x
        """

        # Numexpr - do not change below without corresponding changes above.
        kd = self.kd
        ad = self.ad
        bd = self.bd

        delta_x = ne.evaluate("x - x_i")
        A = ne.evaluate("15 * (2 * kd * delta_x / rotor_diameter_i + 1) ** 4.0 + xi_init ** 2.0")
        B = ne.evaluate("(30 * kd / rotor_diameter_i)")
        B = ne.evaluate("B * ( 2 * kd * delta_x / rotor_diameter_i + 1 ) ** 5.0")
        C = ne.evaluate("xi_init * rotor_diameter_i * (15 + xi_init ** 2.0)")
        D = ne.evaluate("30 * kd")

        yYaw_init = ne.evaluate("(xi_init * A / B) - (C / D)")
        deflection = ne.evaluate("yYaw_init + ad + bd * delta_x")

        return deflection
