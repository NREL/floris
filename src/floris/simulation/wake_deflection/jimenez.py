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

from floris.simulation import Grid
from floris.simulation import FlowField
from floris.simulation import Farm
from floris.utilities import cosd, sind, float_attrib, model_attrib
from floris.simulation import BaseClass


@attr.s(auto_attribs=True)
class JimenezVelocityDeflection(BaseClass):
    """
    Jiménez wake deflection model, dervied from
    :cite:`jdm-jimenez2010application`.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: jdm-
    """

    kd: float = float_attrib(default=0.05)
    ad: float = float_attrib(default=0.0)
    bd: float = float_attrib(default=0.0)
    model_string: str = model_attrib(default="jimenez")

    def prepare_function(
        self,
        grid: Grid,
        farm: Farm,
        flow_field: FlowField
    ) -> Dict[str, Any]:
        reference_rotor_diameter = farm.reference_turbine_diameter * np.ones(
            (
                flow_field.n_wind_directions,
                flow_field.n_wind_speeds,
                *grid.template_grid.shape
            )
        )
        kwargs = dict(
            x=grid.x,
            reference_rotor_diameter=reference_rotor_diameter,
        )
        return kwargs

    def function(
        self,
        x_i: np.ndarray,
        yaw_i: np.ndarray,
        ct_i: np.ndarray,
        *,
        x: np.ndarray,
        reference_rotor_diameter: np.ndarray,
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
        x_locations = x - x_i

        # yaw displacement
        A = 15 * (2 * self.kd * x_locations / reference_rotor_diameter + 1) ** 4.0 + xi_init ** 2.0
        B = (30 * self.kd / reference_rotor_diameter) * (
            2 * self.kd * x_locations / reference_rotor_diameter + 1
        ) ** 5.0
        C = xi_init * reference_rotor_diameter * (15 + xi_init ** 2.0)
        D = 30 * self.kd

        yYaw_init = (xi_init * A / B) - (C / D)

        # corrected yaw displacement with lateral offset
        # This has the same shape as the grid
        deflection = yYaw_init + self.ad + self.bd * x_locations

        x = np.unique(x_locations)
        for i in range(len(x)):
            tmp = np.max(deflection[x_locations == x[i]])
            deflection[x_locations == x[i]] = tmp

        return deflection
