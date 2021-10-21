# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import attr
import numpy as np
from numpy import newaxis as na

from src.turbine import Turbine, Ct
from src.utilities import cosd, sind
from src.utilities import float_attrib, model_attrib
from src.base_class import BaseClass


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

    def function(
        self,
        i: int,
        # *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        reference_turbine: Turbine,
        yaw_angle: float,
        Ct: float
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

        # angle of deflection
        xi_init = cosd(yaw_angle) * sind(yaw_angle) * Ct / 2.0
        x_locations = x - x[i]

        # yaw displacement
        yYaw_init = (
            xi_init
            * (15 * (2 * self.kd * x_locations / reference_turbine.rotor_diameter + 1) ** 4.0 + xi_init ** 2.0)
            / (
                (30 * self.kd / reference_turbine.rotor_diameter)
                * (2 * self.kd * x_locations / reference_turbine.rotor_diameter + 1) ** 5.0
            )
        ) - (xi_init * reference_turbine.rotor_diameter * (15 + xi_init ** 2.0) / (30 * self.kd))

        # corrected yaw displacement with lateral offset
        deflection = yYaw_init + self.ad + self.bd * x_locations

        x = np.unique(x_locations)
        for i in range(len(x)):
            tmp = np.max(deflection[x_locations == x[i]])
            deflection[x_locations == x[i]] = tmp

        return deflection
