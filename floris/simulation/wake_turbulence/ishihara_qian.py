# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import annotations

from typing import Any

import attrs
import numpy as np
import numexpr as ne
from attrs import field, define
from numpy import exp  # noqa: F401

from floris.type_dec import FromDictMixin
from floris.utilities import Vec3


@define(auto_attribs=True)
class IQParam(FromDictMixin):
    """A parameter mapping for all parameters of the Ishihara Qian model.

    Args:
        const (float): The constant coefficient.
        Ct (float): The thrust coefficient exponent.
        TI (float): The turbulence intensity exponent.
    """

    const: float = field(converter=float, validator=attrs.validators.instance_of(float))
    Ct: float = field(converter=float, validator=attrs.validators.instance_of(float))
    TI: float = field(converter=float, validator=attrs.validators.instance_of(float))

    def current_value(self, Ct: float, ti_initial: float) -> float:
        """
        Calculates model parameters using current conditions and parameter settings.

        Args:
            Ct (float): Thrust coefficient of the current turbine.
            ti_initial (float): Turbulence intensity.

        Returns:
            float: Current value of model parameter.
        """
        return self.const * Ct**self.Ct * ti_initial**self.TI


@define(auto_attribs=True)
class IshiharaQian:
    """
    IshiharaQian is a wake velocity subclass that is used to compute the wake
    velocity deficit based on the Gaussian wake model with self-similarity and
    a near wake correction. The IshiharaQian wake model includes a Gaussian
    wake velocity deficit profile in the spanwise and vertical directions and
    includes the effects of ambient turbulence, added turbulence from upstream
    wakes, as well as wind shear and wind veer. For more info, see
    :cite:`iqt-qian2018new`.

    Args:
        kstar (dict): A dictionary of the `const`, `Ct`, and `TI` parameters related to
            the linear relationship between the turbulence intensity and the width of
            the Gaussian wake shape.
        epsilon (dict): A dictionary of the `const`, `Ct`, and `TI` parameters of the
            second parameter used to determine the linear relationship between the
            turbulence intensity and the width of the Gaussian wake shape.
        d (dict): A dictionary of the `const`, `Ct`, and `TI` parameters of the constant
            coefficient used in calculation of wake-added turbulence.
        e (dict): A dictionary of the `const`, `Ct`, and `TI` parameters of the linear
            coefficient used in calculation of wake-added turbulence.
        f (dict): A dictionary of the `const`, `Ct`, and `TI` parameters of the near-wake
            coefficient used in calculation of wake-added turbulence.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: iqt-
    """

    kstar: IQParam = field(converter=IQParam.from_dict, default={"const": 0.11, "Ct": 1.07, "TI": 0.2})
    epsilon: IQParam = field(converter=IQParam.from_dict, default={"const": 0.23, "Ct": -0.25, "TI": 0.17})
    d: IQParam = field(converter=IQParam.from_dict, default={"const": 2.3, "Ct": 1.2, "TI": 0.0})
    e: IQParam = field(converter=IQParam.from_dict, default={"const": 1.0, "Ct": 0.0, "TI": 0.1})
    f: IQParam = field(converter=IQParam.from_dict, default={"const": 0.7, "Ct": -3.2, "TI": -0.45})

    def prepare_function(self) -> dict[str, Any]:
        kwargs = dict()
        return kwargs

    def function(
        self,
        ambient_TI: float,
        coord_ti: Vec3,
        turbine_coord: Vec3,
        rotor_diameter_i: float,
        hub_height_i: float,
        Ct_i: float,
    ) -> np.ndarray:
        # function(self, x_locations, y_locations, z_locations, turbine,
        #  turbine_coord, flow_field, turb_u_wake, sorted_map):
        """
        Calculates wake-added turbulence as a function of
        external conditions and wind turbine operation. This function is
        accessible through the :py:class:`~.wake.Wake` class as the
        :py:meth:`~.Wake.turbulence_function` method.
        Args:
            ambient_TI (float): TI of the background flow field.
            coord_ti (:py:class:`~.utilities.Vec3`): Coordinate where TI
                is to be calculated (e.g. downstream wind turbines).
            turbine_coord (:py:class:`~.utilities.Vec3`): Coordinate of
                the wind turbine adding turbulence to the flow.
            turbine (:py:class:`~.turbine.Turbine`): Wind turbine
                adding turbulence to the flow.
        Returns:
            float: Wake-added turbulence from the current wind turbine
            (**turbine**) at location specified by (**coord_ti**).
        """
        # added turbulence model
        ti_initial = ambient_TI

        # turbine parameters
        D = rotor_diameter_i
        HH = hub_height_i
        Ct = Ct_i

        local_x, local_y, local_z = coord_ti.elements - turbine_coord.elements

        # coordinate info
        r = np.sqrt(local_y**2 + local_z**2)

        kstar = self.kstar.current_value(Ct, ti_initial)
        epsilon = self.epsilon.current_value(Ct, ti_initial)

        d = self.d.current_value(Ct, ti_initial)  # noqa: F841
        e = self.e.current_value(Ct, ti_initial)  # noqa: F841
        f = self.f.current_value(Ct, ti_initial)  # noqa: F841

        k1 = 1.0  # noqa: F841
        k2 = 0.0  # noqa: F841
        if r / D <= 0.5:
            # TODO: make work for array of turbulences/grid points
            # k1[r / D > 0.5] = 1.0
            # k1 = np.where(k1, r / D <= 0.5, 1.0)
            # k2[r / D > 0.5] = 0.0
            # k1 = np.where(k1, r / D <= 0.5, 0.0)
            k1 = np.cos(np.pi / 2 * (r / D - 0.5)) ** 2  # noqa: F841
            k2 = np.cos(np.pi / 2 * (r / D + 0.5)) ** 2  # noqa: F841

        # Representative wake width = \sigma / D
        wake_width = kstar * (local_x / D) + epsilon  # noqa: F841

        # Added turbulence intensity = \Delta I_1 (x,y,z)
        delta = ti_initial * np.sin(np.pi * (HH - local_z) / HH) ** 2
        # delta = np.where(local_z < HH, delta, 0.0)  # TODO: get working for array math

        ti_calculation = (
            1
            / ne.evaluate("d + e * (local_x / D) + f * (1 + (local_x / D)) ** (-2)")
            * (
                ne.evaluate("k1 * exp(-((r - D / 2) ** 2) / (2 * (wake_width * D) ** 2))")
                + ne.evaluate("k2 * exp(-((r + D / 2) ** 2) / (2 * (wake_width * D) ** 2))")
            )
            - delta
        )

        return ti_calculation
