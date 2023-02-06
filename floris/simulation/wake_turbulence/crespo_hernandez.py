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
class CrespoHernandez(BaseModel):
    """
    CrespoHernandez is a wake-turbulence model that is used to compute
    additional variability introduced to the flow field by operation of a wind
    turbine. Implementation of the model follows the original formulation and
    limitations outlined in :cite:`cht-crespo1996turbulence`.

    Args:
        parameter_dictionary (dict): Model-specific parameters.
            Default values are used when a parameter is not included
            in `parameter_dictionary`. Possible key-value pairs include:

            -   **initial** (*float*): The initial ambient turbulence
                intensity, expressed as a decimal fraction.
            -   **constant** (*float*): The constant used to scale the
                wake-added turbulence intensity.
            -   **ai** (*float*): The axial induction factor exponent used
                in in the calculation of wake-added turbulence.
            -   **downstream** (*float*): The exponent applied to the
                distance downstream of an upstream turbine normalized by
                the rotor diameter used in the calculation of wake-added
                turbulence.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: cht-
    """
    initial: float = field(converter=float, default=0.1)
    constant: float = field(converter=float, default=0.9)
    ai: float = field(converter=float, default=0.8)
    downstream: float = field(converter=float, default=-0.32)

    def prepare_function(self) -> dict:
        pass

    def function(
        self,
        ambient_TI: float,
        x: np.ndarray,
        x_i: np.ndarray,
        rotor_diameter: float,
        axial_induction: np.ndarray,
    ) -> None:
        # Replace zeros and negatives with 1 to prevent nans/infs
        delta_x = np.array(x - x_i)

        # TODO: ensure that these fudge factors are needed for different rotations
        upstream_mask = np.array(delta_x <= 0.1)
        downstream_mask = np.array(delta_x > -0.1)

        #        Keep downstream components          Set upstream to 1.0
        delta_x = delta_x * downstream_mask + np.ones_like(delta_x) * np.array(upstream_mask)

        # turbulence intensity calculation based on Crespo et. al.
        ti = (
            self.constant
        * axial_induction ** self.ai
        * ambient_TI ** self.initial
        * ((delta_x) / rotor_diameter) ** self.downstream
        )
        # Mask the 1 values from above with zeros
        return ti * np.array(downstream_mask)
