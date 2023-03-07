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

from floris.simulation import BaseModel


@define
class NoneWakeTurbulence(BaseModel):
    """
    The None wake turbulence model is a placeholder code that simple ignores
    any wake turbulence and just returns an array of the ambient TIs.
    """

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
        """Return unchanged field of turbulence intensities"""
        self.logger.info(
            "The wake-turbulence model is set to 'none'. Turbulence model disabled."
        )
        return np.ones_like(x) * ambient_TI
