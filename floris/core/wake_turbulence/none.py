
from typing import Any, Dict

import numpy as np
from attrs import define, field

from floris.core import BaseModel


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
        return np.zeros_like(x)
