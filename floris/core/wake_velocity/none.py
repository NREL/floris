
from typing import Any, Dict

import numpy as np
from attrs import define, field

from floris.core import (
    BaseModel,
    FlowField,
    Grid,
)


@define
class NoneVelocityDeficit(BaseModel):
    """
    The None deficit model is a placeholder code that simple ignores any
    wake wind speed deficits and returns an array of zeroes.
    """

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = {
            "u_initial": flow_field.u_initial_sorted,
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
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        hub_height_i: float,
        rotor_diameter_i: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        u_initial: np.ndarray,
    ) -> None:
        self.logger.warning("The wake deficit model is set to 'none'. Wake modeling disabled.")
        return np.zeros_like(u_initial)
