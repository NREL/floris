
import numpy as np
from attrs import define

from floris.core import BaseModel


@define
class FLS(BaseModel):
    """
    FLS uses freestream linear superposition to apply the wake velocity
    deficits to the freestream flow field.
    """

    def prepare_function(self) -> dict:
        pass

    def function(self, wake_field: np.ndarray, velocity_field: np.ndarray):
        """
        Combines the base flow field with the velocity deficits
        using freestream linear superposition. In other words, the wake
        field and base fields are simply added together.

        Args:
            u_field (np.array): The base flow field.
            u_wake (np.array): The wake to apply to the base flow field.

        Returns:
            np.array: The resulting flow field after applying the wake to the
                base.
        """
        return wake_field + velocity_field
