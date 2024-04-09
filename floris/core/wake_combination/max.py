
import numpy as np
from attrs import define

from floris.core import BaseModel


@define
class MAX(BaseModel):
    """
    MAX uses the maximum wake velocity deficit to add to the
    base flow field. For more information, refer to
    :cite:`max-gunn2016limitations`.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: max-
    """

    def prepare_function(self) -> dict:
        pass

    def function(self, wake_field: np.ndarray, velocity_field: np.ndarray):
        """
        Incorporates the velocity deficits into the base flow field by
        selecting the maximum of the two for each point.

        Args:
            u_field (np.array): The base flow field.
            u_wake (np.array): The wake to apply to the base flow field.

        Returns:
            np.array: The resulting flow field after applying the wake to the
                base.
        """
        return np.maximum(wake_field, velocity_field)
