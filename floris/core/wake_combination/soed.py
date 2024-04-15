import logging
from typing import Any, Dict

import numpy as np
from attrs import define

from floris.core import BaseModel, FlowField


logger = logging.getLogger(name="floris")

@define
class SOED(BaseModel):
    """
    Sum of energy deficit, as described in Kuo et al, is used by the eddy vicosity model.

    Kuo et al, 2014
    https://mechanicaldesign.asmedigitalcollection.asme.org/IMECE/proceedings/IMECE2014/46521/V06BT07A074/263017
    """

    def function(
        self,
        U_tilde_field: np.ndarray,
    ):
        n_turbines = U_tilde_field.shape[1]

        U_tilde_combined = np.sqrt(1 - n_turbines + np.sum(U_tilde_field**2, axis=1))

        if (U_tilde_combined < 0).any() or np.isnan(U_tilde_combined).any():
            logger.warning(
                "Negative or NaN values detected in combined velocity deficit field. "
                "These values will be set to zero."
            )
            U_tilde_combined[U_tilde_combined < 0] = 0
            U_tilde_combined[np.isnan(U_tilde_combined)] = 0

        return U_tilde_combined
