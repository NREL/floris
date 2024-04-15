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

    # def prepare_function(
    #     self,
    #     flow_field: FlowField,
    # ) -> Dict[str, Any]:

    #     return {"u_initial": flow_field.u_initial_sorted}

    # def function(
    #         self,
    #         wake_field: np.ndarray,
    #         velocity_field: np.ndarray,
    #         *,
    #         u_initial: np.ndarray # Unless u_initial can be stored? Seems possible? # TODO
    # ) -> np.ndarray:
    #     """
    #     Combines the base flow field with the velocity deficits
    #     using sum of energy deficits.

    #     Args:
    #         wake_field (np.array): The existing wake field (as a deficit).
    #         velocity_field (np.array): The new wake to include (as a deficit).

    #     Returns:
    #         np.array: The resulting flow field after applying the new wake.
    #     """

    #     # Convert to nondimensionalized form
    #     U_tilde_field = 1 - wake_field/u_initial
    #     U_tilde_new = 1 - velocity_field/u_initial

    #     # Apply combination model
    #     U_tilde_updated = np.sqrt(U_tilde_field**2 + U_tilde_new**2 - 1)

    #     # Convert back to dimensionalized form and return
    #     return u_initial * (1 - U_tilde_updated)

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
