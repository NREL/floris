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
class WakeInducedMixing(BaseModel):
    """
    WakeInducedMixing is a model used to generalize wake-added turbulence
    in the Empirical Gaussian wake model. It computes the contribution of each
    turbine to a "wake-induced mixing" term that in turn is used in the
    velocity deficit and deflection models.

    Args:
        parameter_dictionary (dict): Model-specific parameters.
            Default values are used when a parameter is not included
            in `parameter_dictionary`. Possible key-value pairs include:

            -   **atmospheric_ti_gain** (*float*): The contribution of ambient
                turbulent intensity to the wake-induced mixing term. Currently
                throws a warning if nonzero.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
    """
    atmospheric_ti_gain: float = field(converter=float, default=0.0)

    def __attrs_post_init__(self) -> None:
        if self.atmospheric_ti_gain != 0.0:
            nonzero_err_msg = \
                "Running wake_induced_mixing model with mixing contributions"+\
                " from the atmospheric turbulence intensity has not been"+\
                " vetted. To avoid this warning, set atmospheric_ti_gain=0."+\
                " in the FLORIS input yaml."
            self.logger.warning(nonzero_err_msg, stack_info=True)

    def prepare_function(self) -> dict:
        pass

    def function(
        self,
        axial_induction_i: np.ndarray,
        downstream_distance_D_i: np.ndarray,
    ) -> None:
        """
        Calculates the contribution of turbine i to all other turbines'
        mixing terms.

        Args:
            axial_induction_i (np.array): Axial induction factor of
                the ith turbine (-).
            downstream_distance_D_i (np.array): The distance downstream
                from turbine i to all other turbines (specified in terms
                of multiples of turbine i's rotor diameter) (D).

        Returns:
            np.array: Components of the wake-induced mixing term due to
                the ith turbine.
        """

        wake_induced_mixing = axial_induction_i[:,:,:,0,0] / downstream_distance_D_i**2

        return wake_induced_mixing
