# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np
from attrs import define

from floris.simulation import BaseModel


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
        Incorporates the velicty deficits into the base flow field by
        selecting the maximum of the two for each point.

        Args:
            u_field (np.array): The base flow field.
            u_wake (np.array): The wake to apply to the base flow field.

        Returns:
            np.array: The resulting flow field after applying the wake to the
                base.
        """
        return np.maximum(wake_field, velocity_field)
