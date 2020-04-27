# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np
from ...utilities import setup_logger
from .base_wake_combination import WakeCombination


class MAX(WakeCombination):
    """
    MAX is a subclass of 
    :py:class:`floris.simulation.wake_combination.WakeCombination` 
    which uses the maximum wake velocity deficit to add to the 
    base flow field. For more information, refer to
    :cite:`max-gunn2016limitations`.
    
    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: max-
    """

    def __init__(self):
        super().__init__()
        self.logger = setup_logger(name=__name__)
        self.model_string = "max"

    def function(self, u_field, u_wake):
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
        return np.maximum(u_wake, u_field)