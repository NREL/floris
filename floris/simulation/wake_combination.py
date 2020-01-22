# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np


class WakeCombination():
    """
    The WakeCombination class provides methods for combining the base flow field with the velocity deficits from the wake models.

    Returns:
        WakeCombination: An instantiated WakeCombination object.
    """

    def __init__(self):
        self.model_string = None


class FLS(WakeCombination):
    """
    FLS is a subclass of 
    :py:class:`floris.simulation.wake_combination.WakeCombination` 
    which uses freestream linear superposition to combine the base flow 
    field with the wake velocity deficits.
    """

    def __init__(self):
        super().__init__()
        self.model_string = "fls"

    def function(self, u_field, u_wake):
        """
        This method combines the base flow field with the velocity 
        defecits using freestream linear superpostion.

        Args:
            u_field (np.array): The base flow field.
            u_wake (np.array): The wake to add to the rest of the flow 
                field.

        Returns:
            array: A linear combination of the base flow field and the 
            velocity deficits.
        """
        return u_field + u_wake


class SOSFS(WakeCombination):
    """
    SOSFS is a subclass of 
    :py:class:`floris.simulation.wake_combination.WakeCombination` 
    which uses sum of squares freestream superposition to combine the 
    base flow field with the wake velocity deficits.
    """

    def __init__(self):
        super().__init__()
        self.model_string = "sosfs"

    def function(self, u_field, u_wake):
        """
        This method combines the base flow field with the velocity 
        defecits using sum of squares.

        Args:
            u_field (np.array): The base flow field.
            u_wake (np.array): The wake to add to the rest of the flow 
                field.

        Returns:
            array: A sum of squares combination of the base flow field 
            and the velocity deficits.
        """
        return np.hypot(u_wake, u_field)
