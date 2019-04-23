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
    these functions return u_field with u_wake incorporated
    u_field: the modified flow field without u_wake
    u_wake: the wake to add into the rest of the flow field
    """

    def __init__(self):
        self.model_string = None

    def __str__(self):
        return self.model_string


class FLS(WakeCombination):
    """
    freestream linear superposition
    """

    def __init__(self):
        super().__init__()
        self.model_string = "fls"

    def function(self, u_field, u_wake):
        return u_field + u_wake


class SOSFS(WakeCombination):
    """
    sum of squares freestream superposition
    """

    def __init__(self):
        super().__init__()
        self.model_string = "sosfs"

    def function(self, u_field, u_wake):
        return np.hypot(u_wake, u_field)
