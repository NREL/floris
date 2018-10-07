# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np

class WakeCombination():

    def __init__(self, typeString):
        self.typeString = typeString
        typeMap = {
            "fls": self._fls,
            "sosfs": self._sosfs,
        }
        self.__combinationFunction = typeMap.get(self.typeString, None)

    def combine(self, u_field, u_wake):
        return self.__combinationFunction(u_field, u_wake)

    # private functions defining the wake combinations
    # u_field: the modified flow field without u_wake
    # u_wake: the wake to add into the rest of the flow field
    #
    # the following functions return u_field with u_wake incorporated

    # freestream linear superposition
    def _fls(self, u_field, u_wake):
        return u_field + u_wake

    # sum of squares freestream superposition
    def _sosfs(self, u_field, u_wake):
        return np.hypot(u_wake, u_field)
