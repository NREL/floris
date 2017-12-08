"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from BaseObject import BaseObject
import numpy as np


class WakeCombination(BaseObject):

    def __init__(self, typeString):
        super().__init__()
        self.typeString = typeString
        typeMap = {
            "fls": self._fls,
            "lvls": self._lvls,
            "sosfs": self._sosfs,
            "soslvs": self._soslvs
        }
        self.__combinationFunction = typeMap.get(self.typeString, None)

    def combine(self, Uinf, Ueff, Ufield, Uwake):
        return self.__combinationFunction(Uinf, Ueff, Ufield, Uwake)

    # private functions defining the wake combinations

    # freestream linear superposition
    def _fls(self, u_inf, u_eff, u_field, u_wake):
        return u_field - u_wake

    # local velocity linear superposition
    def _lvls(Uinf, Ueff, Ufield, Uwake):
        return Uinf - ((Ueff - Uwake) + (Uinf - Ufield))

    # sum of squares freestream superposition
    def _sosfs(Uinf, Ueff, Ufield, Uwake):
        return Uinf - np.sqrt((Uinf - Uwake)**2 + (Uinf - Ufield)**2)

    # sum of squares local velocity superposition
    def _soslvs(Uinf, Ueff, Ufield, Uwake):
        return Ueff - np.sqrt((Ueff - Uwake)**2 + (Uinf - Ufield)**2)
