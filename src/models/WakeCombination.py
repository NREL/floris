from BaseObject import BaseObject
import numpy as np


class WakeCombination(BaseObject):

    def __init__(self, typeString):
        super(WakeCombination, self).__init__()
        self.typeString = typeString
        typeMap = {
            "fls": self.__FLS,
            "lvls": self.__LVLS,
            "sosfs": self.__SOSFS,
            "soslvs": self.__SOSLVS
        }
        self.__combinationFunction = typeMap.get(self.typeString, None)

    def getType(self):
        return "{} - {}".format(type(self), self.typeString)

    def solve(self, Uinf, Ueff, Ufield, Uwake):
        return self.__combinationFunction(Uinf, Ueff, Ufield, Uwake)

    # private functions defining the wake combinations

    # freestream linear superposition
    def __FLS(Uinf, Ueff, Ufield, Uwake):
        return Uinf - ((Uinf - Uwake) + (Uinf - Ufield))

    # local velocity linear superposition
    def __LVLS(Uinf, Ueff, Ufield, Uwake):
        return Uinf - ((Ueff - Uwake) + (Uinf - Ufield))

    # sum of squares freestream superposition
    def __SOSFS(Uinf, Ueff, Ufield, Uwake):
        return Uinf - np.sqrt((Uinf - Uwake)**2 + (Uinf - Ufield)**2)

    # sum of squares local velocity superposition
    def __SOSLVS(Uinf, Ueff, Ufield, Uwake):
        return Ueff - np.sqrt((Ueff - Uwake)**2 + (Uinf - Ufield)**2)
