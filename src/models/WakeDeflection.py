from BaseObject import BaseObject
import numpy as np


class WakeDeflection(BaseObject):

    def __init__(self, typeString):
        super(WakeDeflection, self).__init__()
        self.typeString = typeString
        typeMap = {
            "jimenez": self.__jimenez
            # "jensen": self.__jensen,
            # "floris": self.__floris,
            # "gauss": self.gauss
        }
        self.__deflectionFunction = typeMap.get(self.typeString, None)

    def getType(self):
        return "{} - {}".format(type(self), self.typeString)

    def solve(self, Uinf, Ueff, Ufield, Uwake):
        return self.__combinationFunction(Uinf, Ueff, Ufield, Uwake)

    def __jimenez(self, yaw, Ct, kd, x, D, ad, bd):
        # this function defines the angle at which the wake deflects in
        # relation to the yaw of the turbine
        # this is coded as defined in the Jimenez et. al. paper

        # angle of deflection
        xi_init = 1. / 2. * np.cos(yaw) * np.sin(yaw) * Ct
        # xi = (xi_init) / ((1 + 2 * kd * (x / D))**2)

        # yaw displacement
        yYaw_init = (xi_init * (15 * ((2 * kd * x / D) + 1)**4. + xi_init**2.)
                     / ((30 * kd / D) * (2 * kd * x / D + 1)**5.)) - \
                    (xi_init * D * (15 + xi_init**2.) / (30 * kd))

        # corrected yaw displacement with lateral offset
        yYaw = yYaw_init + (ad + bd * x)

        return yYaw
