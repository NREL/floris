from BaseObject import BaseObject
import numpy as np


class WakeDeflection(BaseObject):

    def __init__(self, typeString):
        super().__init__()
        self.typeString = typeString

        typeMap = {
            "jimenez": self._jimenez
        }
        self.function = typeMap.get(self.typeString, None)

        self.we = .05 # wake expansion

    def _jimenez():
        # this function defines the angle at which the wake deflects in relation to the yaw of the turbine
        # this is coded as defined in the Jimenez et. al. paper
        
        def calc(yaw, Ct, kd, x, D, ad, bd):
            # angle of deflection
            xi_init = 1. / 2. * np.cos(yaw) * np.sin(yaw) * Ct
            xi = (xi_init) / ((1 + 2 * kd * (x / D))**2)

            # yaw displacement
            yYaw_init = (xi_init * (15 * ((2 * kd * x / D) + 1)**4. + xi_init**2.)
                         / ((30 * kd / D) * (2 * kd * x / D + 1)**5.)) - \
                        (xi_init * D * (15 + xi_init**2.) / (30 * kd))

            # corrected yaw displacement with lateral offset
            yYaw = yYaw_init + (ad + bd * x)

            return yYaw

        return calc
