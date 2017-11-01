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

        # to be specified in user input
        self.kd = .17 # wake deflection
        self.ad = -4.5
        self.bd = -0.01

    def _jimenez(self, downstream_distance, turbine_ct, turbine_diameter):
        # this function defines the angle at which the wake deflects in relation to the yaw of the turbine
        # this is coded as defined in the Jimenez et. al. paper

        # TODO: add yaw
        yaw = 0

        # angle of deflection
        xi_init = (1./2.) * np.cos(yaw) * np.sin(yaw) * turbine_ct

        # yaw displacement
        yYaw_init = (xi_init * (15 * ((2 * self.kd * downstream_distance / turbine_diameter) + 1)**4. + xi_init**2.)
                     / ((30 * self.kd / turbine_diameter) * (2 * self.kd * downstream_distance / turbine_diameter + 1)**5.)) - \
                    (xi_init * turbine_diameter * (15 + xi_init**2.) / (30 * self.kd))

        # corrected yaw displacement with lateral offset
        yYaw = yYaw_init + (self.ad + self.bd * downstream_distance)

        return yYaw
