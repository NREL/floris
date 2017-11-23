import numpy as np
from BaseObject import BaseObject


class WakeVelocity(BaseObject):

    def __init__(self, typeString):
        super().__init__()
        self.typeString = typeString

        typeMap = {
            "jensen": self._jensen
        }
        self.function = typeMap[typeString]

        self.we = .05 # wake expansion
    
    def _activation_function(self, x, loc):
        sharpness = 10
        return (1 + np.tanh(sharpness * (x - loc))) / 2.

    def _jensen(self, streamwise_location, horizontal_location, vertical_location, turbine_diameter, turbine_x):
        # compute the velocity deficit based on the classic Jensen/Park model. see Jensen 1983
        # +/- 2keX is the slope of the cone boundary for the wake
        turbine_radius = turbine_diameter / 2.
        upper_bound = turbine_radius + 2 * self.we * streamwise_location
        lower_bound = -1 * upper_bound

        c = (turbine_radius / (self.we * (streamwise_location - turbine_x) + turbine_radius))**2

        c[streamwise_location - turbine_x < 0] = 0  # all points upstream of the turbine
        c[horizontal_location > upper_bound] = 0    # all points beyond the upper bound
        c[horizontal_location < lower_bound] = 0    # all points below the lower bound

        return c