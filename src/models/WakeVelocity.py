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
    
    def _tanh_filter(self, x, loc):
        sharpness = 10
        return (1 + np.tanh(sharpness * (x - loc))) / 2.

    def _jensen(self, streamwise_location, horizontal_location, vertical_location, turbine_diameter, turbine_x):
        # compute the velocity deficit based on the classic Jensen/Park model. see Jensen 1983
        # +/- 2keX is the slope of the cone boundary for the wake
        turbine_radius = turbine_diameter / 2.
        upper_bound = turbine_radius + 2 * self.we * streamwise_location
        lower_bound = -1 * upper_bound

        if streamwise_location - turbine_x < 0:
            return 0
        elif horizontal_location > upper_bound or horizontal_location < lower_bound:
            return 0
        else:
            return (turbine_radius / (self.we * (streamwise_location - turbine_x) + turbine_radius))**2
