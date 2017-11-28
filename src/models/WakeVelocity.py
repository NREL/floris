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

        # wake expansion coefficient
        self.we = .05
    
    def _activation_function(self, x, loc):
        sharpness = 10
        return (1 + np.tanh(sharpness * (x - loc))) / 2.

    def _jensen(self, x_locations, y_locations, z_locations, turbine_radius, turbine_coord):
        """
            x direction is streamwise (with the wind)
            y direction is normal to the streamwise direction and parallel to the ground
            z direction is normal the streamwise direction and normal to the ground=
        """
        # compute the velocity deficit based on the classic Jensen/Park model. see Jensen 1983
        # +/- 2keX is the slope of the cone boundary for the wake

        # y = mx + b
        m = 2 * self.we
        x = x_locations - turbine_coord.x
        b = turbine_radius
        boundary_line = m * x + b
        y_upper = boundary_line + turbine_coord.y
        y_lower = -1 * boundary_line + turbine_coord.y

        # calculate the wake velocity
        c = (turbine_radius / (self.we * (x_locations - turbine_coord.x) + turbine_radius))**2

        # filter points upstream and beyond the upper and lower bounds of the wake
        c[x_locations - turbine_coord.x < 0] = 0
        c[y_locations > y_upper] = 0
        c[y_locations < y_lower] = 0

        return c
