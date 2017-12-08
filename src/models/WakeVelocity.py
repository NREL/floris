"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import numpy as np
from BaseObject import BaseObject


class WakeVelocity(BaseObject):

    def __init__(self, typeString):
        super().__init__()
        self.typeString = typeString

        typeMap = {
            "jensen": self._jensen,
            "floris": self._floris
        }
        self.function = typeMap[typeString]

        # wake expansion coefficient
        self.we = .05

        # floris parameters

    
    def _activation_function(self, x, loc):
        sharpness = 10
        return (1 + np.tanh(sharpness * (x - loc))) / 2.

    def _jensen(self, x_locations, y_locations, z_locations, turbine, turbine_coord):
        """
            x direction is streamwise (with the wind)
            y direction is normal to the streamwise direction and parallel to the ground
            z direction is normal the streamwise direction and normal to the ground=
        """
        # compute the velocity deficit based on the classic Jensen/Park model. see Jensen 1983
        # +/- 2keX is the slope of the cone boundary for the wake

        # define the boundary of the wake model ... y = mx + b
        m = 2 * self.we
        x = x_locations - turbine_coord.x
        b = turbine.rotorRadius
        boundary_line = m * x + b
        y_upper = boundary_line + turbine_coord.y
        y_lower = -1 * boundary_line + turbine_coord.y

        # calculate the wake velocity
        c = (turbine.rotorRadius / 
                (self.we * (x_locations - turbine_coord.x) + turbine.rotorRadius))**2

        # filter points upstream and beyond the upper and lower bounds of the wake
        c[x_locations - turbine_coord.x < 0] = 0
        c[y_locations > y_upper] = 0
        c[y_locations < y_lower] = 0

        return 2 * turbine.aI * c

    def _floris(self, x_locations, y_locations, z_locations, turbine, turbine_coord):# yDisp, zDisp, xTurb, yTurb, zTurb, inputData, turbI):
        # compute the velocity deficit based on wake zones, see Gebraad et. al. 2016

        # wake parameters
        me = np.array([-0.5, 0.3, 1.0]) # inputData['me']
        aU = np.radians(12.0) #inputData['aU']
        bU = np.radians(1.3) #inputData['bU']
        radius = turbine.rotorRadius
        yaw = turbine.yawAngle
        mu = [0.5, 1., 5.5] / np.cos(aU + bU * yaw)

        # distance from wake centerline
        yDisp = 0
        rY = abs(y_locations - (turbine_coord.y + yDisp))
        dx = x_locations - turbine_coord.x

        # wake zone diameters
        d = np.ndarray((3,) + x_locations.shape)
        d[0] = (radius + self.we * me[0] * dx)
        d[1] = (radius + self.we * me[1] * dx)
        d[2] = (radius + self.we * me[2] * dx)

        # initialize the wake field
        c = np.zeros(x_locations.shape)

        # near wake zone
        mask = rY <= d[0]
        c += mask * (radius / (radius + self.we * mu[0] * dx))**2

        # far wake zone
        # ^ is XOR, x^y:
        #   Each bit of the output is the same as the corresponding bit in x
        #   if that bit in y is 0, and it's the complement of the bit in x
        #   if that bit in y is 1.
        # The resulting mask is all the points in far wake zone that are not
        # in the near wake zone
        mask = (rY <= d[1]) ^ (rY <= d[0])
        c += mask * (radius / (radius + self.we * mu[1] * dx))**2

        # mixing zone
        # | is OR, x|y:
        #   Each bit of the output is 0 if the corresponding bit of x AND
        #   of y is 0, otherwise it's 1.
        # The resulting mask is all the points in mixing zone that are not
        # in the far wake zone and not in  near wake zone
        mask = (rY <= d[2]) ^ ((rY <= d[1]) | (rY <= d[0]))
        c += mask * (radius / (radius + self.we * mu[2] * dx))**2

        # filter points upstream
        c[x_locations - turbine_coord.x < 0] = 0

        return 2 * turbine.aI * c

    def print_countvals(self, array): 
        unique, counts = np.unique(array, return_counts=True)
        print(dict(zip(unique, counts)))
    
    # def _gauss(self):

    # def _gauss_thrust_angle(self):
    
