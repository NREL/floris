# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np

class Coordinate():
    """
        The Coordinate class is a container for coordinates. It provides a
        convenient and consistent wrapper in order to avoid referencing components
        of a coordinate by index of list; for example, Coordinate provides
        access to the x component with coordinate.x whereas typical list or tuple
        containers require referencing the x component with coordinate[0].
    """

    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.xprime = self.x
        self.yprime = self.y
        self.zprime = self.z

    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    def as_tuple(self):
        """
        Return the coordinate object as a tuple
        """
        return (self.x, self.y)

    def rotate_z(self, theta, center_of_rotation=(0, 0, 0)):
        """
        Rotate about the z coordinate axis by a given angle and center of rotation.
        The angle theta should be given in radians.
        """
        xoffset = self.x - center_of_rotation[0]
        yoffset = self.y - center_of_rotation[1]
        self.xprime = xoffset * np.cos(theta) - yoffset * np.sin(theta) + center_of_rotation[0]
        self.yprime = yoffset * np.cos(theta) + xoffset * np.sin(theta) + center_of_rotation[1]
        return self.xprime, self.yprime
