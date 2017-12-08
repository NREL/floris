"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import os
import sys
from BaseObject import BaseObject
import numpy as np


class Coordinate(BaseObject):
    """
        The Coordinate class is a model for 
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.xprime = self.x
        self.yprime = self.y

    def __str__(self):
        return str(self.x) + ", " + str(self.y)

    def rotate(self, theta, center_of_rotation=(0, 0)):
        xoffset = self.x - center_of_rotation[0]
        yoffset = self.y - center_of_rotation[1]
        self.xprime = xoffset * np.cos(theta) - yoffset * np.sin(theta) + center_of_rotation[0]
        self.yprime = yoffset * np.cos(theta) + xoffset * np.sin(theta) + center_of_rotation[1]
        return self.xprime, self.yprime
