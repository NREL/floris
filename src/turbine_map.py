# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from .coordinate import Coordinate
from .turbine import Turbine
import numpy as np


class TurbineMap():
    """
    TurbineMap is container object which maps a Turbine instance to a Coordinate
    object. This class also provides some helper methods for sorting and 
    manipulating the turbine layout.

    inputs:
        turbine_map_dict: dict - a dictionary mapping of Turbines to Coordinates
            it should have the following form:
                {
                    Coordinate(): Turbine(),

                    Coordinate(): Turbine(),

                    ...,

                    Coordinate(): Turbine(),

                }

    outputs:
        self: TurbineMap - an instantiated TurbineMap object
    """

    def __init__(self, turbine_map_dict):
        self.turbine_map_dict = turbine_map_dict
        self.coords = [coord for coord, _ in self.items()]
        self.turbines = [turbine for _, turbine in self.items()]

    def items(self):
        return self.turbine_map_dict.items()

    def rotated(self, angle, center_of_rotation):
        """
        Rotated the turbine coordinates by a given angle about a given center
        of rotation. This function returns a new TurbineMap object whose turbines
        are rotated. The original TurbineMap is not modified.
        """
        rotated = {}
        for coord, turbine in self.items():
            coord_list = coord.rotate_z(angle, center_of_rotation.as_tuple())
            rotated_coordinate = Coordinate(coord_list[0], coord_list[1])
            rotated[rotated_coordinate] = turbine
        return TurbineMap(rotated)

    def sorted_in_x_as_list(self):
        coords = sorted(self.turbine_map_dict, key=lambda coord: coord.x)
        return [(c, self.turbine_map_dict[c]) for c in coords]
