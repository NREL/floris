# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from ..utilities import Vec3
from .turbine import Turbine
import numpy as np
import copy


class TurbineMap():
    """
    TurbineMap is container object which maps a Turbine instance to a Vec3
    object. This class also provides some helper methods for sorting and 
    manipulating the turbine layout.

    inputs:
        turbine_map_dict: dict - a dictionary mapping of Turbines to Vec3
            it should have the following form:
                {
                    Vec3(): Turbine(),

                    Vec3(): Turbine(),

                    ...,

                    Vec3(): Turbine(),

                }

    outputs:
        self: TurbineMap - an instantiated TurbineMap object
    """

    def __init__(self, layout_x, layout_y, turbine):
        turbine_dict = {}
        for c in list(zip(layout_x, layout_y)):
            turbine_dict[Vec3(c[0], c[1], turbine.hub_height)] = copy.deepcopy(turbine)
        self._turbine_map_dict = turbine_dict


    def rotated(self, angle, center_of_rotation):
        """
        Rotated the turbine coordinates by a given angle about a given center
        of rotation. This function returns a new TurbineMap object whose turbines
        are rotated. The original TurbineMap is not modified.
        """
        layout_x = np.zeros(len(self.coords))
        layout_y = np.zeros(len(self.coords))
        for i, coord in enumerate(self.coords):
            coord.rotate_on_x3(angle, center_of_rotation)
            layout_x[i] = coord.x1prime
            layout_y[i] = coord.x2prime
        return TurbineMap(layout_x, layout_y, self.turbines[0]) # TODO ASSUMES single turbine type

    def sorted_in_x_as_list(self):
        coords = sorted(self._turbine_map_dict, key=lambda coord: coord.x1)
        return [(c, self._turbine_map_dict[c]) for c in coords]

    @property
    def turbines(self):
        return [turbine for _, turbine in self.items]

    @property
    def coords(self):
        return [coord for coord, _ in self.items]

    @property
    def items(self):
        return self._turbine_map_dict.items()
