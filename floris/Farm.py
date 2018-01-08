"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from .BaseObject import BaseObject
from .Coordinate import Coordinate
from .WakeCombination import WakeCombination
from .FlowField import FlowField
from .TurbineMap import TurbineMap
import copy
import numpy as np

class Farm(BaseObject):
    """
        Describe farm here
    """

    def __init__(self, instance_dictionary, turbine, wake):

        super().__init__()

        self.description = instance_dictionary["description"]

        properties = instance_dictionary["properties"]
        self.wind_speed = properties["wind_speed"]
        self.wind_direction = properties["wind_direction"]
        self.turbulence_intensity = properties["turbulence_intensity"]
        self.wind_shear = properties["wind_shear"]
        self.wind_veer = properties["wind_veer"]
        self.air_density = properties["air_density"]
        self.wake_combination = properties["wake_combination"]
        self.layout_x = properties["layout_x"]
        self.layout_y = properties["layout_y"]

        # these attributes need special attention
        self.wake_combination = WakeCombination(self.wake_combination)
        self.wind_direction = np.radians(self.wind_direction - 270)

        turbine_dict = {}
        for c in list(zip(self.layout_x, self.layout_y)):
            turbine_dict[Coordinate(c[0], c[1])] = copy.deepcopy(turbine)
        self.turbine_map = TurbineMap(turbine_dict)

        self.flow_field = FlowField(wake_combination=self.wake_combination,
                                    wind_speed=self.wind_speed,
                                    wind_direction = self.wind_direction,
                                    wind_shear=self.wind_shear,
                                    wind_veer=self.wind_veer,
                                    turbulence_intensity=self.turbulence_intensity,
                                    turbine_map=self.turbine_map,
                                    wake=wake)
        self.flow_field.calculate_wake()        

    def get_turbine_coords(self):
        return self.turbine_map.coords

    def get_turbine_at_coord(self, coord):
        return self.turbine_map.turbine_at_coord(coord)
