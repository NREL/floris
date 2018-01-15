# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from .coordinate import Coordinate
from .wake_combination import WakeCombination
from .flow_field import FlowField
from .turbine_map import TurbineMap
import copy
import numpy as np

class Farm():
    """
    Farm is the container class of the FLORIS package. It brings together all
    of the component objects after input (ie Turbine, Wake, FlowField) and
    packages everything into the appropriate data type. Farm should also be used
    as an entry point to probe objects for generating output.

    inputs:
        instance_dictionary: dict - the input dictionary as generated from the input_reader;
            it should have the following key-value pairs:
                {
                    "wind_speed": float,

                    "wind_direction": float,

                    "turbulence_intensity": float,

                    "wind_shear": float,

                    "wind_veer": float,

                    "air_density": float,

                    "wake_combination": str,

                    "layout_x": list,

                    "layout_y": list
        
                }
     
        turbine: Turbine - Turbine instance used in Farm

        wake: Wake - Wake instance used in Farm

    outputs:
        self: Farm - an instantiated Farm object
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
