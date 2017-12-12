import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from floris.Farm import Farm
from floris.FlowField import FlowField
from floris.Coordinate import Coordinate
import copy


class TwoByTwo(Farm):
    """
        Describe this farm here
    """

    def __init__(self, turbine, wake, combination):
        
        self.turbineMap = {
            Coordinate(0, 0): copy.deepcopy(turbine),
            Coordinate(800, 0): copy.deepcopy(turbine),
            Coordinate(0, 600): copy.deepcopy(turbine),
            Coordinate(800, 600): copy.deepcopy(turbine)
        }
        self.wake = wake
        self.combination = combination
        self.windSpeed = 7.0        # wind speed [m/s]
        self.windDirection = 300.0  # wind direction [deg] (compass degrees)

        self.flowField = FlowField(wake_combination=self.combination,
                                   wind_speed=self.windSpeed,
                                   wind_direction=self.windDirection,
                                   shear=0.12,
                                   veer=0.0,
                                   turbulence_intensity=0.1,
                                   turbine_map=self.turbineMap,
                                   wake=self.wake)
        super().__init__(self.turbineMap, self.flowField)

    def get_turbine_coords(self):
        return self.turbineMap.keys()

    def get_turbine_at_coord(self, coord):
        return self.turbineMap[coord]
