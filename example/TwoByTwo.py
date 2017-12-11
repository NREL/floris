import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'floris')))
from Farm import Farm
from FlowField import FlowField
from Coordinate import Coordinate
import copy


class TwoByTwo(Farm):
    """
        Describe this farm here
    """

    def __init__(self, turbine=None, wake=None, combination=None):
        super().__init__()

        turbine0 = copy.deepcopy(turbine)
        turbine1 = copy.deepcopy(turbine)
        turbine2 = copy.deepcopy(turbine)
        turbine3 = copy.deepcopy(turbine)

        turbine0.set_yaw_angle(25)
        turbine1.set_yaw_angle(0)
        turbine2.set_yaw_angle(0)
        turbine3.set_yaw_angle(0)

        self.turbineMap = {
            Coordinate(0, 0): turbine0,
            Coordinate(800, 0): turbine1,
            Coordinate(0, 800): turbine2,
            Coordinate(800, 800): turbine3
        }
        self.wake = wake
        self.combination = combination
        self.windSpeed = 7.0        # wind speed [m/s]
        self.windDirection = 270.0  # wind direction [deg] (compass degrees)

        self.flowField = FlowField(wake_combination=self.combination,
                                   wind_speed=self.windSpeed,
                                   shear=0.12,
                                   veer=0.0,
                                   turbulence_intensity=0.1,
                                   turbine_map=self.turbineMap,
                                   characteristic_height=90,
                                   wake=self.wake)
        self.flowField.calculate_wake()

    def get_turbine_coords(self):
        return self.turbineMap.keys()

    def get_turbine_at_coord(self, coord):
        return self.turbineMap[coord]
