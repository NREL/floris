import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                '..', 'src', 'models')))
from src.models.Farm import Farm
from src.models.FlowField import FlowField
import copy


class TwoByTwo(Farm):
    """
        Describe this farm here
    """

    def __init__(self, turbine=None, wake=None, combination=None):
        super().__init__()
        self.turbineMap = {
            (0,      0): copy.deepcopy(turbine),
            (600,    0): copy.deepcopy(turbine)
            # (400,  800): copy.deepcopy(turbine),
            # (1200, 800): copy.deepcopy(turbine)
        }
        self.wake = wake
        self.combination = combination
        self.windSpeed = 7.0        # wind speed [m/s]
        self.windDirection = 270.0  # wind direction [deg] (compass degrees)

        self.flowField = FlowField(wake_combination=self.combination,
                                   wind_speed=self.windSpeed,
                                   shear=0.12,
                                   turbine_map=self.turbineMap,
                                   characteristic_height=90,
                                   wake=self.wake)
        self.flowField.calculate_wake()

    def get_turbine_coords(self):
        return self.turbineMap.keys()

    def getTurbineAtCoord(self, coord):
        return self.turbineMap[coord]
