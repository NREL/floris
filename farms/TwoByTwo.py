import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                '..', 'src', 'models')))
from src.models.Farm import Farm


class TwoByTwo(Farm):
    """
        Describe this farm here
    """

    def __init__(self, turbine=None, wake=None, combination=None):
        super().__init__()
        self.turbineMap = {
            (0,      0): copy.deepcopy(turbine)
            # (800,    0): copy.deepcopy(turbine),
            # (400,  800): copy.deepcopy(turbine),
            # (1200, 800): copy.deepcopy(turbine)
        }
        self.wake = wake
        self.combination = combination
        self.windSpeed = 7.0       # wind speed [m/s]
        self.windDirection = 90.0  # wind direction [deg] (compass degrees)
