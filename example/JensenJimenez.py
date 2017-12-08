import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                '..', 'floris')))
from Wake import Wake
from WakeDeflection import WakeDeflection
from WakeVelocity import WakeVelocity


class JensenJimenez(Wake):
    """
        Describe this wake model here
    """
    def __init__(self):
        super().__init__()
        # self.velocityModel = WakeVelocity("jensen")
        self.velocityModel = WakeVelocity("floris")
        self.deflectionModel = WakeDeflection("jimenez")

    def calculate(self, downstream_distance, turbine_diameter, turbine_ct, turbine_x):
        velocity = self.get_velocity_function()
        deflection = self.get_deflection_function()
        return velocity(downstream_distance, turbine_diameter, turbine_x), deflection(downstream_distance, turbine_ct, turbine_diameter)
