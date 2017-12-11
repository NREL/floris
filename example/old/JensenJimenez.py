import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                '..', 'src', 'models')))
from src.models.Wake import Wake
from src.models.WakeDeflection import WakeDeflection
from src.models.WakeVelocity import WakeVelocity


class JensenJimenez(Wake):
    """
        Describe this wake model here
    """
    def __init__(self):
        super().__init__()
        self.deflectionModel = WakeDeflection("jimenez")
        self.velocityModel = WakeVelocity("jensen")

    def calculate(self, downstream_distance, turbine_diameter, turbine_ct, turbine_x):
        velocity = self.get_velocity_function()
        deflection = self.get_deflection_function()
        return velocity(downstream_distance, turbine_diameter, turbine_x), deflection(downstream_distance, turbine_ct, turbine_diameter)
