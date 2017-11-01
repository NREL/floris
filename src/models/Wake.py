from BaseObject import BaseObject


class Wake(BaseObject):

    def __init__(self):
        super().__init__()
        self.deflectionModel = None  # type: WakeDeflection
        self.velocityModel = None    # type: WakeVelocity

    def valid(self):
        """
            Implement property checking here
        """
        valid = True
        if not super().valid():
            valid = False
        return valid

    def initialize(self):
        if self.valid():
            print("cool")

    def _get_deflection_function(self):
        return self.deflectionModel.function

    def _get_velocity_function(self):
        return self.velocityModel.function
    
