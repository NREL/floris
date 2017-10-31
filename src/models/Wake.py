from BaseObject import BaseObject


class Wake(BaseObject):

    def __init__(self):
        super().__init__()
        self.deflectionModel = None
        self.velocityModel = None

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

    def getDeflectionFunction(self):
        return self.deflection

    def getVelocityFunction(self):
        return self.velocity
