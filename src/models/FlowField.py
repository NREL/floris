from BaseObject import BaseObject


class FlowField(BaseObject):
    """
        Describe FF here
    """

    def __init__(self, wakeCombination=None,
                       windSpeed=None,
                       shear=None,
                       turbineCoords=None,
                       characteristicHeight=None):
        super().__init__()
        self.wakeCombination = wakeCombination
        self.windSpeed = windSpeed
        self.shear = shear
        self.turbineCoords = turbineCoords             # {(x,y,z), (x,y,z), ...}
        self.characteristicHeight = characteristicHeight
        if self.valid():
            self.UfieldOrig = self.initializeVelocities() # initial velocities at each turbine
        
    def valid(self):
        """
            Do validity check
        """
        valid = True
        if not super().valid():
            return False
        if self.characteristicHeight <= 0:
            valid = False
        return valid

    def initializeVelocities(self):
        # initialize the flow field used in the 3D model based on shear using 
        # the power log law.
        vels = [self.windSpeed * (c[2]/self.characteristicHeight)**self.shear for c in self.turbineCoords]
        return set(vels)
