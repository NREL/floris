from BaseObject import BaseObject


class FlowField(BaseObject):
    """
        Describe FF here
    """

    def __init__(self, wakeCombination=None,
                       windSpeed=None,
                       shear=None,
                       turbineMap=None,
                       characteristicHeight=None,
                       wake=None):
        super().__init__()
        self.wakeCombination = wakeCombination
        self.windSpeed = windSpeed
        self.shear = shear
        self.turbineMap = turbineMap
        self.characteristicHeight = characteristicHeight
        self.wake = wake

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

    def initialize(self):
        if self.valid():
            self.initializeTurbineVelocities()
            self.initializeTurbines()

    def initializeTurbineVelocities(self):
        # TODO: why do we actually need to initialize?

        # initialize the flow field used in the 3D model based on shear using the power log law.
        for coord, turbine in self.turbineMap.items():
            grid = turbine.getGrid()

            # use the z coordinate of the turbine grid points for initialization
            velocities = [self.windSpeed * ((turbine.hubHeight+g[1])/self.characteristicHeight)**self.shear for g in grid]

            turbine.setVelocities(velocities)

    def initializeTurbines(self):
        for coord, turbine in self.turbineMap.items():
            turbine.initialize()
