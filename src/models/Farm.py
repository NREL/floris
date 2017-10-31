import matplotlib.pyplot as plt
from BaseObject import BaseObject


class Farm(BaseObject):
    """
        Describe farm here
    """

    def __init__(self):
        super().__init__()
        self.turbineMap = None
        self.flowfield = None

    def valid(self):
        """
            Do validity check
        """
        valid = True
        if not super().valid():
            valid = False
        return valid

    def initialize(self):
        if self.valid():
            self.flowfield.setTurbineMap(turbineMap)

    def plotFarm(self):
        plt.figure()
        x = [turbine[0] for turbine in self.turbineMap]
        y = [turbine[1] for turbine in self.turbineMap]
        plt.plot(x, y, 'o', color='g')
        plt.show()
