import matplotlib.pyplot as plt
import numpy as np
from BaseObject import BaseObject


class Farm(BaseObject):
    """
        Describe farm here
    """

    def __init__(self):
        super().__init__()
        self.turbineMap = None
        self.flowField = None

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

    def get_flow_field(self):
        return self.flowField

    def plot_layout(self):
        plt.figure()
        x = [turbine[0] for turbine in self.turbineMap]
        y = [turbine[1] for turbine in self.turbineMap]
        plt.plot(x, y, 'o', color='g')
        plt.show()
