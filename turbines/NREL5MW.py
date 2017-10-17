import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                '..', 'src', 'models')))
import src.models.Turbine as Turbine


class NREL5MW(Turbine.Turbine):
    """
        Describe NREL 5MW here
    """

    def __init__(self, wake):
        super(NREL5MW, self).__init__()
        self.rotorDiameter = 126.0
        self.hubHeight = 90
        self.numBlades = 3
        self.pP = 1.88
        self.pT = 2.07
        self.generatorEfficiency = 1.0
        self.eta = 0.768
        self.wake = wake

    def getWake(self):
        if self.valid():
            return self.wake
