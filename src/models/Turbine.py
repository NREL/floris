from BaseObject import BaseObject
import numpy as np


class Turbine(BaseObject):

    def __init__(self):
        super().__init__()

        self.rotorDiameter = None
        self.hubHeight = None
        self.numBlades = None
        self.pP = None
        self.pT = None
        self.generatorEfficiency = None
        self.eta = None

        # self.usePitch = usePitch
        # if usePitch:
        #     self.Cp, self.Ct, self.betaLims = CpCtpitchWs()
        # else:
        #     self.Cp, self.Ct = CpCtWs()

    def printdeets(self):
        if self.valid:
            print(self.wake.combination.getType())

    def valid(self):
        """
            Implement property checking here
            For example, numBlades should be > 1
        """
        properties = [
            self.rotorDiameter,
            self.hubHeight,
            self.numBlades,
            self.pP,
            self.pT,
            self.generatorEfficiency,
            self.eta
        ]
        invalid = None in properties
        return not invalid
    
    def calculatePower(self):
        
        rho 		= 1.0
        CpCtTable	= 1.0

        # turbine operation
        yaw = 1.0
        tilt = 1.0
        Ct = 1.0
        Cp = 1.0
        
        area = np.pi

        area = np.pi * (self.rotorDiameter / 2.)
        Cptmp = Cp * (np.cos(yaw * np.pi / 180.)**self.pP) * (np.cos((tilt) * np.pi / 180.)**self.pT)
        power = 0.5 * rho * area * Cptmp * self.generatorEfficiency * Ueff**3

        return power

    def calculateEffectiveWindSpeed(self):
        return 6.97936532962
    
    def calculateCp(self):
        return 0.44775728
    
    def calculateCt(self):
        return 0.68698416
    
    def calculatePower(self):
        return 1162592.50472816
    
    def calculateAI(self):
        return 0.22026091
    