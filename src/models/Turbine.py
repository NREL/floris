from BaseObject import BaseObject
import numpy as np


class Turbine(BaseObject):

    def __init__(self):
        super().__init__()

        # defined attributes
        self.rotorDiameter = None
        self.hubHeight = None
        self.numBlades = None
        self.pP = None
        self.pT = None
        self.generatorEfficiency = None
        self.eta = None

        # calculated attributes
        self.Ct = None  # Thrust Coefficient
        self.Cp = None  # Power Coefficient
        self.aI = None  # Axial Induction
        self.TI = None  # Turbulence intensity at rotor
        self.windSpeed = None  # Windspeed at rotor
        
        # self.usePitch = usePitch
        # if usePitch:
        #     self.Cp, self.Ct, self.betaLims = CpCtpitchWs()
        # else:
        #     self.Cp, self.Ct = CpCtWs()

    def valid(self):
        """
            Implement property checking here
            For example, numBlades should be > 1
        """
        valid = True
        if not super().valid():
            valid = False
        if not 1 < self.numBlades < 4:
            valid = False
        return valid

    def calculatePower(self):
        rho = 1.0
        CpCtTable = 1.0

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
    
    def createSweptAreaGrid(self):
        # TODO: add validity check: 
        # rotor points has a minimum in order to always include points inside
        # the disk ... 2?
        rotorPts = int(np.round(np.sqrt(self.nPointsInGrid)))
        # min, max, n points
        horizontal = np.linspace(-self.rotorDiameter/2, self.rotorDiameter/2, rotorPts)
        vertical = np.linspace(-self.rotorDiameter/2, self.rotorDiameter/2, rotorPts)
        grid = []
        for i in range(rotorPts):
            if np.hypot(horizontal[i], vertical[i]) < self.rotorDiameter/2:
                grid.append((horizontal[i], vertical[i]))
        return grid
    
    def calculateAverageVelocity(self):
        return np.mean(np.array(Utmp)[cond2])
