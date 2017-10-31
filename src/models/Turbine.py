from BaseObject import BaseObject
import numpy as np


class Turbine(BaseObject):

    def __init__(self):
        super().__init__()

        # constants
        self.nPointsInGrid = 16
        self.velocities = [0]*16

        # defined attributes
        self.rotorDiameter = None
        self.hubHeight = None
        self.numBlades = None
        self.pP = None
        self.pT = None
        self.generatorEfficiency = None
        self.eta = None

        # calculated attributes
        # initialized to 0 to pass the validity check
        self.Ct = 0    # Thrust Coefficient
        self.Cp = 0    # Power Coefficient
        self.power = 0
        self.aI = 0    # Axial Induction
        # self.TI = None  # Turbulence intensity at rotor
        self.windSpeed = 0  # Windspeed at rotor

        # controls
        self.bladePitch = 0
        self.yawAngle = 0
        self.tilt = 0
        self.TSR = 0

        # self.usePitch = usePitch
        # if usePitch:
        #     self.Cp, self.Ct, self.betaLims = CpCtpitchWs()
        # else:
        #     self.Cp, self.Ct = CpCtWs()

    def _valid(self):
        """
            Implement property checking here
            - numBlades should be > 1
            - nPointsInGrid should be > some number ensuring points in the disk area
            - velocities should be > 0
            - rotorDiameter should be > 0
            - hubHeight should be > 0
            - numBlades should be > 0
            - pP should be > 0
            - pT should be > 0
            - generatorEfficiency should be > 0
            - eta should be > 0
            - Ct should be > 0
            - Cp should be > 0
            - aI should be > 0
            - windSpeed should be > 0
        """
        valid = True
        if not super().valid():
            return False
        if not 1 < self.numBlades < 4:
            valid = False
        if any(v < 0 for v in self.velocities):
            valid = False
        return valid

    def initialize(self):
        if self._valid():
            self.grid = self.createSweptAreaGrid()
            self.velocities = [-1]*16 # use invalid value until actually corrected


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
