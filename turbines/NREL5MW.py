import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                '..', 'src', 'models')))
from src.models.Turbine import Turbine
from scipy.interpolate import interp1d
import numpy as np


class NREL5MW(Turbine):
    """
        Describe NREL 5MW here
    """

    def __init__(self):
        super().__init__()
        self.rotorDiameter = 126.0
        self.hubHeight = 90
        self.numBlades = 3
        self.pP = 1.88
        self.pT = 2.07
        self.generatorEfficiency = 1.0
        self.eta = 0.768

        self.bladePitch = 1.9
        self.yawAngle = 0.0
        self.tilt = 0.0
        self.TSR = 8.0

        super().initialize()
        # self.initialize()

    def initialize(self):

        #TODO: improve this
        self.fCp, self.fCt = self.CpCtWs()

        self.Ct = self.calculateCt()
        self.Cp = self.calculateCp()
        self.power = self.calculatePower()
        self.aI = self.calculateAI()
        self.windSpeed = self.calculateEffectiveWindSpeed()

    def CpCtWs(self):
        CP = [0.        , 0.15643578, 0.31287155, 0.41306749, 0.44895632,
              0.46155227, 0.46330747, 0.46316077, 0.46316077, 0.46280642,
              0.45223111, 0.39353012, 0.3424487 , 0.2979978 , 0.25931677,
              0.22565665, 0.19636572, 0.17087684, 0.1486965 , 0.12939524,
              0.11259934, 0.0979836 , 0.08526502, 0.07419736, 0.06456631,
              0.05618541, 0.04889237, 0.]

        CT = [1.10610965, 1.09515807, 1.0227122 , 0.9196487 , 0.85190470,
              0.80328229, 0.76675469, 0.76209299, 0.76209299, 0.75083241,
              0.67210674, 0.52188504, 0.43178758, 0.36443258, 0.31049874,
              0.26696686, 0.22986909, 0.19961578, 0.17286245, 0.15081457,
              0.13146666, 0.11475968, 0.10129584, 0.0880188 , 0.07746819,
              0.06878621, 0.05977061, 0.]

        wind_speed = [0.        ,  2.5       ,  3.52338654,  4.57015961,
                      5.61693268,  6.66370575,  7.71047882,  8.75725189,
                      9.80402496, 10.85079803, 11.70448774, 12.25970155,
                      12.84125247, 13.45038983, 14.08842222, 14.75672029,
                      15.45671974, 16.18992434, 16.95790922, 17.76232421,
                      18.60489742, 19.48743891, 20.41184461, 21.38010041,
                      22.39428636, 23.45658122, 24.56926707, 30.]

        fCpInterp = interp1d(wind_speed, CP)
        fCtInterp = interp1d(wind_speed, CT)

        def fCp(Ws):
            print(Ws < min(wind_speed), Ws, min(wind_speed))
            return max(CP) if Ws < min(wind_speed) else fCpInterp(Ws)

        def fCt(Ws):
            print(Ws < min(wind_speed), Ws, min(wind_speed))
            return 0.99 if Ws < min(wind_speed) else fCtInterp(Ws)

        return fCp, fCt

    def calculateEffectiveWindSpeed(self):
        # TODO: why is this here?
        return self.getAverageVelocity()

    def calculateCp(self):
        # with average velocity
        print(self.getAverageVelocity())
        return self.fCp(self.getAverageVelocity())

    def calculateCt(self):
        # with average velocity
        return self.fCt(self.getAverageVelocity())

    def calculatePower(self):
        # TODO: Add yaw and pitch control
        yaw, tilt = 0, 0

        cptmp = self.Cp * (np.cos(yaw * np.pi / 180.)**self.pP) * (np.cos(tilt*np.pi/180.)**self.pT)

        #TODO: air density (1.225) is hard coded below. should this be variable in the flow field?
        return 0.5 * 1.225 * (np.pi * (self.rotorDiameter/2)**2) * cptmp * self.generatorEfficiency * self.getAverageVelocity()**3

    def calculateAI(self):
        # TODO: Add yaw and pitch control
        yaw, tilt = 0, 0

        return (0.5 / np.cos(yaw * np.pi / 180.)) * (1 - np.sqrt(1 - self.Ct * np.cos(yaw * np.pi/180)))
