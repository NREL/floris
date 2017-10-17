from BaseObject import BaseObject


class Turbine(BaseObject):

    def __init__(self):
        super(Turbine, self).__init__()
        self.rotorDiameter = 126.0
        self.hubHeight = 90
        self.numBlades = 3
        self.pP = 1.88
        self.pT = 2.07
        self.generatorEfficiency = 1.0
        self.eta = 0.768

        # self.usePitch = usePitch
        # if usePitch:
        #     self.Cp, self.Ct, self.betaLims = CpCtpitchWs()
        # else:
        #     self.Cp, self.Ct = CpCtWs()

        self.wake = None

    def printdeets(self):
        print(self.wake.combination.getType())
