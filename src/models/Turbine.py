from BaseObject import BaseObject


class Turbine(BaseObject):

    def __init__(self):
        super(Turbine, self).__init__()

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

        self.wake = None

    def printdeets(self):
        print(self.wake.combination.getType())
