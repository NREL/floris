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
    
