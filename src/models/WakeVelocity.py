from BaseObject import BaseObject


class WakeVelocity(BaseObject):

    def __init__(self, typeString):
        super().__init__()
        self.typeString = typeString

        typeMap = {
            "jensen": self._jensen
        }
        self.function = typeMap[typeString]

        self.we = .05 # wake expansion

    def _jensen():
        # compute the velocity deficit based on the classic Jensen/Park model. see Jensen 1983
        
        def calc(D, X, xTurb):
            # D: turbine diameter
            # X: downstream location
            # xTurb: turbine location
            # ke: wake expansion
            return (D / (2 * ke * (X - xTurb) + D))**2

        return calc
