from BaseObject import BaseObject


class WakeVelocity(BaseObject):

    def __init__(self, typeString):
        super().__init__()
        self.typeString = typeString
        typeMap = {
            "jensen": self.__jensen
        }
        self.__deflectionFunction = typeMap.get(self.typeString, None)

    def getType(self):
        return "{} - {}".format(type(self), self.typeString)

    def __jensen(D, ke, X, xTurb):
        # compute the velocity deficit based on the classic Jensen/Park model
        # see Jensen 1983
        return (D / (2 * ke * (X - xTurb) + D))**2
