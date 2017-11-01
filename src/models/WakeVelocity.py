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

    def _jensen(self, downstream_distance, turbine_diameter, turbine_x):
        # compute the velocity deficit based on the classic Jensen/Park model. see Jensen 1983
        # +/- 2keX is the slope of the cone boundary for the wake
        return (turbine_diameter / (2 * self.we * (downstream_distance - turbine_x) + turbine_diameter))**2
