from BaseObject import BaseObject
import numpy as np


class Turbine(BaseObject):

    def __init__(self):
        super().__init__()

        # constants
        self.nPointsInGrid = 16
        self.velocities = [0]*16
        self.grid = [0]*16

        # defined attributes
        self.description = None
        self.rotorDiameter = None
        self.hubHeight = None
        self.numBlades = None
        self.pP = None
        self.pT = None
        self.generatorEfficiency = None
        self.eta = None

        # calculated attributes
        # initialized to 0 to pass the validity check
        self.Ct = 0         # Thrust Coefficient
        self.Cp = 0         # Power Coefficient
        self.power = 0      # Power (W) <-- True?
        self.aI = 0         # Axial Induction
        self.TI = 0         # Turbulence intensity at rotor
        self.windSpeed = 0  # Windspeed at rotor

        # controls
        self.bladePitch = 0 # radians
        self.yawAngle = 0   # radians
        self.tilt = 0       # radians
        self.TSR = 0

        # self.usePitch = usePitch
        # if usePitch:
        #     self.Cp, self.Ct, self.betaLims = CpCtpitchWs()
        # else:
        #     self.Cp, self.Ct = CpCtWs()


    def valid(self):
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

    def init_with_dict(self, dictionary):
        self.description = dictionary["description"]

        properties = dictionary["properties"]
        self.rotorDiameter = properties["rotorDiameter"]
        self.hubHeight = properties["hubHeight"]
        self.numBlades = properties["numBlades"]
        self.pP = properties["pP"]
        self.pT = properties["pT"]
        self.generatorEfficiency = properties["generatorEfficiency"]
        self.eta = properties["eta"]
        self.bladePitch = properties["bladePitch"]
        self.yawAngle = properties["yawAngle"]
        self.tilt = properties["tilt"]
        self.TSR = properties["TSR"]

    def initialize_velocities(self):
        if self.valid():
            self.grid = self._create_swept_area_grid()
            self.velocities = [-1]*16  # use invalid value until actually corrected

    # Private methods

    def _create_swept_area_grid(self):
        # TODO: add validity check:
        # rotor points has a minimum in order to always include points inside
        # the disk ... 2?
        #
        # the grid consists of the y,z coordinates of the discrete points which
        # lie within the rotor area: [(y1,z1), (y2,z2), ... , (yN, zN)]

        # update:
        # using all the grid point because that how roald did it.
        # are the points outside of the rotor disk used later?

        # determine the dimensions of the square grid
        num_points = int(np.round(np.sqrt(self.nPointsInGrid)))
        # syntax: np.linspace(min, max, n points)
        horizontal = np.linspace(-self.rotorDiameter/2, self.rotorDiameter/2, num_points)
        vertical = np.linspace(-self.rotorDiameter/2, self.rotorDiameter/2, num_points)

        # build the grid with all of the points
        grid = [(h, vertical[i]) for i in range(num_points) for h in horizontal]

        # keep only the points in the swept area
        # grid = [point for point in grid if np.hypot(point[0], point[1]) < self.rotorDiameter/2]

        return grid

    # Public methods

    def get_grid(self):
        return self.grid

    def set_velocities(self, velocities):
        # TODO: safety check before or after setting velocities
        # verifying correct size of array and correct values
        self.velocities = velocities

    def get_average_velocity(self):
        return np.mean(self.velocities)
