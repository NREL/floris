from BaseObject import BaseObject
import numpy as np
from scipy.interpolate import interp1d


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

        self.grid = self._create_swept_area_grid()
        # use invalid value until actually corrected
        self.velocities = [-1] * 16

    def initialize(self):
        """
            Creates the grid on the disk and initializes the descrete velocities
        """
        #TODO: improve this
        self.fCp, self.fCt = self.CpCtWs()

        self.Ct = self.calculate_ct()
        self.Cp = self.calculate_cp()
        self.power = self.calculate_power()
        self.aI = self.calculate_ai()
        self.windSpeed = self.calculate_effective_wind_speed()


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

    def calculate_effective_wind_speed(self):
        # TODO: why is this here?
        return self.get_average_velocity()

    def calculate_cp(self):
        # with average velocity
        return self.fCp(self.get_average_velocity())

    def calculate_ct(self):
        # with average velocity
        print("average velocity: ", self.get_average_velocity())
        return self.fCt(self.get_average_velocity())

    def calculate_power(self):
        # TODO: Add yaw and pitch control
        yaw, tilt = 0, 0

        cptmp = self.Cp * (np.cos(yaw * np.pi / 180.)**self.pP) * (np.cos(tilt*np.pi/180.)**self.pT)

        #TODO: air density (1.225) is hard coded below. should this be variable in the flow field?
        return 0.5 * 1.225 * (np.pi * (self.rotorDiameter/2)**2) * cptmp * self.generatorEfficiency * self.get_average_velocity()**3

    def calculate_ai(self):
        # TODO: Add yaw and pitch control
        yaw, tilt = 0, 0
        print(1 - self.Ct * np.cos(yaw * np.pi / 180))
        return (0.5 / np.cos(yaw * np.pi / 180.)) * (1 - np.sqrt(1 - self.Ct * np.cos(yaw * np.pi/180)))

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
            return max(CP) if Ws < min(wind_speed) else fCpInterp(Ws)

        def fCt(Ws):
            return 0.99 if Ws < min(wind_speed) else fCtInterp(Ws)

        return fCp, fCt
        
    # def _show_constant_z_plot(self):

    # Public methods

    def get_grid(self):
        return self.grid

    def set_velocities(self, velocities):
        # TODO: safety check before or after setting velocities
        # verifying correct size of array and correct values
        self.velocities = velocities

    def get_average_velocity(self):
        return np.mean(self.velocities)

    def show_wake_plot(self):
        self._show_constant_z_plot(self)
