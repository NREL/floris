"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

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
        self.rotorRadius = self.rotorDiameter / 2.
        self.hubHeight = properties["hubHeight"]
        self.numBlades = properties["numBlades"]
        self.pP = properties["pP"]
        self.pT = properties["pT"]
        self.generatorEfficiency = properties["generatorEfficiency"]
        self.eta = properties["eta"]
        self.power_thrust_table = properties["power_thrust_table"]
        self.bladePitch = properties["bladePitch"]
        self.yawAngle = np.radians(properties["yawAngle"])
        self.tiltAngle = np.radians(properties["tilt"])
        self.TSR = properties["TSR"]

        self.fCp, self.fCt = self._CpCtWs()
        self.grid = self._create_swept_area_grid()
        self.velocities = [-1] * 16  # initialize to an invalid value until calculated

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

    def _calculate_cp(self):
        # with average velocity
        return self.fCp(self.get_average_velocity())

    def _calculate_ct(self):
        # with average velocity        
        return self.fCt(self.get_average_velocity())

    def _calculate_power(self):
        cptmp = self.Cp \
                * np.cos(self.yawAngle * np.pi / 180.)**self.pP \
                * np.cos(self.tiltAngle * np.pi / 180.)**self.pT

        #TODO: air density (1.225) is hard coded below. should this be variable in the flow field?
        return 0.5 * 1.225 * (np.pi * (self.rotorDiameter/2)**2) * cptmp * self.generatorEfficiency * self.get_average_velocity()**3

    def _calculate_ai(self):
        return 0.5 / np.cos(self.yawAngle * np.pi / 180.) \
               * (1 - np.sqrt(1 - self.Ct * np.cos(self.yawAngle * np.pi / 180) ) )

    def _CpCtWs(self):
        cp = self.power_thrust_table["power"]
        ct = self.power_thrust_table["thrust"]
        windspeed = self.power_thrust_table["wind_speed"]

        fCpInterp = interp1d(windspeed, cp)
        fCtInterp = interp1d(windspeed, ct)

        def fCp(Ws):
            return max(cp) if Ws < min(windspeed) else fCpInterp(Ws)

        def fCt(Ws):
            return 0.99 if Ws < min(windspeed) else fCtInterp(Ws)

        return fCp, fCt

    def _calculate_swept_area_velocities(self, local_wind_speed, shear):
        """
            TODO: explain these velocities
            initialize the turbine disk velocities used in the 3D model based on shear using the power log law.
        """
        return [local_wind_speed * ((self.hubHeight + g[1]) / self.hubHeight)**shear for g in self.grid]

    # Public methods

    def update_quantities(self, local_wind_speed, shear):
        self.velocities = self._calculate_swept_area_velocities(local_wind_speed, shear)
        self.Cp = self._calculate_cp()
        self.Ct = self._calculate_ct()
        self.power = self._calculate_power()
        self.aI = self._calculate_ai()
        self.windSpeed = self.get_average_velocity()

    def set_yaw_angle(self, angle):
        self.yawAngle = np.radians(angle)

    def get_grid(self):
        return self.grid

    def get_average_velocity(self):
        return np.mean(self.velocities)
