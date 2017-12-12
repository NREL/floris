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

    def __init__(self, instance_dictionary):

        super().__init__()

        # constants
        self.grid_point_count = 16
        self.velocities = [0] * self.grid_point_count
        self.grid = [0] * self.grid_point_count

        self.description = instance_dictionary["description"]

        # loop through all the properties defined in the input dict and 
        # store as attributes of the turbine object
        # included attributes are found in InputReader._turbine_properties
        for key, value in instance_dictionary["properties"].items():
            setattr(self, key, value)

        # these attributes need special attention
        self.rotor_radius = self.rotor_diameter / 2.
        self.yaw_angle = np.radians(self.yaw_angle)
        self.tilt_angle = np.radians(self.tilt_angle)

        # initialize derived attributes
        self.fCp, self.fCt = self._CpCtWs()
        self.grid = self._create_swept_area_grid()
        self.velocities = [-1] * 16  # initialize to an invalid value until calculated

        # calculated attributes are
        # self.Ct         # Thrust Coefficient
        # self.Cp         # Power Coefficient
        # self.power      # Power (W) <-- True?
        # self.aI         # Axial Induction
        # self.TI         # Turbulence intensity at rotor
        # self.windSpeed  # Windspeed at rotor

        # self.usePitch = usePitch
        # if usePitch:
        #     self.Cp, self.Ct, self.betaLims = CpCtpitchWs()
        # else:
        #     self.Cp, self.Ct = CpCtWs()

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
        num_points = int(np.round(np.sqrt(self.grid_point_count)))
        # syntax: np.linspace(min, max, n points)
        horizontal = np.linspace(-self.rotor_diameter/2, self.rotor_diameter/2, num_points)
        vertical = np.linspace(-self.rotor_diameter/2, self.rotor_diameter/2, num_points)

        # build the grid with all of the points
        grid = [(h, vertical[i]) for i in range(num_points) for h in horizontal]

        # keep only the points in the swept area
        # grid = [point for point in grid if np.hypot(point[0], point[1]) < self.rotor_diameter/2]

        return grid

    def _calculate_cp(self):
        # with average velocity
        return self.fCp(self.get_average_velocity())

    def _calculate_ct(self):
        # with average velocity        
        return self.fCt(self.get_average_velocity())

    def _calculate_power(self):
        cptmp = self.Cp \
                * np.cos(self.yaw_angle * np.pi / 180.)**self.pP \
                * np.cos(self.tilt_angle * np.pi / 180.)**self.pT

        #TODO: air density (1.225) is hard coded below. should this be variable in the flow field?
        return 0.5 * 1.225 * (np.pi * (self.rotor_diameter/2)**2) * cptmp * self.generator_efficiency * self.get_average_velocity()**3

    def _calculate_ai(self):
        return 0.5 / np.cos(self.yaw_angle * np.pi / 180.) \
               * (1 - np.sqrt(1 - self.Ct * np.cos(self.yaw_angle * np.pi / 180) ) )

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
        return [local_wind_speed * ((self.hub_height + g[1]) / self.hub_height)**shear for g in self.grid]

    # Public methods

    def update_quantities(self, local_wind_speed, shear):
        self.velocities = self._calculate_swept_area_velocities(local_wind_speed, shear)
        self.Cp = self._calculate_cp()
        self.Ct = self._calculate_ct()
        self.power = self._calculate_power()
        self.aI = self._calculate_ai()
        self.windSpeed = self.get_average_velocity()

    def set_yaw_angle(self, angle):
        """
        Sets the turbine yaw angle
        inputs:
            angle: float - new yaw angle in degrees
        outputs:
            none
        """
        self.yaw_angle = np.radians(angle)

    def get_grid(self):
        return self.grid

    def get_average_velocity(self):
        return np.mean(self.velocities)
