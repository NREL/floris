# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import griddata


class Turbine():
    """
    Turbine is model object representing a particular wind turbine. It is largely
    a container of data and parameters, but also contains method to probe properties
    for output.

    inputs:
        instance_dictionary: dict - the input dictionary as generated from the input_reader;
            it should have the following key-value pairs:
                {
        
                    "rotor_diameter": float,
        
                    "hub_height": float,
        
                    "blade_count": int,
        
                    "pP": float,
        
                    "pT": float,
        
                    "generator_efficiency": float,
        
                    "eta": float,
        
                    "power_thrust_table": dict,
        
                    "yaw_angle": float,
        
                    "tilt_angle": float,
        
                    "TSR": float
        
                }

    outputs:
        self: Turbine - an instantiated Turbine object
    """

    def __init__(self, instance_dictionary):

        # constants
        self.grid_point_count = 16
        if np.sqrt(self.grid_point_count) % 1 != 0.0:
            raise ValueError("Turbine.grid_point_count must be the square of a number")

        self.velocities = [0] * self.grid_point_count
        self.grid = [0] * self.grid_point_count

        self.description = instance_dictionary["description"]

        properties = instance_dictionary["properties"]
        self.rotor_diameter = properties["rotor_diameter"]
        self.hub_height = properties["hub_height"]
        self.blade_count = properties["blade_count"]
        self.pP = properties["pP"]
        self.pT = properties["pT"]
        self.generator_efficiency = properties["generator_efficiency"]
        self.eta = properties["eta"]
        self.power_thrust_table = properties["power_thrust_table"]
        self.yaw_angle = properties["yaw_angle"]
        self.tilt_angle = properties["tilt_angle"]
        self.tsr = properties["TSR"]

        # these attributes need special attention
        self.rotor_radius = self.rotor_diameter / 2.0

        # initialize derived attributes
        self.grid = self._create_swept_area_grid()
        # initialize to an invalid value until calculated
        self.velocities = [-1] * self.grid_point_count
        self.air_density = -1

        # calculated attributes are
        # self.Ct                   # Thrust Coefficient
        # self.Cp                   # Power Coefficient
        # self.power                # Power (W)
        # self.aI                   # Axial Induction
        # self.windSpeed            # Windspeed at rotor (m/s)
        # self.turbulence_intensity # turbulence intensity at a downstream turbine

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
        horizontal = np.linspace(-self.rotor_radius, self.rotor_radius, num_points)
        vertical = np.linspace(-self.rotor_radius, self.rotor_radius, num_points)

        # build the grid with all of the points
        grid = [(h, vertical[i]) for i in range(num_points) for h in horizontal]

        # keep only the points in the swept area
        # grid = [point for point in grid if np.hypot(point[0], point[1]) < self.rotor_radius]

        return grid

    def _fCp(self, at_wind_speed):
        cp = self.power_thrust_table["power"]
        wind_speed = self.power_thrust_table["wind_speed"]
        fCpInterp = interp1d(wind_speed, cp, fill_value='extrapolate')
        if at_wind_speed < min(wind_speed):
            return max(cp)
        else:
            _cp = fCpInterp(at_wind_speed)
            if _cp.size > 1:
                _cp = _cp[0]
            return float(_cp)

    def _fCt(self, at_wind_speed):
        ct = self.power_thrust_table["thrust"]
        wind_speed = self.power_thrust_table["wind_speed"]
        fCtInterp = interp1d(wind_speed, ct, fill_value='extrapolate')
        if at_wind_speed < min(wind_speed):
            return 0.99
        else:
            _ct = fCtInterp(at_wind_speed)
            if _ct.size > 1:
                _ct = _ct[0]
            return float(_ct)

    def _calculate_swept_area_velocities(self, wind_direction, local_wind_speed, coord, x, y, z):
        """
        Initialize the turbine disk velocities used in the 3D model based on shear using the power log law.
        """
        u_at_turbine = local_wind_speed
        x_grid = x
        y_grid = y
        z_grid = z

        yPts = np.array([point[0] for point in self.grid])
        zPts = np.array([point[1] for point in self.grid])

        # interpolate from the flow field to get the flow field at the grid points
        dist = [np.sqrt((coord.x1 - x_grid)**2 + (coord.x2 + yPts[i] - y_grid)**2 + (self.hub_height + zPts[i] - z_grid)**2) for i in range(len(yPts))]
        idx = [np.where(dist[i] == np.min(dist[i])) for i in range(len(yPts))]
        data = [u_at_turbine[idx[i]] for i in range(len(yPts))]
        data = [np.mean(u_at_turbine[idx[i]]) for i in range(len(yPts))]

        return np.array(data)

    # Public methods

    def calculate_turbulence_intensity(self, flow_field_ti, velocity_model, turbine_coord, wake_coord, turbine_wake):
        """
        flow_field_ti
        velocity_model
        turbine_coord
        wake_coord
        turbine_wake
        """

        ti_initial = flow_field_ti

        # user-input turbulence intensity parameters
        ti_i = velocity_model.ti_initial
        ti_constant = velocity_model.ti_constant
        ti_ai = velocity_model.ti_ai
        ti_downstream = velocity_model.ti_downstream

        # turbulence intensity calculation based on Crespo et. al.
        ti_calculation = ti_constant \
                       * turbine_wake.aI**ti_ai \
                       * ti_initial**ti_i \
                       * ((turbine_coord.x1 - wake_coord.x1) / self.rotor_diameter)**ti_downstream

        return np.sqrt(ti_calculation**2 + flow_field_ti**2)

    def update_velocities(self, u_wake, coord, flow_field, rotated_x, rotated_y, rotated_z):
        """
        """
        # reset the initial velocities
        self.initial_velocities = self._calculate_swept_area_velocities(
            flow_field.wind_direction,
            flow_field.u_initial,
            coord,
            rotated_x,
            rotated_y,
            rotated_z
        )

        # reset the waked velocities
        local_wind_speed = flow_field.u_initial - u_wake
        self.velocities = self._calculate_swept_area_velocities(
            flow_field.wind_direction,
            local_wind_speed,
            coord,
            rotated_x,
            rotated_y,
            rotated_z
        )



    # Getters & Setters
    @property
    def yaw_angle(self):
        return self._yaw_angle

    @yaw_angle.setter
    def yaw_angle(self, value):
        self._yaw_angle = np.radians(value)

    @property
    def tilt_angle(self):
        return self._tilt_angle

    @tilt_angle.setter
    def tilt_angle(self, value):
        self._tilt_angle = np.radians(value)

    @property
    def average_velocity(self):
        return np.mean(self.velocities)

    @property
    def Cp(self):
        return self._fCp(self.average_velocity)
    
    @property
    def Ct(self):
        return self._fCt(self.average_velocity)

    @property
    def power(self):
        cptmp = self.Cp \
                * np.cos(self.yaw_angle)**self.pP \
                * np.cos(self.tilt_angle)**self.pT
        return 0.5 * self.air_density * (np.pi * self.rotor_radius**2) \
                * cptmp * self.generator_efficiency \
                * self.average_velocity**3

    @property
    def aI(self):
        return 0.5 / np.cos(self.yaw_angle) \
            * (1 - np.sqrt(1 - self.Ct * np.cos(self.yaw_angle)))
