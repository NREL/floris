# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import math

import numpy as np
from scipy.stats import norm
from scipy.spatial import distance_matrix
from scipy.interpolate import interp1d

from ..utilities import cosd, sind, tand
from ..logging_manager import LoggerBase


class Turbine(LoggerBase):
    """
    Turbine is a class containing objects pertaining to the individual
    turbines.

    Turbine is a model class representing a particular wind turbine. It
    is largely a container of data and parameters, but also contains
    methods to probe properties for output.

    Args:
        instance_dictionary: A dictionary that is generated from the
            input_reader; it should have the following key-value pairs:

            -   **description** (*str*): A string containing a description of
                the turbine.
            -   **properties** (*dict*): A dictionary containing the following
                key-value pairs:

                -   **rotor_diameter** (*float*): The rotor diameter (m).
                -   **hub_height** (*float*): The hub height (m).
                -   **blade_count** (*int*): The number of blades.
                -   **pP** (*float*): The cosine exponent relating the yaw
                    misalignment angle to power.
                -   **pT** (*float*): The cosine exponent relating the rotor
                    tilt angle to power.
                -   **generator_efficiency** (*float*): The generator
                    efficiency factor used to scale the power production.
                -   **power_thrust_table** (*dict*): A dictionary containing the
                    following key-value pairs:

                    -   **power** (*list(float)*): The coefficient of power at
                        different wind speeds.
                    -   **thrust** (*list(float)*): The coefficient of thrust
                        at different wind speeds.
                    -   **wind_speed** (*list(float)*): The wind speeds for
                        which the power and thrust values are provided (m/s).

                -   **yaw_angle** (*float*): The yaw angle of the turbine
                    relative to the wind direction (deg). A positive value
                    represents a counter-clockwise rotation relative to the
                    wind direction.
                -   **tilt_angle** (*float*): The tilt angle of the turbine
                    (deg). Positive values correspond to a downward rotation of
                    the rotor for an upstream turbine.
                -   **TSR** (*float*): The tip-speed ratio of the turbine. This
                    parameter is used in the "curl" wake model.
                -   **ngrid** (*int*, optional): The square root of the number
                    of points to use on the turbine grid. This number will be
                    squared so that the points can be evenly distributed.
                    Defaults to 5.
                -   **rloc** (*float, optional): A value, from 0 to 1, that determines
                    the width/height of the grid of points on the rotor as a ratio of
                    the rotor radius.
                    Defaults to 0.5.

    Returns:
        Turbine: An instantiated Turbine object.
    """

    def __init__(self, instance_dictionary):
        self.description = instance_dictionary["description"]

        properties = instance_dictionary["properties"]
        self.rotor_diameter = properties["rotor_diameter"]
        self.hub_height = properties["hub_height"]
        self.blade_count = properties["blade_count"]
        self.pP = properties["pP"]
        self.pT = properties["pT"]
        self.generator_efficiency = properties["generator_efficiency"]
        self.power_thrust_table = properties["power_thrust_table"]
        self.yaw_angle = properties["yaw_angle"]
        self.tilt_angle = properties["tilt_angle"]
        self.tsr = properties["TSR"]

        # For the following parameters, use default values if not user-specified
        self.ngrid = int(properties["ngrid"]) if "ngrid" in properties else 5
        self.rloc = float(properties["rloc"]) if "rloc" in properties else 0.5
        if "use_points_on_perimeter" in properties:
            self.use_points_on_perimeter = bool(properties["use_points_on_perimeter"])
        else:
            self.use_points_on_perimeter = False

        # initialize to an invalid value until calculated
        self.air_density = -1
        self.use_turbulence_correction = False

        self._initialize_turbine()

        # The indices for this Turbine instance's points from the FlowField
        # are set in `FlowField._discretize_turbine_domain` and stored
        # in this variable.
        self.flow_field_point_indices = None

    # Private methods

    def _initialize_turbine(self):
        # Initialize the turbine given saved parameter settings

        # Precompute interps
        wind_speed = self.power_thrust_table["wind_speed"]

        cp = self.power_thrust_table["power"]
        self.fCpInterp = interp1d(wind_speed, cp, fill_value="extrapolate")

        ct = self.power_thrust_table["thrust"]
        self.fCtInterp = interp1d(wind_speed, ct, fill_value="extrapolate")

        # constants
        self.grid_point_count = self.ngrid * self.ngrid
        if np.sqrt(self.grid_point_count) % 1 != 0.0:
            raise ValueError("Turbine.grid_point_count must be the square of a number")

        self.reset_velocities()

        # initialize derived attributes
        self.grid = self._create_swept_area_grid()

        # Compute list of inner powers
        inner_power = np.array([self._power_inner_function(ws) for ws in wind_speed])
        self.powInterp = interp1d(wind_speed, inner_power, fill_value="extrapolate")

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

        pt = self.rloc * self.rotor_radius
        # syntax: np.linspace(min, max, n points)
        horizontal = np.linspace(-pt, pt, num_points)

        vertical = np.linspace(-pt, pt, num_points)

        # build the grid with all of the points
        grid = [(h, vertical[i]) for i in range(num_points) for h in horizontal]

        # keep only the points in the swept area
        if self.use_points_on_perimeter:
            grid = [
                point
                for point in grid
                if np.hypot(point[0], point[1]) <= self.rotor_radius
            ]
        else:
            grid = [
                point
                for point in grid
                if np.hypot(point[0], point[1]) < self.rotor_radius
            ]

        return grid

    def _power_inner_function(self, yaw_effective_velocity):
        """
        This method calculates the power for an array of yaw effective wind
        speeds without the air density and turbulence correction parameters.
        This is used to initialize the power interpolation method used to
        compute turbine power.
        """

        # Now compute the power
        cptmp = self._fCp(
            yaw_effective_velocity
        )  # Note Cp is also now based on yaw effective velocity
        return (
            0.5
            * (np.pi * self.rotor_radius ** 2)
            * cptmp
            * self.generator_efficiency
            * yaw_effective_velocity ** 3
        )

    def _fCp(self, at_wind_speed):
        wind_speed = self.power_thrust_table["wind_speed"]
        if at_wind_speed < min(wind_speed):
            return 0.0
        else:
            _cp = self.fCpInterp(at_wind_speed)
            if _cp.size > 1:
                _cp = _cp[0]
            if _cp > 1.0:
                _cp = 1.0
            if _cp < 0.0:
                _cp = 0.0
            return float(_cp)

    def _fCt(self, at_wind_speed):
        wind_speed = self.power_thrust_table["wind_speed"]
        if at_wind_speed < min(wind_speed):
            return 0.99
        else:
            _ct = self.fCtInterp(at_wind_speed)
            if _ct.size > 1:
                _ct = _ct[0]
            if _ct > 1.0:
                _ct = 0.9999
            if _ct <= 0.0:
                _ct = 0.0001
            return float(_ct)

    # Public methods

    def change_turbine_parameters(self, turbine_change_dict):
        """
        Change a turbine parameter and call the initialize function.

        Args:
            turbine_change_dict (dict): A dictionary of parameters to change.
        """
        for param in turbine_change_dict:
            self.logger.info(
                "Setting {} to {}".format(param, turbine_change_dict[param])
            )
            setattr(self, param, turbine_change_dict[param])
        self._initialize_turbine()

    def calculate_swept_area_velocities(
        self, local_wind_speed, coord, x, y, z, additional_wind_speed=None
    ):
        """
        This method calculates and returns the wind speeds at each
        rotor swept area grid point for the turbine, interpolated from
        the flow field grid.

        Args:
            wind_direction (float): The wind farm wind direction (deg).
            local_wind_speed (np.array): The wind speed at each grid point in
                the flow field (m/s).
            coord (:py:obj:`~.utilities.Vec3`): The coordinate of the turbine.
            x (np.array): The x-coordinates of the flow field grid.
            y (np.array): The y-coordinates of the flow field grid.
            z (np.array): The z-coordinates of the flow field grid.

        Returns:
            np.array: The wind speed at each rotor grid point
            for the turbine (m/s).
        """
        u_at_turbine = local_wind_speed

        # TODO:
        # # PREVIOUS METHOD========================
        # # UNCOMMENT IF ANY ISSUE UNCOVERED WITH NEW MOETHOD
        # x_grid = x
        # y_grid = y
        # z_grid = z

        # yPts = np.array([point[0] for point in self.grid])
        # zPts = np.array([point[1] for point in self.grid])

        # # interpolate from the flow field to get the flow field at the grid
        # # points
        # dist = [np.sqrt((coord.x1 - x_grid)**2 \
        #      + (coord.x2 + yPts[i] - y_grid) **2 \
        #      + (self.hub_height + zPts[i] - z_grid)**2) \
        #      for i in range(len(yPts))]
        # idx = [np.where(dist[i] == np.min(dist[i])) for i in range(len(yPts))]
        # data = [np.mean(u_at_turbine[idx[i]]) for i in range(len(yPts))]
        # # PREVIOUS METHOD========================

        # Use this if no saved points (curl)
        if self.flow_field_point_indices is None:
            # # NEW METHOD========================
            # Sort by distance
            flow_grid_points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

            # Set up a grid array
            y_array = np.array(self.grid)[:, 0] + coord.x2
            z_array = np.array(self.grid)[:, 1] + self.hub_height
            x_array = np.ones_like(y_array) * coord.x1
            grid_array = np.column_stack([x_array, y_array, z_array])

            ii = np.argmin(distance_matrix(flow_grid_points, grid_array), axis=0)
        else:
            ii = self.flow_field_point_indices

        # return np.array(data)
        if additional_wind_speed is not None:
            return (
                np.array(u_at_turbine.flatten()[ii]),
                np.array(additional_wind_speed.flatten()[ii]),
            )
        else:
            return np.array(u_at_turbine.flatten()[ii])

    def return_grid_points(self, coord):
        """
        Retrieve the x, y, and z grid points on the rotor.

        Args:
            coord (:py:obj:`~.utilities.Vec3`): The coordinate of the turbine.

        Returns:
            np.array, np.array, np.array:

                - x grid points on the rotor.
                - y grid points on the rotor.
                - xzgrid points on the rotor.
        """
        y_array = np.array(self.grid)[:, 0] + coord.x2
        z_array = np.array(self.grid)[:, 1] + self.hub_height
        x_array = np.ones_like(y_array) * coord.x1

        return x_array, y_array, z_array

    def update_velocities(
        self, u_wake, coord, flow_field, rotated_x, rotated_y, rotated_z
    ):
        """
        This method updates the velocities at the rotor swept area grid
        points based on the flow field freestream velocities and wake
        velocities.

        Args:
            u_wake (np.array): The wake deficit velocities at all grid points
                in the flow field (m/s).
            coord (:py:obj:`~.utilities.Vec3`): The coordinate of the turbine.
            flow_field (:py:class:`~.flow_field.FlowField`): The flow field.
            rotated_x (np.array): The x-coordinates of the flow field grid
                rotated so the new x axis is aligned with the wind direction.
            rotated_y (np.array): The y-coordinates of the flow field grid
                rotated so the new x axis is aligned with the wind direction.
            rotated_z (np.array): The z-coordinates of the flow field grid
                rotated so the new x axis is aligned with the wind direction.
        """
        # reset the waked velocities
        local_wind_speed = flow_field.u_initial - u_wake
        self.velocities = self.calculate_swept_area_velocities(
            local_wind_speed, coord, rotated_x, rotated_y, rotated_z
        )

    def reset_velocities(self):
        """
        This method sets the velocities at the turbine's rotor swept
        area grid points to zero.
        """
        self.velocities = [0.0] * self.grid_point_count

    def set_yaw_angle(self, yaw_angle):
        """
        This method sets the turbine's yaw angle.

        Args:
            yaw_angle (float): The new yaw angle (deg).

        Examples:
            To set a turbine's yaw angle:

            >>> floris.farm.turbines[0].set_yaw_angle(20.0)
        """
        self._yaw_angle = yaw_angle

    def TKE_to_TI(self, turbulence_kinetic_energy):
        """
        Converts a list of turbulence kinetic energy values to
        turbulence intensity.
        Args:
            turbulence_kinetic_energy (list): Values of turbulence kinetic
                energy in units of meters squared per second squared.
            wind_speed (list): Measurements of wind speed in meters per second.
        Returns:
            list: converted turbulence intensity values expressed as a decimal
            (e.g. 10%TI -> 0.10).
        """
        total_turbulence_intensity = (
            np.sqrt((2 / 3) * turbulence_kinetic_energy)
        ) / self.average_velocity
        return total_turbulence_intensity

    def TI_to_TKE(self):
        """
        Converts TI to TKE.
        Args:
            wind_speed (list): Measurements of wind speed in meters per second.
        Returns:
            list: converted TKE values
        """
        return ((self.average_velocity * self.current_turbulence_intensity) ** 2) / (
            2 / 3
        )

    def u_prime(self):
        """
        Converts a TKE to horizontal deviation component.
        Args:
            wind_speed (list): Measurements of wind speed in meters per second.
        Returns:
            list: converted u_prime values in meters per second
        """
        tke = self.TI_to_TKE()
        return np.sqrt(2 * tke)

    # Getters & Setters

    @property
    def turbulence_parameter(self):
        """
        This property calculates and returns the turbulence correction
        parameter for the turbine, a value used to account for the
        change in power output due to the effects of turbulence.

        Returns:
            float: The value of the turbulence parameter.
        """
        if not self.use_turbulence_correction:
            return 1.0
        else:
            # define wind speed, ti, and power curve components
            ws = np.array(self.power_thrust_table["wind_speed"])
            cp = np.array(self.power_thrust_table["power"])
            ws = ws[np.where(cp != 0)]
            ciws = ws[0]  # cut in wind speed
            cows = ws[len(ws) - 1]  # cut out wind speed
            speed = self.average_velocity
            ti = self.current_turbulence_intensity

            if ciws >= speed or cows <= speed or ti == 0.0 or math.isnan(speed):
                return 1.0
            else:
                # define mean and standard deviation to create normalized pdf with sum = 1
                mu = speed
                sigma = ti * mu
                if mu + sigma >= cows:
                    xp = np.linspace((mu - sigma), cows, 100)
                else:
                    xp = np.linspace((mu - sigma), (mu + sigma), 100)
                pdf = norm.pdf(xp, mu, sigma)
                npdf = np.array(pdf) * (1 / np.sum(pdf))

                # calculate turbulence parameter (ratio of corrected power to original power)
                return np.sum([npdf[k] * self.powInterp(xp[k]) for k in range(100)]) / (
                    self.powInterp(mu)
                )

    @property
    def current_turbulence_intensity(self):
        """
        This method returns the current turbulence intensity at
        the turbine expressed as a decimal fraction.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Examples:
            To get the turbulence intensity for a turbine:

            >>> current_turbulence_intensity = floris.farm.turbines[0].turbulence_intensity()
        """
        return self._turbulence_intensity

    @current_turbulence_intensity.setter
    def current_turbulence_intensity(self, value):
        self._turbulence_intensity = value

    @property
    def rotor_radius(self):
        """
        This method returns the rotor radius of the turbine (m).

        **Note:** This is a virtual property used to "get" a value.

        Returns:
            float: The rotor radius of the turbine.

        Examples:
            To get the rotor radius for a turbine:

            >>> rotor_radius = floris.farm.turbines[0].rotor_radius()
        """
        return self.rotor_diameter / 2.0

    @property
    def yaw_angle(self):
        """
        This method gets or sets the turbine's yaw angle.

        **Note:** This is a virtual property used to "get"  or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Examples:
            To set the yaw angle for each turbine in the wind farm:

            >>> yaw_angles = [20.0, 10.0, 0.0]
            >>> for yaw_angle, turbine in
            ... zip(yaw_angles, floris.farm.turbines):
            ...     turbine.yaw_angle = yaw_angle

            To get the current yaw angle for each turbine in the wind
            farm:

            >>> yaw_angles = []
            >>> for i, turbine in enumerate(floris.farm.turbines):
            ...     yaw_angles.append(turbine.yaw_angle())
        """
        return self._yaw_angle

    @yaw_angle.setter
    def yaw_angle(self, value):
        self._yaw_angle = value

    @property
    def tilt_angle(self):
        """
        This method gets the turbine's tilt angle.

        **Note:** This is a virtual property used to "get"  or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Examples:
            To get the current tilt angle for a turbine:

            >>> tilt_angle = floris.farm.turbines[0].tilt_angle()
        """
        return self._tilt_angle

    @tilt_angle.setter
    def tilt_angle(self, value):
        self._tilt_angle = value

    @property
    def average_velocity(self):
        """
        This property calculates and returns the cube root of the
        mean cubed velocity in the turbine's rotor swept area (m/s).

        Returns:
            float: The average velocity across a rotor.

        Examples:
            To get the average velocity for a turbine:

            >>> avg_vel = floris.farm.turbines[0].average_velocity()
        """
        # remove all invalid numbers from interpolation
        data = self.velocities[np.where(~np.isnan(self.velocities))]
        avg_vel = np.cbrt(np.mean(data ** 3))
        if np.isnan(avg_vel):
            avg_vel = 0
        elif np.isinf(avg_vel):
            avg_vel = 0

        return avg_vel

    @property
    def Cp(self):
        """
        This property returns the power coeffcient of a turbine.

        This property returns the coefficient of power of the turbine
        using the rotor swept area average velocity, interpolated from
        the coefficient of power table. The average velocity is
        calculated as the cube root of the mean cubed velocity in the
        rotor area.

        **Note:** The velocity is scalled to an effective velocity by the yaw.

        Returns:
            float: The power coefficient of a turbine at the current
            operating conditions.

        Examples:
            To get the power coefficient value for a turbine:

            >>> Cp = floris.farm.turbines[0].Cp()
        """
        # Compute the yaw effective velocity
        pW = self.pP / 3.0  # Convert from pP to pW
        yaw_effective_velocity = self.average_velocity * cosd(self.yaw_angle) ** pW

        return self._fCp(yaw_effective_velocity)

    @property
    def Ct(self):
        """
        This property returns the thrust coefficient of a turbine.

        This method returns the coefficient of thrust of the yawed
        turbine, interpolated from the coefficient of power table,
        using the rotor swept area average velocity and the turbine's
        yaw angle. The average velocity is calculated as the cube root
        of the mean cubed velocity in the rotor area.

        Returns:
            float: The thrust coefficient of a turbine at the current
            operating conditions.

        Examples:
            To get the thrust coefficient value for a turbine:

            >>> Ct = floris.farm.turbines[0].Ct()
        """
        return self._fCt(self.average_velocity) * cosd(self.yaw_angle)  # **self.pP

    @property
    def power(self):
        """
        This property returns the power produced by turbine (W),
        adjusted for yaw and tilt.

        Returns:
            float: Power of a turbine in watts.

        Examples:
            To get the power for a turbine:

            >>> power = floris.farm.turbines[0].power()
        """
        # Update to power calculation which replaces the fixed pP exponent with
        # an exponent pW, that changes the effective wind speed input to the power
        # calculation, rather than scaling the power.  This better handles power
        # loss to yaw in above rated conditions
        #
        # based on the paper "Optimising yaw control at wind farm level" by
        # Ervin Bossanyi

        # Compute the yaw effective velocity
        pW = self.pP / 3.0  # Convert from pP to w
        yaw_effective_velocity = self.average_velocity * cosd(self.yaw_angle) ** pW

        # Now compute the power
        return (
            self.air_density
            * self.powInterp(yaw_effective_velocity)
            * self.turbulence_parameter
        )

    @property
    def aI(self):
        """
        This property returns the axial induction factor of the yawed
        turbine calculated from the coefficient of thrust and the yaw
        angle.

        Returns:
            float: Axial induction factor of a turbine.

        Examples:
            To get the axial induction factor for a turbine:

            >>> aI = floris.farm.turbines[0].aI()
        """
        return (
            0.5
            / cosd(self.yaw_angle)
            * (1 - np.sqrt(1 - self.Ct * cosd(self.yaw_angle)))
        )
