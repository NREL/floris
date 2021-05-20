# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import numpy as np
from scipy.interpolate import interp1d
from .utilities import cosd, sind, tand
from .logging_manager import LoggerBase


def power(air_density: float, average_velocity: float, yaw_angle: float, pP: float, power_interp: callable) -> float:
    """
    Power produced by a turbine adjusted for yaw and tilt. Value
    given in Watts.
    """
    # Update to power calculation which replaces the fixed pP exponent with
    # an exponent pW, that changes the effective wind speed input to the power
    # calculation, rather than scaling the power.  This better handles power
    # loss to yaw in above rated conditions
    #
    # based on the paper "Optimising yaw control at wind farm level" by
    # Ervin Bossanyi

    # Compute the yaw effective velocity
    pW = pP / 3.0  # Convert from pP to w
    yaw_effective_velocity = average_velocity * cosd(yaw_angle) ** pW
    return air_density * power_interp(yaw_effective_velocity)


def Ct(average_velocity: float, yaw_angle: float, fCt: callable) -> float:
    """
    Thrust coefficient of a turbine incorporating the yaw angle.
    The value is interpolated from the coefficient of thrust vs
    wind speed table using the rotor swept area average velocity.
    """
    return fCt(average_velocity) * cosd(yaw_angle)  # **self.pP


def axial_induction(Ct: float, yaw_angle: float) -> float:
    """
    Axial induction factor of the turbine incorporating
    the thrust coefficient and yaw angle.
    """
    return 0.5 / cosd(yaw_angle) * (1 - np.sqrt(1 - Ct * cosd(yaw_angle) ) )


def average_velocity(velocities: list[list[float]]) -> float:
    """
    This property calculates and returns the cube root of the
    mean cubed velocity in the turbine's rotor swept area (m/s).

    **Note:** The velocity is scalled to an effective velocity by the yaw.

    Returns:
        float: The average velocity across a rotor.

    Examples:
        To get the average velocity for a turbine:

        >>> avg_vel = floris.farm.turbines[0].average_velocity()
    """
    # Remove all invalid numbers from interpolation
    # data = np.array(self.velocities)[~np.isnan(self.velocities)]
    print("**", velocities)
    return np.cbrt(np.mean(velocities ** 3))


class Turbine(LoggerBase):
    """
    Turbine is a class containing objects pertaining to the individual
    turbines.

    Turbine is a model class representing a particular wind turbine. It
    is largely a container of data and parameters, but also contains
    methods to probe properties for output.
    """
    def __init__(self, input_dictionary):
        """
        Args:
            input_dictionary: A dictionary containing the initialization data for
                the turbine model; it should have the following key-value pairs:

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

        self.rotor_diameter: float = input_dictionary["rotor_diameter"]
        self.hub_height: float = input_dictionary["hub_height"]
        self.pP: float = input_dictionary["pP"]
        self.pT: float = input_dictionary["pT"]
        self.generator_efficiency: float = input_dictionary["generator_efficiency"]
        self.power_thrust_table: list = input_dictionary["power_thrust_table"]

        # For the following parameters, use default values if not user-specified
        # self.ngrid = int(input_dictionary["ngrid"]) if "ngrid" in input_dictionary else 5
        # self.rloc = float(input_dictionary["rloc"]) if "rloc" in input_dictionary else 0.5
        # if "use_points_on_perimeter" in input_dictionary:
        #     self.use_points_on_perimeter = bool(input_dictionary["use_points_on_perimeter"])
        # else:
        #     self.use_points_on_perimeter = False

        # # initialize to an invalid value until calculated
        # self.air_density = -1
        # self.use_turbulence_correction = False

        # Precompute interpolation functions
        wind_speeds = self.power_thrust_table["wind_speed"]
        self.fCpInterp = interp1d(wind_speeds, self.power_thrust_table["power"], fill_value="extrapolate")
        self.fCtInterp = interp1d(wind_speeds, self.power_thrust_table["thrust"], fill_value="extrapolate")
        inner_power = np.array([self._power_inner_function(ws) for ws in wind_speeds])
        self.power_interp = interp1d(wind_speeds, inner_power, fill_value="extrapolate")

    def _power_inner_function(self, yaw_effective_velocity):
        """
        This method calculates the power for an array of yaw effective wind
        speeds without the air density and turbulence correction parameters.
        This is used to initialize the power interpolation method used to
        compute turbine power.
        """

        # Now compute the power
        cptmp = self.fCp(
            yaw_effective_velocity
        )  # Note Cp is also now based on yaw effective velocity
        return (
            0.5
            * (np.pi * self.rotor_radius ** 2)
            * cptmp
            * self.generator_efficiency
            * yaw_effective_velocity ** 3
        )

    def fCp(self, at_wind_speed):
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

    def fCt(self, at_wind_speed):
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

    @property
    def rotor_radius(self) -> float:
        """
        Rotor radius of the turbine in meters.

        Returns:
            float: The rotor radius of the turbine.
        """
        return self.rotor_diameter / 2.0
    
    @rotor_radius.setter
    def rotor_radius(self, value: float) -> None:
        self.rotor_diameter = value * 2.0
