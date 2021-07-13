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

from typing import Dict, List, Union

import attr
import numpy as np
from scipy.interpolate import interp1d
from numpy.lib.index_tricks import ix_

from src.utilities import FromDictMixin, cosd, float_attrib, attrs_array_converter
from src.base_model import BaseModel


def power(
    air_density: Union[float, np.ndarray],
    average_velocity: Union[float, np.ndarray],
    yaw_angle: Union[float, np.ndarray],
    pP: Union[float, np.ndarray],
    power_interp: callable,
    ix_filter: Union[List[int], np.ndarray] = None,
) -> float:
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

    # NOTE: The below has a trivial performance hit for floats being passed (3.4% longer
    # on a meaningless test), but is actually faster when an array is passed through
    # That said, it adds overhead to convert the floats to 1-D arrays, so I don't
    # recommend just converting all values to arrays
    if isinstance(ix_filter, list):
        ix_filter = np.array(ix_filter)

    if ix_filter is None and isinstance(ix_filter, (list, np.ndarray)):
        ix_filter = np.ones(len(air_density), dtype=bool)

    if ix_filter is not None:
        air_density = air_density[ix_filter]
        average_velocity = average_velocity[ix_filter]
        yaw_angle = yaw_angle[ix_filter]
        pP = pP[ix_filter]

    # Compute the yaw effective velocity
    pW = pP / 3.0  # Convert from pP to w
    yaw_effective_velocity = average_velocity * cosd(yaw_angle) ** pW
    return air_density * power_interp(yaw_effective_velocity)


def Ct(
    velocities: np.ndarray,  # rows: turbines; columns: velocities
    yaw_angle: Union[float, np.ndarray],
    fCt: Union[callable, List[callable]],
    ix_filter: Union[List[int], np.ndarray] = None,
) -> np.ndarray:
    """
    Thrust coefficient of a turbine incorporating the yaw angle.
    The value is interpolated from the coefficient of thrust vs
    wind speed table using the rotor swept area average velocity.
    """
    if isinstance(ix_filter, list):
        ix_filter = np.array(ix_filter)

    if isinstance(fCt, list):
        fCt = np.ndarray(fCt)

    if ix_filter is None and isinstance(ix_filter, (list, np.ndarray)):
        ix_filter = np.ones(len(yaw_angle), dtype=bool)

    if ix_filter is not None:
        velocities = velocities[ix_filter]
        yaw_angle = yaw_angle[ix_filter]
        fCt = fCt[ix_filter]

    Ct = [fCt[i](average_velocity(velocities[i])) for i in range(len(fCt))]
    Ct *= cosd(yaw_angle)  # **self.pP
    return Ct


def axial_induction(
    Ct: Union[float, np.ndarray], yaw_angle: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Axial induction factor of the turbine incorporating
    the thrust coefficient and yaw angle.
    """
    return 0.5 / cosd(yaw_angle) * (1 - np.sqrt(1 - Ct * cosd(yaw_angle)))


def average_velocity(velocities: np.ndarray) -> float:
    """
    This property calculates and returns the cube root of the
    mean cubed velocity in the turbine's rotor swept area (m/s).

    **Note:** The velocity is scaled to an effective velocity by the yaw.

    Returns:
        float: The average velocity across a rotor.

    Examples:
        To get the average velocity for a turbine:

        >>> avg_vel = floris.farm.turbines[0].average_velocity()
    """
    # Remove all invalid numbers from interpolation
    # data = np.array(self.velocities)[~np.isnan(self.velocities)]

    # axis=0 supports multi-turbine input if each turbine is a row
    return np.cbrt(np.mean(velocities ** 3, axis=0))


@attr.s(frozen=True, auto_attribs=True)
class PowerThrustTable(FromDictMixin):
    power: List[float] = attr.ib(converter=attrs_array_converter)
    thrust: List[float] = attr.ib(converter=attrs_array_converter)
    wind_speed: List[float] = attr.ib(converter=attrs_array_converter)

    def __attrs_post_init__(self) -> None:
        inputs = (self.power, self.thrust, self.wind_speed)
        if any(el.ndim > 1 for el in inputs):
            raise ValueError("power, thrust, and wind_speed inputs must be 1-D!")
        if self.power.size != sum(el.size for el in inputs) / 3:
            raise ValueError(
                "power, thrust, and wind_speed inputs must be the same size!"
            )


@attr.s(auto_attribs=True)
class Turbine(BaseModel):
    """
    Turbine is a class containing objects pertaining to the individual
    turbines.

    Turbine is a model class representing a particular wind turbine. It
    is largely a container of data and parameters, but also contains
    methods to probe properties for output.

    Parameters:
        rotor_diameter (:py:obj: float): The rotor diameter (m).
        hub_height (:py:obj: float): The hub height (m).
        blade_count (:py:obj: float): The number of blades.
        pP (:py:obj: float): The cosine exponent relating the yaw
            misalignment angle to power.
        pT (:py:obj: float): The cosine exponent relating the rotor
            tilt angle to power.
        generator_efficiency (:py:obj: float): The generator
            efficiency factor used to scale the power production.
        power_thrust_table (:py:obj: float): A dictionary containing the
            following key-value pairs:

            power (:py:obj: List[float]): The coefficient of power at
                different wind speeds.
            thrust (:py:obj: List[float]): The coefficient of thrust
                at different wind speeds.
            wind_speed (:py:obj: List[float]): The wind speeds for
                which the power and thrust values are provided (m/s).

        yaw_angle (:py:obj: float): The yaw angle of the turbine
            relative to the wind direction (deg). A positive value
            represents a counter-clockwise rotation relative to the
            wind direction.
        tilt_angle (:py:obj: float): The tilt angle of the turbine
            (deg). Positive values correspond to a downward rotation of
            the rotor for an upstream turbine.
        TSR (:py:obj: float): The tip-speed ratio of the turbine. This
            parameter is used in the "curl" wake model.
        ngrid (*int*, optional): The square root of the number
            of points to use on the turbine grid. This number will be
            squared so that the points can be evenly distributed.
            Defaults to 5.
        rloc (:py:obj: float, optional): A value, from 0 to 1, that determines
            the width/height of the grid of points on the rotor as a ratio of
            the rotor radius.
            Defaults to 0.5.
    """

    rotor_diameter: float = float_attrib()
    hub_height: float = float_attrib()
    pP: float = float_attrib()
    pT: float = float_attrib()
    generator_efficiency: float = float_attrib()
    power_thrust_table: Dict[str, List[float]] = attr.ib(
        converter=PowerThrustTable.from_dict, kw_only=True,
    )

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

    def __attrs_post_init__(self) -> None:
        # Post-init initialization for the power curve interpolation functions
        wind_speeds = self.power_thrust_table.wind_speed
        self.fCp_interp = interp1d(
            wind_speeds, self.power_thrust_table.power, fill_value="extrapolate",
        )
        self.fCt_interp = interp1d(
            wind_speeds, self.power_thrust_table.thrust, fill_value="extrapolate",
        )

        inner_power = np.array([self._power_inner_function(ws) for ws in wind_speeds])
        self.power_interp = interp1d(wind_speeds, inner_power, fill_value="extrapolate")

    def _power_inner_function(self, yaw_effective_velocity: float) -> float:
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

    def fCp(self, sample_wind_speeds):
        # NOTE: IS THIS SUPPOSED TO BE A SINGLE INPUT?
        if sample_wind_speeds < self.power_thrust_table.wind_speed.min():
            return 0.0
        else:
            _cp = self.fCpInterp(sample_wind_speeds)
            if _cp.size > 1:
                _cp = _cp[0]
            if _cp > 1.0:
                return 1.0
            if _cp < 0.0:
                return 0.0
            return float(_cp)

    def fCt(self, at_wind_speed):
        # NOTE: IS THIS SUPPOSED TO BE A SINGLE INPUT?
        if at_wind_speed < self.power_thrust_table.wind_speed.min():
            return 0.99
        else:
            _ct = self.fCtInterp(at_wind_speed)
            if _ct.size > 1:
                _ct = _ct[0]
            if _ct > 1.0:
                return 0.9999
            if _ct <= 0.0:
                return 0.0001
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
