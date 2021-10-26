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
from collections.abc import Iterable
import math

import attr
import numpy as np
from scipy.interpolate import interp1d

from src.utilities import FromDictMixin, cosd, float_attrib, attrs_array_converter
from src.base_class import BaseClass


def _filter_convert(
    ix_filter: Union[List[Union[int, bool]], np.ndarray], sample_arg: np.ndarray
) -> Union[np.ndarray, None]:
    """Converts the ix_filter to a standard format of `np.ndarray`s for filtering
    certain arguments.

    Args:
        ix_filter (Union[List[Union[int, bool]], np.ndarray]): The indices, or truth
            array-like object to use for filtering.
        sample_arg (np.ndarray): Any argument that will be filtered, to be used for
            creating the shape.

    Returns:
        Union[np.ndarray, None]: Returns an array of a truth or index list if a list is
            passed, a truth array if ix_filter is None, or None if ix_filter is None
            and the `sample_arg` is a single value.
    """
    if isinstance(ix_filter, list):
        return np.array(ix_filter)
    if ix_filter is None and isinstance(sample_arg, np.ndarray):
        return np.ones(sample_arg.shape[0], dtype=bool)
    return None


def power(
    air_density: Union[float, np.ndarray],
    velocities: np.ndarray,  # rows: turbines; columns: velocities
    yaw_angle: Union[float, np.ndarray],
    pP: Union[float, np.ndarray],
    power_interp: Union[callable, List[callable]],
    ix_filter: Union[List[int], np.ndarray] = None,
) -> float:
    """Power produced by a turbine adjusted for yaw and tilt. Value
    given in Watts.

    Args:
        air_density (Union[float, np.ndarray]): The air density value(s) at each turbine.
        velocities (np.ndarray): The velocity field at a turbine. Shape should be: (number
            of turbines, ngrid, ngrid), or (ngrid, ngrid) for a single turbine..
        pP (Union[float, np.ndarray]): The pP value(s) of the cosine exponent relating
            the yaw misalignment angle to power for each turbine.
        power_interp (Union[callable, List[callable]]): The power interpolation function
            for each turbine.
        ix_filter (Union[List[int], np.ndarray], optional): The boolean array, or
            integer indices to filter out before calculation. Defaults to None.

    Returns:
        float: The power, in Watts, for each turbine after adjusting for yaw and tilt.
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

    if isinstance(air_density, list):
        air_density = np.array(air_density)
    if isinstance(yaw_angle, list):
        yaw_angle = np.array(yaw_angle)
    if isinstance(pP, list):
        pP = np.array(pP)
    if isinstance(power_interp, list):
        power_interp = np.array(power_interp)

    ix_filter = _filter_convert(ix_filter, air_density)
    if ix_filter is not None:
        air_density = air_density[ix_filter]
        velocities = velocities[ix_filter]
        yaw_angle = yaw_angle[ix_filter]
        pP = pP[ix_filter]
        power_interp = power_interp[ix_filter]

    # Compute the yaw effective velocity
    pW = pP / 3.0  # Convert from pP to w
    yaw_effective_velocity = average_velocity(velocities) * cosd(yaw_angle) ** pW

    if isinstance(power_interp, np.ndarray):
        p = np.array([_fCp(v) for _fCp, v in zip(power_interp, yaw_effective_velocity)])
    else:
        p = power_interp(yaw_effective_velocity)
    return p * air_density


def Ct(
    velocities: np.ndarray,  # rows: turbines; columns: velocities
    yaw_angle: Union[float, np.ndarray],
    fCt: Union[callable, List[callable]],
    ix_filter: Union[List[int], np.ndarray] = None,
) -> np.ndarray:
    """Thrust coefficient of a turbine incorporating the yaw angle.
    The value is interpolated from the coefficient of thrust vs
    wind speed table using the rotor swept area average velocity.

    Args:
        velocities (np.ndarray): The velocity field at the turbine; should be shape:
            (number of turbines, ngrid, ngrid), or (ngrid, ngrid) for a single turbine..
        fCt (Union[callable, List[callable]]): The thrust coefficient for each turbine.
        ix_filter (Union[List[int], np.ndarray], optional): The boolean array, or
            integer indices to filter out before calculation. Defaults to None.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        np.ndarray: [description]
    """

    if isinstance(yaw_angle, list):
        yaw_angle = np.array(yaw_angle)
    if isinstance(fCt, list):
        fCt = np.array(fCt)

    ix_filter = _filter_convert(ix_filter, yaw_angle)
    if ix_filter is not None:
        velocities = velocities[ix_filter, :, :]
        yaw_angle = yaw_angle[ix_filter]
        fCt = fCt[ix_filter]

    if isinstance(fCt, np.ndarray):
        Ct = np.array([_fCt(average_velocity(v)) * cosd(yaw) for _fCt, v, yaw in zip(fCt, velocities, yaw_angle)])
    else:
        turbine_velocity = average_velocity(velocities)
        Ct = fCt(turbine_velocity) * cosd(yaw_angle)

    return Ct


def axial_induction(
    velocities: np.ndarray,  # rows: turbines; columns: velocities
    yaw_angle: Union[float, np.ndarray],
    fCt: Union[callable, List[callable]],
    ix_filter: Union[List[int], np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """Axial induction factor of the turbine incorporating
    the thrust coefficient and yaw angle.

    Args:
        velocities (np.ndarray): The velocity field at each turbine; should be shape:
            (number of turbines, ngrid, ngrid), or (ngrid, ngrid) for a single turbine.
        fCt (Union[callable, List[callable]]): The thrust coefficient function for each
            turbine.
        ix_filter (Union[List[int], np.ndarray], optional): The boolean array, or
            integer indices to filter out before calculation. Defaults to None.

    Returns:
        Union[float, np.ndarray]: [description]
    """

    if isinstance(fCt, list):
        fCt = np.array(fCt)
    if isinstance(yaw_angle, list):
        yaw_angle = np.array(yaw_angle)

    ix_filter = _filter_convert(ix_filter, yaw_angle)
    if ix_filter is not None:
        velocities = velocities[ix_filter, :, :]
        yaw_angle = yaw_angle[ix_filter]
        fCt = fCt[ix_filter]

    thrust_coefficient = Ct(velocities, yaw_angle, fCt)
    return (
        0.5 / cosd(yaw_angle) * (1 - np.sqrt(1 - thrust_coefficient * cosd(yaw_angle)))
    )


def average_velocity(
    velocities: np.ndarray, ix_filter: Union[List[Union[int, bool]], np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """This property calculates and returns the cube root of the
    mean cubed velocity in the turbine's rotor swept area (m/s).

    **Note:** The velocity is scaled to an effective velocity by the yaw.

    Args:
        velocities (np.ndarray): The velocity field at each turbine; should be shape:
            (number of turbines, ngrid, ngrid), or (ngrid, ngrid) for a single turbine.
        ix_filter (Union[List[Union[int, bool]], np.ndarray], optional): The boolean array, or
            integer indices to filter out before calculation. Defaults to None.

    Returns:
        Union[float, np.ndarray]: The average velocity across the rotor(s).
    """
    # Remove all invalid numbers from interpolation
    # data = np.array(self.velocities)[~np.isnan(self.velocities)]
    ix_filter = _filter_convert(ix_filter, velocities)
    if velocities.ndim == 3:
        velocities = velocities[ix_filter, :, :]
    else:
        velocities[:, :]

    axis = (1, 2) if velocities.ndim == 3 else (0, 1)
    return np.cbrt(np.mean(velocities ** 3, axis=axis))


@attr.s(frozen=True, auto_attribs=True)
class PowerThrustTable(FromDictMixin):
    """Helper class to convert the dictionary and list-based inputs to a object of arrays.

    Args:
        power ([type]): The power produced at a given windspeed.
        thrust ([type]): The thrust at a given windspeed.
        wind_speed ([type]): Windspeed values, m/s.

    Raises:
        ValueError: Raised if the power, thrust, and wind_speed are not all 1-d array-like shapes.
        ValueError: Raised if power, thrust, and wind_speed don't have the same number of values.
    """

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
class Turbine(BaseClass):
    """
    Turbine is a class containing objects pertaining to the individual
    turbines.

    Turbine is a model class representing a particular wind turbine. It
    is largely a container of data and parameters, but also contains
    methods to probe properties for output.

    Parameters:
        rotor_diameter (:py:obj: float): The rotor diameter (m).
        hub_height (:py:obj: float): The hub height (m).
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
    # ngrid: float = float_attrib()  # TODO: goes here or on the Grid?
    # rloc: float = float_attrib()  # TODO: goes here or on the Grid?
    # use_points_on_perimeter: bool = bool_attrib()

    # Initialized in the post_init function
    fCp_interp: interp1d = attr.ib(init=False)
    fCt_interp: interp1d = attr.ib(init=False)
    power_interp: interp1d = attr.ib(init=False)

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

    def _power_inner_function(self, velocities: List[float]) -> List[float]:
        """
        This method calculates the power for an array of yaw effective wind
        speeds without the air density and turbulence correction parameters.
        This is used to initialize the power interpolation method used to
        compute turbine power.
        """
        return (
            0.5
            * self.rotor_area
            * self.fCp(velocities)
            * self.generator_efficiency
            * velocities ** 3
        )

    def fCp(
        self, sample_wind_speeds: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """????

        Args:
            sample_wind_speeds (Union[float, np.ndarray]): The wind speed(s).

        Returns:
            Union[float, np.ndarray]: The coefficient of power for a given wind speed.
        """
        is_single = not isinstance(sample_wind_speeds, Iterable)
        if is_single:
            sample_wind_speeds = np.array([sample_wind_speeds])

        _cp = self.fCp_interp(sample_wind_speeds)
        _cp = np.clip(_cp, 0, 1)

        # TODO: What are the circumstances that led to this?
        # if _cp.size > 1:
        #     _cp = _cp[0]
        _cp[sample_wind_speeds < self.power_thrust_table.wind_speed.min()] = 0.0

        # Return the data type that matches the input size
        if is_single:
            return _cp[0]
        return _cp

    def fCt(self, at_wind_speed):
        # NOTE: IS THIS SUPPOSED TO BE A SINGLE INPUT?
        if at_wind_speed < self.power_thrust_table.wind_speed.min():
            return 0.9999
        else:
            _ct = self.fCt_interp(at_wind_speed)
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

    @property
    def rotor_area(self) -> float:
        """
        Rotor area of the turbine in meters squared.

        Returns:
            float: The rotor area of the turbine.
        """
        return np.pi * self.rotor_radius ** 2

    @rotor_area.setter
    def rotor_area(self, value: float) -> None:
        self.rotor_radius = math.sqrt(value / np.pi)
