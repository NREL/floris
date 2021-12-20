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
from __future__ import annotations

import math
from typing import Union
from collections.abc import Iterable

import attr
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from floris.utilities import (
    FromDictMixin,
    cosd,
    float_attrib,
    model_attrib,
    attr_serializer,
    attr_floris_filter,
    attrs_array_converter,
)
from floris.simulation import BaseClass


NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFilter = Union[npt.NDArray[np.int_], npt.NDArray[np.bool_]]
NDArrayObject = npt.NDArray[np.object_]


def _filter_convert(
    ix_filter: NDArrayFilter | Iterable[int] | None, sample_arg: NDArrayFloat | NDArrayInt
) -> NDArrayFloat | None:
    """This function selects turbine indeces from the given array of turbine properties
    over the simulation's atmospheric conditions (wind directions / wind speeds).
    It converts the ix_filter to a standard format of `np.ndarray`s for filtering
    certain arguments.

    Args:
        ix_filter (NDArrayFilter | Iterable[int] | None): The indices, or truth
            array-like object to use for filtering. None implies that all indeces in the
            sample_arg should be selected.
        sample_arg (NDArrayFloat | NDArrayInt): Any argument that will be filtered, to be used for
            creating the shape. This should be of shape:
            (n wind directions, n wind speeds, n turbines)

    Returns:
        NDArrayFloat | None: Returns an array of a truth or index list if a list is
            passed, a truth array if ix_filter is None, or None if ix_filter is None
            and the `sample_arg` is a single value.
    """
    # Check that the ix_filter is either None or an Iterable. Otherwise,
    # there's nothing we can do with it.
    if not isinstance(ix_filter, Iterable) and ix_filter is not None:
        raise TypeError("Expected ix_filter to be an Iterable or None")

    # Check that the sample_arg is a Numpy array. If it isn't, we
    # can't get its shape.
    if not isinstance(sample_arg, np.ndarray):
        raise TypeError("Expected sample_arg to be a float or integer np.ndarray")

    # At this point, the arguments have this type:
    # ix_filter: Union[Iterable, None]
    # sample_arg: np.ndarray

    # Return all values in the turbine-dimension
    # if the index filter is None
    if ix_filter is None:
        return np.ones(sample_arg.shape[-1], dtype=bool)

    # Finally, we should have an index filter list of type Iterable,
    # so cast it to Numpy array and return
    return np.array(ix_filter)


def power(
    air_density: NDArrayFloat,
    velocities: NDArrayFloat,
    yaw_angle: NDArrayFloat,
    pP: NDArrayFloat,
    power_interp: NDArrayObject,
    ix_filter: NDArrayInt | Iterable[int] | None = None,
) -> NDArrayFloat:
    """Power produced by a turbine adjusted for yaw and tilt. Value
    given in Watts.

    Args:
        air_density (NDArrayFloat[wd, ws, turbines]): The air density value(s) at each turbine.
        velocities (NDArrayFloat[wd, ws, turbines, grid1, grid2]): The velocity field at a turbine.
        pP (NDArrayFloat[wd, ws, turbines]): The pP value(s) of the cosine exponent relating
            the yaw misalignment angle to power for each turbine.
        power_interp (NDArrayObject[wd, ws, turbines]): The power interpolation function
            for each turbine.
        ix_filter (NDArrayInt, optional): The boolean array, or
            integer indices to filter out before calculation. Defaults to None.

    Returns:
        NDArrayFloat: The power, in Watts, for each turbine after adjusting for yaw and tilt.
    """
    # TODO: Change the order of input arguments to be consistent with the other
    # utility functions - velocities first...
    # Update to power calculation which replaces the fixed pP exponent with
    # an exponent pW, that changes the effective wind speed input to the power
    # calculation, rather than scaling the power.  This better handles power
    # loss to yaw in above rated conditions
    #
    # based on the paper "Optimising yaw control at wind farm level" by
    # Ervin Bossanyi

    # TODO: check this - where is it?
    # P = 1/2 rho A V^3 Cp

    # NOTE: The below has a trivial performance hit for floats being passed (3.4% longer
    # on a meaningless test), but is actually faster when an array is passed through
    # That said, it adds overhead to convert the floats to 1-D arrays, so I don't
    # recommend just converting all values to arrays

    if isinstance(air_density, list):
        air_density = np.array(air_density)
    if isinstance(yaw_angle, list):
        yaw_angle = np.array(yaw_angle)

    ix_filter = _filter_convert(ix_filter, yaw_angle)
    if ix_filter is not None:
        air_density = air_density[:, :, ix_filter]
        velocities = velocities[:, :, ix_filter]
        yaw_angle = yaw_angle[:, :, ix_filter]

    # Compute the yaw effective velocity
    pW = pP / 3.0  # Convert from pP to w
    yaw_effective_velocity = average_velocity(velocities) * cosd(yaw_angle) ** pW

    n_wind_directions, n_wind_speeds, n_turbines, *_ = yaw_angle.shape
    p = np.zeros_like(yaw_effective_velocity)
    # TODO: convert to list comprehension
    for i in range(n_wind_directions):
        for j in range(n_wind_speeds):
            for k in range(n_turbines):
                interpolator = power_interp
                p[i, j, k] = interpolator(yaw_effective_velocity[i, j, k])

    return p * air_density


def Ct(
    velocities: NDArrayFloat,  # (wind directions, wind speeds, turbines, grid, grid)
    yaw_angle: NDArrayFloat,  # (wind directions, wind speeds, turbines)
    fCt: NDArrayObject,  # (wind directions, wind speeds, turbines)
    ix_filter: NDArrayFilter | Iterable[int] | None = None,
) -> NDArrayFloat:  # (wind directions, wind speeds, turbines)
    """Thrust coefficient of a turbine incorporating the yaw angle.
    The value is interpolated from the coefficient of thrust vs
    wind speed table using the rotor swept area average velocity.

    Args:
        velocities (NDArrayFloat[wd, ws, turbines, grid1, grid2]): The velocity field at a turbine.
        yaw_angle (NDArrayFloat[wd, ws, turbines]): The yaw angle for each turbine.
        fCt (NDArrayObject[wd, ws, turbines]): The thrust coefficient for each turbine.
        ix_filter (NDArrayFilter | Iterable[int] | None, optional): The boolean array, or
            integer indices as an iterable of array to filter out before calculation. Defaults to None.

    Returns:
        NDArrayFloat: Coefficient of thrust.
    """

    if isinstance(yaw_angle, list):
        yaw_angle = np.array(yaw_angle)

    ix_filter = _filter_convert(ix_filter, yaw_angle)
    if ix_filter is not None:
        velocities = velocities[:, :, ix_filter]
        yaw_angle = yaw_angle[:, :, ix_filter]

    n_wind_directions, n_wind_speeds, n_turbines, *_ = yaw_angle.shape
    average_velocities = average_velocity(velocities)
    thrust_coefficient = np.zeros_like(average_velocities)
    for i in range(n_wind_directions):
        for j in range(n_wind_speeds):
            for k in range(n_turbines):
                _fCt = fCt
                thrust_coefficient[i, j, k] = _fCt(average_velocities[i, j, k])

    effective_thrust = thrust_coefficient * cosd(yaw_angle)

    return effective_thrust


def axial_induction(
    velocities: NDArrayFloat,  # (wind directions, wind speeds, turbines, grid, grid)
    yaw_angle: NDArrayFloat,  # (wind directions, wind speeds, turbines)
    fCt: NDArrayObject,  # (wind directions, wind speeds, turbines)
    ix_filter: NDArrayFilter | Iterable[int] | None = None,
) -> NDArrayFloat:  # (wind directions, wind speeds, turbines)
    """Axial induction factor of the turbine incorporating
    the thrust coefficient and yaw angle.

    Args:
        velocities (NDArrayFloat): The velocity field at each turbine; should be shape:
            (number of turbines, ngrid, ngrid), or (ngrid, ngrid) for a single turbine.
        fCt (np.array): The thrust coefficient function for each
            turbine.
        ix_filter (NDArrayFilter | Iterable[int] | None, optional): The boolean array, or
            integer indices (as an aray or iterable) to filter out before calculation.
            Defaults to None.

    Returns:
        Union[float, NDArrayFloat]: [description]
    """

    if isinstance(yaw_angle, list):
        yaw_angle = np.array(yaw_angle)

    # Get Ct first before modifying any data
    thrust_coefficient = Ct(velocities, yaw_angle, fCt, ix_filter)

    # Then, process the input arguments as needed for this function
    ix_filter = _filter_convert(ix_filter, yaw_angle)
    if ix_filter is not None:
        yaw_angle = yaw_angle[:, :, ix_filter]

    return 0.5 / cosd(yaw_angle) * (1 - np.sqrt(1 - thrust_coefficient * cosd(yaw_angle)))


def average_velocity(velocities: NDArrayFloat, ix_filter: NDArrayFilter | Iterable[int] | None = None) -> NDArrayFloat:
    """This property calculates and returns the cube root of the
    mean cubed velocity in the turbine's rotor swept area (m/s).

    **Note:** The velocity is scaled to an effective velocity by the yaw.

    Args:
        velocities (NDArrayFloat): The velocity field at each turbine; should be shape:
            (number of turbines, ngrid, ngrid), or (ngrid, ngrid) for a single turbine.
        ix_filter (NDArrayFilter | Iterable[int] | None], optional): The boolean array, or
            integer indices (as an iterable or array) to filter out before calculation.
            Defaults to None.

    Returns:
        NDArrayFloat: The average velocity across the rotor(s).
    """
    # Remove all invalid numbers from interpolation
    # data = np.array(self.velocities)[~np.isnan(self.velocities)]

    # The input velocities are expected to be a 5 dimensional array with shape:
    # (# wind directions, # wind speeds, # turbines, grid resolution, grid resolution)

    if ix_filter is not None:
        velocities = velocities[:, :, ix_filter]
    axis = tuple([3 + i for i in range(velocities.ndim - 3)])
    return np.cbrt(np.mean(velocities ** 3, axis=axis))


@attr.s(frozen=True, auto_attribs=True)
class PowerThrustTable(FromDictMixin):
    """Helper class to convert the dictionary and list-based inputs to a object of arrays.

    Args:
        power (NDArrayFloat): The power produced at a given windspeed.
        thrust (NDArrayFloat): The thrust at a given windspeed.
        wind_speed (NDArrayFloat): Windspeed values, m/s.

    Raises:
        ValueError: Raised if the power, thrust, and wind_speed are not all 1-d array-like shapes.
        ValueError: Raised if power, thrust, and wind_speed don't have the same number of values.
    """

    # TODO: How to handle duplicate entries for a single wind speed?
    # This affects the interpolation in fCt / fCp
    power: NDArrayFloat = attr.ib(converter=attrs_array_converter)
    thrust: NDArrayFloat = attr.ib(converter=attrs_array_converter)
    wind_speed: NDArrayFloat = attr.ib(converter=attrs_array_converter)

    def __attrs_post_init__(self) -> None:
        inputs = (self.power, self.thrust, self.wind_speed)
        if any(el.ndim > 1 for el in inputs):
            raise ValueError("power, thrust, and wind_speed inputs must be 1-D.")
        if self.power.size != sum(el.size for el in inputs) / 3:
            raise ValueError("power, thrust, and wind_speed inputs must be the same size.")

        self.deduplicate_by_windspeed()

    def deduplicate_by_windspeed(self) -> None:
        """Identifies duplicated wind speed values in the power curve, and filters out
        all instances after the first, and then all corresponding `power` and `thurst`
        vaules.
        """
        _, duplicate_filter = np.unique(self.wind_speed, return_index=True)
        object.__setattr__(self, "power", self.power[duplicate_filter])
        object.__setattr__(self, "thrust", self.thrust[duplicate_filter])
        object.__setattr__(self, "wind_speed", self.wind_speed[duplicate_filter])


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
        power_thrust_table (PowerThrustTable): A dictionary containing the
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
    TSR: float = float_attrib()
    generator_efficiency: float = float_attrib()
    power_thrust_table: PowerThrustTable | dict[str, list[float]] = attr.ib(
        converter=PowerThrustTable.from_dict,
        kw_only=True,
    )
    # ngrid: float = float_attrib()  # TODO: goes here or on the Grid?
    # rloc: float = float_attrib()  # TODO: goes here or on the Grid?
    # use_points_on_perimeter: bool = bool_attrib()

    # Initialized in the post_init function
    rotor_radius: float = float_attrib(init=False)
    rotor_area: float = float_attrib(init=False)
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

    def _asdict(self) -> dict:
        """Creates a JSON and YAML friendly dictionary that can be save for future reloading.
        This dictionary will contain only `Python` types that can later be converted to their
        proper `Turbine` formats.

        Returns:
            dict: All key, vaue pais required for class recreation.
        """
        return attr.asdict(self, filter=attr_floris_filter, value_serializer=attr_serializer)

    def __attrs_post_init__(self) -> None:

        # Set the rotor_radius and rotor_area attributes
        self.rotor_radius = self.rotor_diameter / 2.0
        self.rotor_area = np.pi * self.rotor_radius ** 2

        # Post-init initialization for the power curve interpolation functions
        wind_speeds = self.power_thrust_table.wind_speed
        self.fCp_interp = interp1d(
            wind_speeds,
            self.power_thrust_table.power,
            fill_value="extrapolate",
        )

        # The fill_value arguments sets (upper, lower) bounds for any values
        # outside of the input range
        self.fCt_interp = interp1d(
            wind_speeds,
            self.power_thrust_table.thrust,
            fill_value=(0.0001, 0.9999),
            bounds_error=False,
        )

        inner_power = np.array([self._power_inner_function(ws) for ws in wind_speeds])
        self.power_interp = interp1d(wind_speeds, inner_power, fill_value="extrapolate")

    def _power_inner_function(self, velocities: NDArrayFloat | list[float]) -> NDArrayFloat:
        """
        This method calculates the power for an array of yaw effective wind
        speeds without the air density and turbulence correction parameters.
        This is used to initialize the power interpolation method used to
        compute turbine power.
        """
        return 0.5 * self.rotor_area * self.fCp(velocities) * self.generator_efficiency * velocities ** 3

    def fCp(self, sample_wind_speeds: float | NDArrayFloat) -> float | NDArrayFloat:
        """Calculates the coefficient of power at a given wind speed.

        Returns:
            float | NDArrayFloat: The coefficient of power for a given wind speed.
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

        # Return the data type that matches the input size, but figure out the dimensionality
        if is_single:
            return _cp[0]
        return _cp

    def fCt(self, at_wind_speed: float | NDArrayFloat) -> float | NDArrayFloat:
        """
        Given an array of wind speeds, this function
        returns an array of the interpolated thrust coefficients
        from the power / thrust table used to define the Turbine.
        The values are bound by the range of the input values.
        Any requested wind speeds outside of the range of
        input wind speeds are assigned Ct of 0.0001 or 0.9999.

        Args:
            at_wind_speed (float | NDArrayFloat): Wind speeds to find Ct

        Returns:
            float | NDArrayFloat: The interpolated Ct values
        """
        return self.fCt_interp(at_wind_speed)

    @rotor_diameter.validator
    def reset_rotor_diameter_dependencies(self, instance: attr.Attribute, value: float) -> None:
        """Resets the `rotor_radius` and `rotor_area` attributes."""
        # Temporarily turn off validators to avoid infinite recursion
        attr.set_run_validators(False)

        # Reset the values
        self.rotor_radius = value / 2.0
        self.rotor_area = np.pi * self.rotor_radius ** 2.0

        # Turn validators back on
        attr.set_run_validators(True)

    @rotor_radius.validator
    def reset_rotor_radius(self, instance: attr.Attribute, value: float) -> None:
        """
        Resets the `rotor_diameter` value to trigger the recalculation of
        `rotor_diameter`, `rotor_radius` and `rotor_area`.
        """
        self.rotor_diameter = value * 2.0

    @rotor_area.validator
    def reset_rotor_area(self, instance: attr.Attribute, value: float) -> None:
        """
        Resets the `rotor_radius` value to trigger the recalculation of
        `rotor_diameter`, `rotor_radius` and `rotor_area`.
        """
        self.rotor_radius = math.sqrt(value / np.pi)
