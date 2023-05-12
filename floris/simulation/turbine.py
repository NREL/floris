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

import copy
from collections.abc import Iterable
from typing import Any

import attrs
import numpy as np
from attrs import define, field
from scipy.interpolate import interp1d

from floris.simulation import BaseClass
from floris.type_dec import (
    floris_array_converter,
    FromDictMixin,
    NDArrayBool,
    NDArrayFilter,
    NDArrayFloat,
    NDArrayInt,
    NDArrayObject,
)
from floris.utilities import cosd


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


def _rotor_velocity_yaw_correction(
    pP: float,
    yaw_angle: NDArrayFloat,
    rotor_effective_velocities: NDArrayFloat,
) -> NDArrayFloat:
    # Compute the rotor effective velocity adjusting for yaw settings
    pW = pP / 3.0  # Convert from pP to w
    rotor_effective_velocities = rotor_effective_velocities * cosd(yaw_angle) ** pW

    return rotor_effective_velocities


def _rotor_velocity_tilt_correction(
    turbine_type_map: NDArrayObject,
    tilt_angle: NDArrayFloat,
    ref_tilt_cp_ct: NDArrayFloat,
    pT: float,
    tilt_interp: NDArrayObject,
    correct_cp_ct_for_tilt: NDArrayBool,
    rotor_effective_velocities: NDArrayFloat,
) -> NDArrayFloat:
    # Compute the tilt, if using floating turbines
    old_tilt_angle = copy.deepcopy(tilt_angle)
    tilt_angle = compute_tilt_angles_for_floating_turbines(
        turbine_type_map,
        tilt_angle,
        tilt_interp,
        rotor_effective_velocities,
    )
    # Only update tilt angle if requested (if the tilt isn't accounted for in the Cp curve)
    tilt_angle = np.where(correct_cp_ct_for_tilt, tilt_angle, old_tilt_angle)

    # Compute the rotor effective velocity adjusting for tilt
    rotor_effective_velocities = (
        rotor_effective_velocities
        * cosd(tilt_angle - ref_tilt_cp_ct) ** (pT / 3.0)
    )
    return rotor_effective_velocities


def compute_tilt_angles_for_floating_turbines(
    turbine_type_map: NDArrayObject,
    tilt_angle: NDArrayFloat,
    tilt_interp: NDArrayObject,
    rotor_effective_velocities: NDArrayFloat,
) -> NDArrayFloat:
    # Loop over each turbine type given to get tilt angles for all turbines
    tilt_angles = np.zeros(np.shape(rotor_effective_velocities))
    tilt_interp = dict(tilt_interp)
    turb_types = np.unique(turbine_type_map)
    for turb_type in turb_types:
        # If no tilt interpolation is specified, assume no modification to tilt
        if tilt_interp[turb_type] is None:
            # TODO should this be break? Should it be continue? Do we want to support mixed
            # fixed-bottom and floating? Or non-tilting floating?
            pass
        # Using a masked array, apply the tilt angle for all turbines of the current
        # type to the main tilt angle array
        else:
            tilt_angles += (
                tilt_interp[turb_type](rotor_effective_velocities)
                * np.array(turbine_type_map == turb_type)
            )

    # TODO: Not sure if this is the best way to do this? Basically replaces the initialized
    # tilt_angles if there are non-zero tilt angles calculated above (meaning that the turbine
    # definition contained  a wind_speed/tilt table definition)
    if not tilt_angles.all() == 0.:
        tilt_angle = tilt_angles

    return tilt_angle


def rotor_effective_velocity(
    air_density: float,
    ref_density_cp_ct: float,
    velocities: NDArrayFloat,
    yaw_angle: NDArrayFloat,
    tilt_angle: NDArrayFloat,
    ref_tilt_cp_ct: NDArrayFloat,
    pP: float,
    pT: float,
    tilt_interp: NDArrayObject,
    correct_cp_ct_for_tilt: NDArrayBool,
    turbine_type_map: NDArrayObject,
    ix_filter: NDArrayInt | Iterable[int] | None = None,
    average_method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None
) -> NDArrayFloat:

    if isinstance(yaw_angle, list):
        yaw_angle = np.array(yaw_angle)
    if isinstance(tilt_angle, list):
        tilt_angle = np.array(tilt_angle)

    # Down-select inputs if ix_filter is given
    if ix_filter is not None:
        ix_filter = _filter_convert(ix_filter, yaw_angle)
        velocities = velocities[:, :, ix_filter]
        yaw_angle = yaw_angle[:, :, ix_filter]
        tilt_angle = tilt_angle[:, :, ix_filter]
        ref_tilt_cp_ct = ref_tilt_cp_ct[:, :, ix_filter]
        pP = pP[:, :, ix_filter]
        pT = pT[:, :, ix_filter]
        turbine_type_map = turbine_type_map[:, :, ix_filter]

    # Compute the rotor effective velocity adjusting for air density
    # TODO: This correction is currently split across two functions: this one and `power`, where in
    # `power` the returned power is multiplied by the reference air density
    average_velocities = average_velocity(
        velocities,
        method=average_method,
        cubature_weights=cubature_weights
    )
    rotor_effective_velocities = (air_density/ref_density_cp_ct)**(1/3) * average_velocities

    # Compute the rotor effective velocity adjusting for yaw settings
    rotor_effective_velocities = _rotor_velocity_yaw_correction(
        pP, yaw_angle, rotor_effective_velocities
    )

    # Compute the tilt, if using floating turbines
    rotor_effective_velocities = _rotor_velocity_tilt_correction(
        turbine_type_map,
        tilt_angle,
        ref_tilt_cp_ct,
        pT,
        tilt_interp,
        correct_cp_ct_for_tilt,
        rotor_effective_velocities,
    )

    return rotor_effective_velocities


def power(
    ref_density_cp_ct: float,
    rotor_effective_velocities: NDArrayFloat,
    power_interp: NDArrayObject,
    turbine_type_map: NDArrayObject,
    ix_filter: NDArrayInt | Iterable[int] | None = None,
) -> NDArrayFloat:
    """Power produced by a turbine adjusted for yaw and tilt. Value
    given in Watts.

    Args:
        ref_density_cp_cts (NDArrayFloat[wd, ws, turbines]): The reference density for each turbine
        rotor_effective_velocities (NDArrayFloat[wd, ws, turbines, grid1, grid2]): The rotor
            effective velocities at a turbine.
        power_interp (NDArrayObject[wd, ws, turbines]): The power interpolation function
            for each turbine.
        turbine_type_map: (NDArrayObject[wd, ws, turbines]): The Turbine type definition for
            each turbine.
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

    # Down-select inputs if ix_filter is given
    if ix_filter is not None:
        ix_filter = _filter_convert(ix_filter, rotor_effective_velocities)
        rotor_effective_velocities = rotor_effective_velocities[:, :, ix_filter]
        turbine_type_map = turbine_type_map[:, :, ix_filter]

    # Loop over each turbine type given to get power for all turbines
    p = np.zeros(np.shape(rotor_effective_velocities))
    turb_types = np.unique(turbine_type_map)
    for turb_type in turb_types:
        # Using a masked array, apply the thrust coefficient for all turbines of the current
        # type to the main thrust coefficient array
        p += (
            power_interp[turb_type](rotor_effective_velocities)
            * np.array(turbine_type_map == turb_type)
        )

    return p * ref_density_cp_ct


def Ct(
    velocities: NDArrayFloat,
    yaw_angle: NDArrayFloat,
    tilt_angle: NDArrayFloat,
    ref_tilt_cp_ct: NDArrayFloat,
    fCt: dict,
    tilt_interp: NDArrayObject,
    correct_cp_ct_for_tilt: NDArrayBool,
    turbine_type_map: NDArrayObject,
    ix_filter: NDArrayFilter | Iterable[int] | None = None,
    average_method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None
) -> NDArrayFloat:

    """Thrust coefficient of a turbine incorporating the yaw angle.
    The value is interpolated from the coefficient of thrust vs
    wind speed table using the rotor swept area average velocity.

    Args:
        velocities (NDArrayFloat[wd, ws, turbines, grid1, grid2]): The velocity field at
            a turbine.
        yaw_angle (NDArrayFloat[wd, ws, turbines]): The yaw angle for each turbine.
        tilt_angle (NDArrayFloat[wd, ws, turbines]): The tilt angle for each turbine.
        ref_tilt_cp_ct (NDArrayFloat[wd, ws, turbines]): The reference tilt angle for each turbine
            that the Cp/Ct tables are defined at.
        fCt (dict): The thrust coefficient interpolation functions for each turbine. Keys are
            the turbine type string and values are the interpolation functions.
        tilt_interp (Iterable[tuple]): The tilt interpolation functions for each
            turbine.
        correct_cp_ct_for_tilt (NDArrayBool[wd, ws, turbines]): Boolean for determining if the
            turbines Cp and Ct should be corrected for tilt.
        turbine_type_map: (NDArrayObject[wd, ws, turbines]): The Turbine type definition
            for each turbine.
        ix_filter (NDArrayFilter | Iterable[int] | None, optional): The boolean array, or
            integer indices as an iterable of array to filter out before calculation.
            Defaults to None.

    Returns:
        NDArrayFloat: Coefficient of thrust for each requested turbine.
    """

    if isinstance(yaw_angle, list):
        yaw_angle = np.array(yaw_angle)

    if isinstance(tilt_angle, list):
        tilt_angle = np.array(tilt_angle)

    # Down-select inputs if ix_filter is given
    if ix_filter is not None:
        ix_filter = _filter_convert(ix_filter, yaw_angle)
        velocities = velocities[:, :, ix_filter]
        yaw_angle = yaw_angle[:, :, ix_filter]
        tilt_angle = tilt_angle[:, :, ix_filter]
        ref_tilt_cp_ct = ref_tilt_cp_ct[:, :, ix_filter]
        turbine_type_map = turbine_type_map[:, :, ix_filter]
        correct_cp_ct_for_tilt = correct_cp_ct_for_tilt[:, :, ix_filter]

    average_velocities = average_velocity(
        velocities,
        method=average_method,
        cubature_weights=cubature_weights
    )

    # Compute the tilt, if using floating turbines
    old_tilt_angle = copy.deepcopy(tilt_angle)
    tilt_angle = compute_tilt_angles_for_floating_turbines(
        turbine_type_map,
        tilt_angle,
        tilt_interp,
        average_velocities,
    )
    # Only update tilt angle if requested (if the tilt isn't accounted for in the Ct curve)
    tilt_angle = np.where(correct_cp_ct_for_tilt, tilt_angle, old_tilt_angle)

    # Loop over each turbine type given to get thrust coefficient for all turbines
    thrust_coefficient = np.zeros(np.shape(average_velocities))
    turb_types = np.unique(turbine_type_map)
    for turb_type in turb_types:
        # Using a masked array, apply the thrust coefficient for all turbines of the current
        # type to the main thrust coefficient array
        thrust_coefficient += (
            fCt[turb_type](average_velocities)
            * np.array(turbine_type_map == turb_type)
        )
    thrust_coefficient = np.clip(thrust_coefficient, 0.0001, 0.9999)
    effective_thrust = thrust_coefficient * cosd(yaw_angle) * cosd(tilt_angle - ref_tilt_cp_ct)
    return effective_thrust


def axial_induction(
    velocities: NDArrayFloat,  # (wind directions, wind speeds, turbines, grid, grid)
    yaw_angle: NDArrayFloat,  # (wind directions, wind speeds, turbines)
    tilt_angle: NDArrayFloat,  # (wind directions, wind speeds, turbines)
    ref_tilt_cp_ct: NDArrayFloat,
    fCt: dict,  # (turbines)
    tilt_interp: NDArrayObject,  # (turbines)
    correct_cp_ct_for_tilt: NDArrayBool, # (wind directions, wind speeds, turbines)
    turbine_type_map: NDArrayObject, # (wind directions, 1, turbines)
    ix_filter: NDArrayFilter | Iterable[int] | None = None,
    average_method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None
) -> NDArrayFloat:
    """Axial induction factor of the turbine incorporating
    the thrust coefficient and yaw angle.

    Args:
        velocities (NDArrayFloat): The velocity field at each turbine; should be shape:
            (number of turbines, ngrid, ngrid), or (ngrid, ngrid) for a single turbine.
        yaw_angle (NDArrayFloat[wd, ws, turbines]): The yaw angle for each turbine.
        tilt_angle (NDArrayFloat[wd, ws, turbines]): The tilt angle for each turbine.
        ref_tilt_cp_ct (NDArrayFloat[wd, ws, turbines]): The reference tilt angle for each turbine
            that the Cp/Ct tables are defined at.
        fCt (dict): The thrust coefficient interpolation functions for each turbine. Keys are
            the turbine type string and values are the interpolation functions.
        tilt_interp (Iterable[tuple]): The tilt interpolation functions for each
            turbine.
        correct_cp_ct_for_tilt (NDArrayBool[wd, ws, turbines]): Boolean for determining if the
            turbines Cp and Ct should be corrected for tilt.
        turbine_type_map: (NDArrayObject[wd, ws, turbines]): The Turbine type definition
            for each turbine.
        ix_filter (NDArrayFilter | Iterable[int] | None, optional): The boolean array, or
            integer indices (as an aray or iterable) to filter out before calculation.
            Defaults to None.

    Returns:
        Union[float, NDArrayFloat]: [description]
    """

    if isinstance(yaw_angle, list):
        yaw_angle = np.array(yaw_angle)

    # TODO: Should the tilt_angle used for the return calculation be modified the same as the
    # tilt_angle in Ct, if the user has supplied a tilt/wind_speed table?
    if isinstance(tilt_angle, list):
        tilt_angle = np.array(tilt_angle)

    # Get Ct first before modifying any data
    thrust_coefficient = Ct(
        velocities,
        yaw_angle,
        tilt_angle,
        ref_tilt_cp_ct,
        fCt,
        tilt_interp,
        correct_cp_ct_for_tilt,
        turbine_type_map,
        ix_filter,
        average_method,
        cubature_weights
    )

    # Then, process the input arguments as needed for this function
    ix_filter = _filter_convert(ix_filter, yaw_angle)
    if ix_filter is not None:
        yaw_angle = yaw_angle[:, :, ix_filter]
        tilt_angle = tilt_angle[:, :, ix_filter]
        ref_tilt_cp_ct = ref_tilt_cp_ct[:, :, ix_filter]

    return (
        0.5
        / (cosd(yaw_angle)
        * cosd(tilt_angle - ref_tilt_cp_ct))
        * (
            1 - np.sqrt(
                1 - thrust_coefficient * cosd(yaw_angle) * cosd(tilt_angle - ref_tilt_cp_ct)
            )
        )
    )


def simple_mean(array, axis=0):
    return np.mean(array, axis=axis)

def cubic_mean(array, axis=0):
    return np.cbrt(np.mean(array ** 3.0, axis=axis))

def simple_cubature(array, cubature_weights, axis=0):
    weights = cubature_weights.flatten()
    weights = weights * len(weights) / np.sum(weights)
    product = (array * weights[None, None, None, :, None])
    return simple_mean(product, axis)

def cubic_cubature(array, cubature_weights, axis=0):
    weights = cubature_weights.flatten()
    weights = weights * len(weights) / np.sum(weights)
    return np.cbrt(np.mean((array**3.0 * weights[None, None, None, :, None]), axis=axis))

def average_velocity(
    velocities: NDArrayFloat,
    ix_filter: NDArrayFilter | Iterable[int] | None = None,
    method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None
) -> NDArrayFloat:
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

    # The input velocities are expected to be a 5 dimensional array with shape:
    # (# wind directions, # wind speeds, # turbines, grid resolution, grid resolution)

    if ix_filter is not None:
        velocities = velocities[:, :, ix_filter]

    axis = tuple([3 + i for i in range(velocities.ndim - 3)])
    if method == "simple-mean":
        return simple_mean(velocities, axis)

    elif method == "cubic-mean":
        return cubic_mean(velocities, axis)

    elif method == "simple-cubature":
        if cubature_weights is None:
            raise ValueError("cubature_weights is required for 'simple-cubature' method.")
        return simple_cubature(velocities, cubature_weights, axis)

    elif method == "cubic-cubature":
        if cubature_weights is None:
            raise ValueError("cubature_weights is required for 'cubic-cubature' method.")
        return cubic_cubature(velocities, cubature_weights, axis)

    else:
        raise ValueError("Incorrect method given.")

@define
class PowerThrustTable(FromDictMixin):
    """Helper class to convert the dictionary and list-based inputs to a object of arrays.

    Args:
        power (NDArrayFloat): The power produced at a given wind speed.
        thrust (NDArrayFloat): The thrust at a given wind speed.
        wind_speed (NDArrayFloat): Wind speed values, m/s.

    Raises:
        ValueError: Raised if the power, thrust, and wind_speed are not all 1-d array-like shapes.
        ValueError: Raised if power, thrust, and wind_speed don't have the same number of values.
    """
    power: NDArrayFloat = field(converter=floris_array_converter)
    thrust: NDArrayFloat = field(converter=floris_array_converter)
    wind_speed: NDArrayFloat = field(converter=floris_array_converter)

    def __attrs_post_init__(self) -> None:
        # Validate the power, thrust, and wind speed inputs.

        inputs = (self.power, self.thrust, self.wind_speed)

        if any(el.ndim > 1 for el in inputs):
            raise ValueError("power, thrust, and wind_speed inputs must be 1-D.")

        if len( {self.power.size, self.thrust.size, self.wind_speed.size} ) > 1:
            raise ValueError("power, thrust, and wind_speed tables must be the same size.")

        # Remove any duplicate wind speed entries
        _, duplicate_filter = np.unique(self.wind_speed, return_index=True)
        self.power = self.power[duplicate_filter]
        self.thrust = self.thrust[duplicate_filter]
        self.wind_speed = self.wind_speed[duplicate_filter]


@define
class TiltTable(FromDictMixin):
    """Helper class to convert the dictionary and list-based inputs to a object of arrays.

    Args:
        tilt (NDArrayFloat): The tilt angle at a given wind speed.
        wind_speeds (NDArrayFloat): Wind speed values, m/s.

    Raises:
        ValueError: Raised if tilt and wind_speeds are not all 1-d array-like shapes.
        ValueError: Raised if tilt and wind_speeds don't have the same number of values.
    """
    tilt: NDArrayFloat = field(converter=floris_array_converter)
    wind_speeds: NDArrayFloat = field(converter=floris_array_converter)

    def __attrs_post_init__(self) -> None:
        # Validate the power, thrust, and wind speed inputs.

        inputs = (self.tilt, self.wind_speeds)

        if any(el.ndim > 1 for el in inputs):
            raise ValueError("tilt and wind_speed inputs must be 1-D.")

        if len({self.tilt.size, self.wind_speeds.size}) > 1:
            raise ValueError("tilt and wind_speed tables must be the same size.")

        # Remove any duplicate wind speed entries
        _, duplicate_filter = np.unique(self.wind_speeds, return_index=True)
        self.tilt = self.tilt[duplicate_filter]
        self.wind_speeds = self.wind_speeds[duplicate_filter]


@define
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
        ref_density_cp_ct (:py:obj: float): The density at which the provided
            cp and ct is defined
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

    turbine_type: str = field()
    rotor_diameter: float = field()
    hub_height: float = field()
    pP: float = field()
    pT: float = field()
    TSR: float = field()
    generator_efficiency: float = field()
    ref_density_cp_ct: float = field()
    ref_tilt_cp_ct: float = field()
    power_thrust_table: PowerThrustTable = field(converter=PowerThrustTable.from_dict)
    floating_tilt_table = field(default=None)
    floating_correct_cp_ct_for_tilt = field(default=None)

    # rloc: float = float_attrib()  # TODO: goes here or on the Grid?
    # use_points_on_perimeter: bool = bool_attrib()

    # Initialized in the post_init function
    rotor_radius: float = field(init=False)
    rotor_area: float = field(init=False)
    fCp_interp: interp1d = field(init=False)
    fCt_interp: interp1d = field(init=False)
    power_interp: interp1d = field(init=False)
    tilt_interp: interp1d = field(init=False)


    # For the following parameters, use default values if not user-specified
    # self.rloc = float(input_dictionary["rloc"]) if "rloc" in input_dictionary else 0.5
    # if "use_points_on_perimeter" in input_dictionary:
    #     self.use_points_on_perimeter = bool(input_dictionary["use_points_on_perimeter"])
    # else:
    #     self.use_points_on_perimeter = False

    def __attrs_post_init__(self) -> None:

        # Post-init initialization for the power curve interpolation functions
        wind_speeds = self.power_thrust_table.wind_speed
        self.fCp_interp = interp1d(
            wind_speeds,
            self.power_thrust_table.power,
            fill_value=(0.0, 1.0),
            bounds_error=False,
        )
        inner_power = (
            0.5 * self.rotor_area
            * self.fCp_interp(wind_speeds)
            * self.generator_efficiency
            * wind_speeds ** 3
        )
        self.power_interp = interp1d(
            wind_speeds,
            inner_power
        )

        """
        Given an array of wind speeds, this function returns an array of the
        interpolated thrust coefficients from the power / thrust table used
        to define the Turbine. The values are bound by the range of the input
        values. Any requested wind speeds outside of the range of input wind
        speeds are assigned Ct of 0.0001 or 0.9999.

        The fill_value arguments sets (upper, lower) bounds for any values
        outside of the input range.
        """
        self.fCt_interp = interp1d(
            wind_speeds,
            self.power_thrust_table.thrust,
            fill_value=(0.0001, 0.9999),
            bounds_error=False,
        )

        # If defined, create a tilt interpolation function for floating turbines.
        # fill_value currently set to apply the min or max tilt angles if outside
        # of the interpolation range.
        if self.floating_tilt_table is not None:
            self.floating_tilt_table = TiltTable.from_dict(self.floating_tilt_table)
            self.fTilt_interp = interp1d(
                self.floating_tilt_table.wind_speeds,
                self.floating_tilt_table.tilt,
                fill_value=(0.0, self.floating_tilt_table.tilt[-1]),
                bounds_error=False,
            )
            self.tilt_interp = self.fTilt_interp
            self.correct_cp_ct_for_tilt = self.floating_correct_cp_ct_for_tilt
        else:
            self.fTilt_interp = None
            self.tilt_interp = None
            self.correct_cp_ct_for_tilt = False

    @rotor_diameter.validator
    def reset_rotor_diameter_dependencies(self, instance: attrs.Attribute, value: float) -> None:
        """Resets the `rotor_radius` and `rotor_area` attributes."""
        # Temporarily turn off validators to avoid infinite recursion
        with attrs.validators.disabled():
            # Reset the values
            self.rotor_radius = value / 2.0
            self.rotor_area = np.pi * self.rotor_radius ** 2.0

    @rotor_radius.validator
    def reset_rotor_radius(self, instance: attrs.Attribute, value: float) -> None:
        """
        Resets the `rotor_diameter` value to trigger the recalculation of
        `rotor_diameter`, `rotor_radius` and `rotor_area`.
        """
        self.rotor_diameter = value * 2.0

    @rotor_area.validator
    def reset_rotor_area(self, instance: attrs.Attribute, value: float) -> None:
        """
        Resets the `rotor_radius` value to trigger the recalculation of
        `rotor_diameter`, `rotor_radius` and `rotor_area`.
        """
        self.rotor_radius = (value / np.pi) ** 0.5

    @floating_tilt_table.validator
    def check_floating_tilt_table(self, instance: attrs.Attribute, value: Any) -> None:
        """
        Check that if the tile/wind_speed table is defined, that the tilt and
        wind_speed arrays are the same length so that the interpolation will work.
        """
        if self.floating_tilt_table is not None:
            if (
                len(self.floating_tilt_table["tilt"])
                != len(self.floating_tilt_table["wind_speeds"])
            ):
                raise ValueError(
                    "tilt and wind_speeds must be the same length for the interpolation to work."
                )

    @floating_correct_cp_ct_for_tilt.validator
    def check_for_cp_ct_correct_flag_if_floating(
        self,
        instance: attrs.Attribute,
        value: Any
    ) -> None:
        """
        Check that the boolean flag exists for correcting Cp/Ct for tilt
        if a tile/wind_speed table is also defined.
        """
        if self.floating_tilt_table is not None:
            if self.floating_correct_cp_ct_for_tilt is None:
                raise ValueError(
                    "If a floating tilt/wind_speed table is defined, the boolean flag"
                    "floating_correct_cp_ct_for_tilt must also be defined."
                )
