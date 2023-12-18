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

import attrs
import numpy as np
from attrs import define, field
from scipy.interpolate import interp1d

from floris.simulation import BaseClass
from floris.type_dec import (
    floris_numeric_dict_converter,
    NDArrayBool,
    NDArrayFilter,
    NDArrayFloat,
    NDArrayInt,
    NDArrayObject,
)
from floris.utilities import cosd


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
    relative_tilt = tilt_angle - ref_tilt_cp_ct
    rotor_effective_velocities = rotor_effective_velocities * cosd(relative_tilt) ** (pT / 3.0)
    return rotor_effective_velocities


def compute_tilt_angles_for_floating_turbines(
    turbine_type_map: NDArrayObject,
    tilt_angle: NDArrayFloat,
    tilt_interp: dict[str, interp1d],
    rotor_effective_velocities: NDArrayFloat,
) -> NDArrayFloat:
    # Loop over each turbine type given to get tilt angles for all turbines
    tilt_angles = np.zeros(np.shape(rotor_effective_velocities))
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
                * (turbine_type_map == turb_type)
            )

    # TODO: Not sure if this is the best way to do this? Basically replaces the initialized
    # tilt_angles if there are non-zero tilt angles calculated above (meaning that the turbine
    # definition contained  a wind_speed/tilt table definition)
    if not tilt_angles.all() == 0.0:
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
        velocities = velocities[:, ix_filter]
        yaw_angle = yaw_angle[:, ix_filter]
        tilt_angle = tilt_angle[:, ix_filter]
        ref_tilt_cp_ct = ref_tilt_cp_ct[:, ix_filter]
        pP = pP[:, ix_filter]
        pT = pT[:, ix_filter]
        turbine_type_map = turbine_type_map[:, ix_filter]

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
    power_interp: dict[str, interp1d],
    turbine_type_map: NDArrayObject,
    ix_filter: NDArrayInt | Iterable[int] | None = None,
) -> NDArrayFloat:
    """Power produced by a turbine adjusted for yaw and tilt. Value
    given in Watts.

    Args:
        ref_density_cp_cts (NDArrayFloat[wd, ws, turbines]): The reference density for each turbine
        rotor_effective_velocities (NDArrayFloat[wd, ws, turbines]): The rotor
            effective velocities at a turbine.
        power_interp (dict[str, interp1d]): A dictionary of power interpolation functions for
            each turbine type.
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
        rotor_effective_velocities = rotor_effective_velocities[:, ix_filter]
        turbine_type_map = turbine_type_map[:, ix_filter]

    # Loop over each turbine type given to get power for all turbines
    p = np.zeros(np.shape(rotor_effective_velocities))
    turb_types = np.unique(turbine_type_map)
    for turb_type in turb_types:
        # Using a masked array, apply the thrust coefficient for all turbines of the current
        # type to the main thrust coefficient array
        p += power_interp[turb_type](rotor_effective_velocities) * (turbine_type_map == turb_type)

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
        velocities (NDArrayFloat[findex, turbines, grid1, grid2]): The velocity field at
            a turbine.
        yaw_angle (NDArrayFloat[findex, turbines]): The yaw angle for each turbine.
        tilt_angle (NDArrayFloat[findex, turbines]): The tilt angle for each turbine.
        ref_tilt_cp_ct (NDArrayFloat[findex, turbines]): The reference tilt angle for each turbine
            that the Cp/Ct tables are defined at.
        fCt (dict): The thrust coefficient interpolation functions for each turbine. Keys are
            the turbine type string and values are the interpolation functions.
        tilt_interp (Iterable[tuple]): The tilt interpolation functions for each
            turbine.
        correct_cp_ct_for_tilt (NDArrayBool[findex, turbines]): Boolean for determining if the
            turbines Cp and Ct should be corrected for tilt.
        turbine_type_map: (NDArrayObject[findex, turbines]): The Turbine type definition
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
        velocities = velocities[:, ix_filter]
        yaw_angle = yaw_angle[:, ix_filter]
        tilt_angle = tilt_angle[:, ix_filter]
        ref_tilt_cp_ct = ref_tilt_cp_ct[:, ix_filter]
        turbine_type_map = turbine_type_map[:, ix_filter]
        correct_cp_ct_for_tilt = correct_cp_ct_for_tilt[:, ix_filter]

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
            * (turbine_type_map == turb_type)
        )
    thrust_coefficient = np.clip(thrust_coefficient, 0.0001, 0.9999)
    effective_thrust = thrust_coefficient * cosd(yaw_angle) * cosd(tilt_angle - ref_tilt_cp_ct)
    return effective_thrust


def axial_induction(
    velocities: NDArrayFloat,  # (findex, turbines, grid, grid)
    yaw_angle: NDArrayFloat,  # (findex, turbines)
    tilt_angle: NDArrayFloat,  # (findex, turbines)
    ref_tilt_cp_ct: NDArrayFloat,
    fCt: dict,  # (turbines)
    tilt_interp: NDArrayObject,  # (turbines)
    correct_cp_ct_for_tilt: NDArrayBool, # (findex, turbines)
    turbine_type_map: NDArrayObject, # (findex, turbines)
    ix_filter: NDArrayFilter | Iterable[int] | None = None,
    average_method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None
) -> NDArrayFloat:
    """Axial induction factor of the turbine incorporating
    the thrust coefficient and yaw angle.

    Args:
        velocities (NDArrayFloat): The velocity field at each turbine; should be shape:
            (number of turbines, ngrid, ngrid), or (ngrid, ngrid) for a single turbine.
        yaw_angle (NDArrayFloat[findex, turbines]): The yaw angle for each turbine.
        tilt_angle (NDArrayFloat[findex, turbines]): The tilt angle for each turbine.
        ref_tilt_cp_ct (NDArrayFloat[findex, turbines]): The reference tilt angle for each turbine
            that the Cp/Ct tables are defined at.
        fCt (dict): The thrust coefficient interpolation functions for each turbine. Keys are
            the turbine type string and values are the interpolation functions.
        tilt_interp (Iterable[tuple]): The tilt interpolation functions for each
            turbine.
        correct_cp_ct_for_tilt (NDArrayBool[findex, turbines]): Boolean for determining if the
            turbines Cp and Ct should be corrected for tilt.
        turbine_type_map: (NDArrayObject[findex, turbines]): The Turbine type definition
            for each turbine.
        ix_filter (NDArrayFilter | Iterable[int] | None, optional): The boolean array, or
            integer indices (as an array or iterable) to filter out before calculation.
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
    if ix_filter is not None:
        yaw_angle = yaw_angle[:, ix_filter]
        tilt_angle = tilt_angle[:, ix_filter]
        ref_tilt_cp_ct = ref_tilt_cp_ct[:, ix_filter]

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
    product = (array * weights[None, None, :, None])
    return simple_mean(product, axis)

def cubic_cubature(array, cubature_weights, axis=0):
    weights = cubature_weights.flatten()
    weights = weights * len(weights) / np.sum(weights)
    return np.cbrt(np.mean((array**3.0 * weights[None, None, :, None]), axis=axis))

def average_velocity(
    velocities: NDArrayFloat,
    ix_filter: NDArrayFilter | Iterable[int] | None = None,
    method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None
) -> NDArrayFloat:
    """This property calculates and returns the average of the velocity field
    in turbine's rotor swept area. The average is calculated using the
    user-specified method. This is a vectorized function, so it can be used
    to calculate the average velocity for multiple turbines at once or
    a single turbine.

    **Note:** The velocity is scaled to an effective velocity by the yaw.

    Args:
        velocities (NDArrayFloat): The velocity field at each turbine; should be shape:
            (number of turbines, ngrid, ngrid), or (ngrid, ngrid) for a single turbine.
        ix_filter (NDArrayFilter | Iterable[int] | None], optional): The boolean array, or
            integer indices (as an iterable or array) to filter out before calculation.
            Defaults to None.
        method (str, optional): The method to use for averaging. Options are:
            - "simple-mean": The simple mean of the velocities
            - "cubic-mean": The cubic mean of the velocities
            - "simple-cubature": A cubature integration of the velocities
            - "cubic-cubature": A cubature integration of the cube of the velocities
            Defaults to "cubic-mean".
        cubature_weights (NDArrayFloat, optional): The cubature weights to use for the
            cubature integration methods. Defaults to None.

    Returns:
        NDArrayFloat: The average velocity across the rotor(s).
    """

    # The input velocities are expected to be a 5 dimensional array with shape:
    # (# findex, # turbines, grid resolution, grid resolution)

    if ix_filter is not None:
        velocities = velocities[:, ix_filter]

    axis = tuple([2 + i for i in range(velocities.ndim - 2)])
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
class Turbine(BaseClass):
    """
    A class containing the parameters and infrastructure to model a wind turbine's performance
    for a particular atmospheric condition.

    Args:
        turbine_type (str): An identifier for this type of turbine such as "NREL_5MW".
        rotor_diameter (float): The rotor diameter in meters.
        hub_height (float): The hub height in meters.
        pP (float): The cosine exponent relating the yaw misalignment angle to turbine power.
        pT (float): The cosine exponent relating the rotor tilt angle to turbine power.
        TSR (float): The Tip Speed Ratio of the turbine.
        generator_efficiency (float): The efficiency of the generator used to scale
            power production.
        ref_density_cp_ct (float): The density at which the provided Cp and Ct curves are defined.
        ref_tilt_cp_ct (float): The implicit tilt of the turbine for which the Cp and Ct
            curves are defined. This is typically the nacelle tilt.
        power_thrust_table (dict[str, float]): Contains power coefficient and thrust coefficient
            values at a series of wind speeds to define the turbine performance.
            The dictionary must have the following three keys with equal length values:
                {
                    "wind_speeds": List[float],
                    "power": List[float],
                    "thrust": List[float],
                }
        correct_cp_ct_for_tilt (bool): A flag to indicate whether to correct Cp and Ct for tilt
            usually for a floating turbine.
            Optional, defaults to False.
        floating_tilt_table (dict[str, float]): Look up table of tilt angles at a series of
            wind speeds. The dictionary must have the following keys with equal length values:
                {
                    "wind_speeds": List[float],
                    "tilt": List[float],
                }
            Required if `correct_cp_ct_for_tilt = True`. Defaults to None.
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
    power_thrust_table: dict[str, NDArrayFloat] = field(converter=floris_numeric_dict_converter)

    correct_cp_ct_for_tilt: bool = field(default=False)
    floating_tilt_table: dict[str, NDArrayFloat] | None = field(default=None)

    # Even though this Turbine class does not support the multidimensional features as they
    # are implemented in TurbineMultiDim, providing the following two attributes here allows
    # the turbine data inputs to keep the multidimensional Cp and Ct curve but switch them off
    # with multi_dimensional_cp_ct = False
    multi_dimensional_cp_ct: bool = field(default=False)
    power_thrust_data_file: str = field(default=None)

    # Initialized in the post_init function
    rotor_radius: float = field(init=False)
    rotor_area: float = field(init=False)
    fCt_interp: interp1d = field(init=False)
    power_interp: interp1d = field(init=False)
    tilt_interp: interp1d = field(init=False, default=None)

    def __attrs_post_init__(self) -> None:
        self._initialize_power_thrust_interpolation()
        self.__post_init__()

    def __post_init__(self) -> None:
        self._initialize_tilt_interpolation()

    def _initialize_power_thrust_interpolation(self) -> None:
        # TODO This validation for the power thrust tables should go in the turbine library
        # since it's preprocessing
        # Remove any duplicate wind speed entries
        # _, duplicate_filter = np.unique(self.wind_speed, return_index=True)
        # self.power = self.power[duplicate_filter]
        # self.thrust = self.thrust[duplicate_filter]
        # self.wind_speed = self.wind_speed[duplicate_filter]

        wind_speeds = self.power_thrust_table["wind_speed"]
        cp_interp = interp1d(
            wind_speeds,
            self.power_thrust_table["power"],
            fill_value=(0.0, 1.0),
            bounds_error=False,
        )
        self.power_interp = interp1d(
            wind_speeds,
            (
                0.5 * self.rotor_area
                * cp_interp(wind_speeds)
                * self.generator_efficiency
                * wind_speeds ** 3
            ),
            bounds_error=False,
            fill_value=0
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
            self.power_thrust_table["thrust"],
            fill_value=(0.0001, 0.9999),
            bounds_error=False,
        )

    def _initialize_tilt_interpolation(self) -> None:
        # TODO:
        # Remove any duplicate wind speed entries
        # _, duplicate_filter = np.unique(self.wind_speeds, return_index=True)
        # self.tilt = self.tilt[duplicate_filter]
        # self.wind_speeds = self.wind_speeds[duplicate_filter]

        if self.floating_tilt_table is not None:
            self.floating_tilt_table = floris_numeric_dict_converter(self.floating_tilt_table)

        # If defined, create a tilt interpolation function for floating turbines.
        # fill_value currently set to apply the min or max tilt angles if outside
        # of the interpolation range.
        if self.correct_cp_ct_for_tilt:
            self.tilt_interp = interp1d(
                self.floating_tilt_table["wind_speed"],
                self.floating_tilt_table["tilt"],
                fill_value=(0.0, self.floating_tilt_table["tilt"][-1]),
                bounds_error=False,
            )

    @power_thrust_table.validator
    def check_power_thrust_table(self, instance: attrs.Attribute, value: dict) -> None:
        """
        Verify that the power and thrust tables are given with arrays of equal length
        to the wind speed array.
        """
        if len(value.keys()) != 3 or set(value.keys()) != {"wind_speed", "power", "thrust"}:
            raise ValueError(
                """
                power_thrust_table dictionary must have the form:
                    {
                        "wind_speed": List[float],
                        "power": List[float],
                        "thrust": List[float],
                    }
                """
            )

        if any(e.ndim > 1 for e in (value["power"], value["thrust"], value["wind_speed"])):
            raise ValueError("power, thrust, and wind_speed inputs must be 1-D.")

        if len( {value["power"].size, value["thrust"].size, value["wind_speed"].size} ) > 1:
            raise ValueError("power, thrust, and wind_speed tables must be the same size.")

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
    def check_floating_tilt_table(self, instance: attrs.Attribute, value: dict | None) -> None:
        """
        If the tilt / wind_speed table is defined, verify that the tilt and
        wind_speed arrays are the same length.
        """
        if value is None:
            return

        if len(value.keys()) != 2 or set(value.keys()) != {"wind_speed", "tilt"}:
            raise ValueError(
                """
                floating_tilt_table dictionary must have the form:
                    {
                        "wind_speed": List[float],
                        "tilt": List[float],
                    }
                """
            )

        if any(len(np.shape(e)) > 1 for e in (value["tilt"], value["wind_speed"])):
            raise ValueError("tilt and wind_speed inputs must be 1-D.")

        if len( {len(value["tilt"]), len(value["wind_speed"])} ) > 1:
            raise ValueError("tilt and wind_speed inputs must be the same size.")

    @correct_cp_ct_for_tilt.validator
    def check_for_cp_ct_correct_flag_if_floating(
        self,
        instance: attrs.Attribute,
        value: bool
    ) -> None:
        """
        Check that the boolean flag exists for correcting Cp/Ct for tilt
        if a tile/wind_speed table is also defined.
        """
        if self.correct_cp_ct_for_tilt and self.floating_tilt_table is None:
            raise ValueError(
                "To enable the Cp and Ct tilt correction, a tilt table must be given."
            )
