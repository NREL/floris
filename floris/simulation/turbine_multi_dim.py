# Copyright 2023 NREL

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

import numpy as np
import pandas as pd
from attrs import define, field
from flatten_dict import flatten
from scipy.interpolate import interp1d

# import floris.simulation.turbine as turbine
from floris.simulation import (
    average_velocity,
    compute_tilt_angles_for_floating_turbines,
    TiltTable,
    Turbine,
)
from floris.simulation.turbine import _filter_convert
from floris.type_dec import (
    NDArrayBool,
    NDArrayFilter,
    NDArrayFloat,
    NDArrayInt,
    NDArrayObject,
)
from floris.utilities import cosd


def power_multidim(
    ref_density_cp_ct: float,
    rotor_effective_velocities: NDArrayFloat,
    power_interp: NDArrayObject,
    ix_filter: NDArrayInt | Iterable[int] | None = None,
) -> NDArrayFloat:
    """Power produced by a turbine defined with multi-dimensional
    Cp/Ct values, adjusted for yaw and tilt. Value given in Watts.

    Args:
        ref_density_cp_cts (NDArrayFloat[wd, ws, turbines]): The reference density for each turbine
        rotor_effective_velocities (NDArrayFloat[wd, ws, turbines, grid1, grid2]): The rotor
            effective velocities at a turbine.
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

    # Down-select inputs if ix_filter is given
    if ix_filter is not None:
        ix_filter = _filter_convert(ix_filter, rotor_effective_velocities)
        power_interp = power_interp[:, :, ix_filter]
        rotor_effective_velocities = rotor_effective_velocities[:, :, ix_filter]
    # Loop over each turbine to get power for all turbines
    p = np.zeros(np.shape(rotor_effective_velocities))
    for i, wd in enumerate(power_interp):
        for j, ws in enumerate(wd):
            for k, turb in enumerate(ws):
                p[i, j, k] = power_interp[i, j, k](rotor_effective_velocities[i, j, k])

    return p * ref_density_cp_ct


def Ct_multidim(
    velocities: NDArrayFloat,
    yaw_angle: NDArrayFloat,
    tilt_angle: NDArrayFloat,
    ref_tilt_cp_ct: NDArrayFloat,
    fCt: list,
    tilt_interp: NDArrayObject,
    correct_cp_ct_for_tilt: NDArrayBool,
    turbine_type_map: NDArrayObject,
    ix_filter: NDArrayFilter | Iterable[int] | None = None,
    average_method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None
) -> NDArrayFloat:

    """Thrust coefficient of a turbine defined with multi-dimensional
    Cp/Ct values, incorporating the yaw angle. The value is interpolated
    from the coefficient of thrust vs wind speed table using the rotor
    swept area average velocity.

    Args:
        velocities (NDArrayFloat[wd, ws, turbines, grid1, grid2]): The velocity field at
            a turbine.
        yaw_angle (NDArrayFloat[wd, ws, turbines]): The yaw angle for each turbine.
        tilt_angle (NDArrayFloat[wd, ws, turbines]): The tilt angle for each turbine.
        ref_tilt_cp_ct (NDArrayFloat[wd, ws, turbines]): The reference tilt angle for each turbine
            that the Cp/Ct tables are defined at.
        fCt (list): The thrust coefficient interpolation functions for each turbine.
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
        fCt = fCt[:, :, ix_filter]
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

    # Loop over each turbine to get thrust coefficient for all turbines
    thrust_coefficient = np.zeros(np.shape(average_velocities))
    for i, wd in enumerate(fCt):
        for j, ws in enumerate(wd):
            for k, turb in enumerate(ws):
                thrust_coefficient[i, j, k] = fCt[i, j, k](average_velocities[i, j, k])
    thrust_coefficient = np.clip(thrust_coefficient, 0.0001, 0.9999)
    effective_thrust = thrust_coefficient * cosd(yaw_angle) * cosd(tilt_angle - ref_tilt_cp_ct)
    return effective_thrust


def axial_induction_multidim(
    velocities: NDArrayFloat,  # (wind directions, wind speeds, turbines, grid, grid)
    yaw_angle: NDArrayFloat,  # (wind directions, wind speeds, turbines)
    tilt_angle: NDArrayFloat,  # (wind directions, wind speeds, turbines)
    ref_tilt_cp_ct: NDArrayFloat,
    fCt: list,  # (turbines)
    tilt_interp: NDArrayObject,  # (turbines)
    correct_cp_ct_for_tilt: NDArrayBool, # (wind directions, wind speeds, turbines)
    turbine_type_map: NDArrayObject, # (wind directions, 1, turbines)
    ix_filter: NDArrayFilter | Iterable[int] | None = None,
    average_method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None
) -> NDArrayFloat:
    """Axial induction factor of the turbines defined with multi-dimensional
    Cp/Ct values, incorporating the thrust coefficient and yaw angle.

    Args:
        velocities (NDArrayFloat): The velocity field at each turbine; should be shape:
            (number of turbines, ngrid, ngrid), or (ngrid, ngrid) for a single turbine.
        yaw_angle (NDArrayFloat[wd, ws, turbines]): The yaw angle for each turbine.
        tilt_angle (NDArrayFloat[wd, ws, turbines]): The tilt angle for each turbine.
        ref_tilt_cp_ct (NDArrayFloat[wd, ws, turbines]): The reference tilt angle for each turbine
            that the Cp/Ct tables are defined at.
        fCt (list): The thrust coefficient interpolation functions for each turbine.
        tilt_interp (Iterable[tuple]): The tilt interpolation functions for each
            turbine.
        correct_cp_ct_for_tilt (NDArrayBool[wd, ws, turbines]): Boolean for determining if the
            turbines Cp and Ct should be corrected for tilt.
        turbine_type_map: (NDArrayObject[wd, ws, turbines]): The Turbine type definition
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
    thrust_coefficient = Ct_multidim(
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


def multidim_Ct_down_select(
    turbine_fCts,
    conditions,
) -> list:
    """
    Ct interpolants are down selected from the multi-dimensional Ct data
    provided for the turbine based on the specified conditions.

    Args:
        turbine_fCts (NDArray[wd, ws, turbines]): The Ct interpolants generated from the
            multi-dimensional Ct turbine data for all specified conditions.
        conditions (dict): The conditions at which to determine which Ct interpolant to use.

    Returns:
        NDArray: The down selected Ct interpolants for the selected conditions.
    """
    downselect_turbine_fCts = np.empty_like(turbine_fCts)
    # Loop over the wind directions, wind speeds, and turbines, finding the Ct interpolant
    # that is closest to the specified multi-dimensional condition.
    for i, wd in enumerate(turbine_fCts):
        for j, ws in enumerate(wd):
            for k, turb in enumerate(ws):
                # Get the interpolant keys in float type for comparison
                keys_float = np.array([[float(v) for v in val] for val in turb.keys()])

                # Find the nearest key to the specified conditions.
                key_vals = []
                for ii, cond in enumerate(conditions.values()):
                    key_vals.append(
                        keys_float[:, ii][np.absolute(keys_float[:, ii] - cond).argmin()]
                    )

                downselect_turbine_fCts[i, j, k] = turb[tuple(key_vals)]

    return downselect_turbine_fCts


def multidim_power_down_select(
    power_interps,
    conditions,
) -> list:
    """
    Cp interpolants are down selected from the multi-dimensional Cp data
    provided for the turbine based on the specified conditions.

    Args:
        power_interps (NDArray[wd, ws, turbines]): The power interpolants generated from the
            multi-dimensional Cp turbine data for all specified conditions.
        conditions (dict): The conditions at which to determine which Ct interpolant to use.

    Returns:
        NDArray: The down selected power interpolants for the selected conditions.
    """
    downselect_power_interps = np.empty_like(power_interps)
    # Loop over the wind directions, wind speeds, and turbines, finding the power interpolant
    # that is closest to the specified multi-dimensional condition.
    for i, wd in enumerate(power_interps):
        for j, ws in enumerate(wd):
            for k, turb in enumerate(ws):
                # Get the interpolant keys in float type for comparison
                keys_float = np.array([[float(v) for v in val] for val in turb.keys()])

                # Find the nearest key to the specified conditions.
                key_vals = []
                for ii, cond in enumerate(conditions.values()):
                    key_vals.append(
                        keys_float[:, ii][np.absolute(keys_float[:, ii] - cond).argmin()]
                    )

                # Use the constructed key to choose the correct interpolant
                downselect_power_interps[i, j, k] = turb[tuple(key_vals)]

    return downselect_power_interps


@define
class MultiDimensionalPowerThrustTable():
    """Helper class to convert the multi-dimensional inputs to a dictionary of objects.
    """

    @classmethod
    def from_dataframe(self, df) -> None:
        # Validate the dataframe
        if not all(ele in df.columns.values.tolist() for ele in ["ws", "Cp", "Ct"]):
            print(df.columns.values.tolist())
            raise ValueError("Multidimensional data missing required ws/Cp/Ct data.")
        if df.columns.values[-3:].tolist() != ["ws", "Cp", "Ct"]:
            print(df.columns.values[-3:].tolist())
            raise ValueError(
                "Multidimensional data not in correct form. ws, Cp, and Ct must be "
                "defined as the last 3 columns, in that order."
            )

        # Extract the supplied dimensions, minus the required ws, Cp, and Ct columns.
        keys = df.columns.values[:-3].tolist()
        values = [df[df.columns.values[i]].unique().tolist() for i in range(len(keys))]
        values = [[str(val) for val in value] for value in values]

        # Functions for recursively building a nested dictionary from
        # an arbitrary number of paired-inputs.
        def add_level(obj, k, v):
            tmp = {}
            for val in v:
                tmp.update({val: []})
            obj.update({k: tmp})
            return obj

        def add_sub_level(obj, k):
            tmp = {}
            for key in k:
                tmp.update({key: obj})
            return tmp

        obj = {}
        # Reverse the lists to start from the lowest level of the dictionary
        keys.reverse()
        values.reverse()
        # Recursively build a nested dictionary from the user-supplied dimensions
        for i, key in enumerate(keys):
            if i == 0:
                obj = add_level(obj, key, values[i])
            else:
                obj = add_sub_level(obj, values[i])
                obj = {key: obj}

        return flatten(obj)


@define
class TurbineMultiDimensional(Turbine):
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

    power_thrust_data_file: str = field(default=None)
    multi_dimensional_cp_ct: bool = field(default=False)

    # rloc: float = float_attrib()  # TODO: goes here or on the Grid?
    # use_points_on_perimeter: bool = bool_attrib()

    # Initialized in the post_init function
    # rotor_radius: float = field(init=False)
    # rotor_area: float = field(init=False)
    # fCp_interp: interp1d = field(init=False)
    # fCt_interp: interp1d = field(init=False)
    # power_interp: interp1d = field(init=False)
    # tilt_interp: interp1d = field(init=False)


    # For the following parameters, use default values if not user-specified
    # self.rloc = float(input_dictionary["rloc"]) if "rloc" in input_dictionary else 0.5
    # if "use_points_on_perimeter" in input_dictionary:
    #     self.use_points_on_perimeter = bool(input_dictionary["use_points_on_perimeter"])
    # else:
    #     self.use_points_on_perimeter = False

    def __attrs_post_init__(self) -> None:

        # Read in the multi-dimensional data supplied by the user.
        df = pd.read_csv(self.power_thrust_data_file)

        # Build the multi-dimensional power/thrust table
        self.power_thrust_data = MultiDimensionalPowerThrustTable.from_dataframe(df)

        # Create placeholders for the interpolation functions
        self.fCt_interp = {}
        self.power_interp = {}

        # Down-select the DataFrame to have just the ws, Cp, and Ct values
        index_col = df.columns.values[:-3]
        df2 = df.set_index(index_col.tolist())

        # Loop over the multi-dimensional keys to get the correct ws/Cp/Ct data to make
        # the Ct and power interpolants.
        for key in df2.index.unique():
            # Select the correct ws/Cp/Ct data
            data = df2.loc[key]

            # Build the interpolants
            wind_speeds = data['ws'].values
            self.fCp_interp = interp1d(
                wind_speeds,
                data['Cp'].values,
                fill_value=(0.0, 1.0),
                bounds_error=False,
            )
            inner_power = (
                0.5 * self.rotor_area
                * self.fCp_interp(wind_speeds)
                * self.generator_efficiency
                * wind_speeds ** 3
            )
            self.power_interp.update({
                key: interp1d(
                    wind_speeds,
                    inner_power,
                    bounds_error=False,
                    fill_value=0
                )
            })
            self.fCt_interp.update({
                key: interp1d(
                    wind_speeds,
                    data['Ct'].values,
                    fill_value=(0.0001, 0.9999),
                    bounds_error=False,
                )
            })

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
