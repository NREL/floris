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
from collections.abc import Iterable, Callable

import attrs
import numpy as np
from attrs import define, field
from scipy.interpolate import interp1d

from floris.simulation import BaseClass
from floris.simulation.turbine import (
    CosineLossTurbine,
    SimpleTurbine,
)
from floris.simulation.turbine.rotor_velocity import (
    average_velocity,
    compute_tilt_angles_for_floating_turbines,
)
from floris.type_dec import (
    floris_numeric_dict_converter,
    NDArrayBool,
    NDArrayFilter,
    NDArrayFloat,
    NDArrayInt,
    NDArrayObject,
)
from floris.utilities import cosd


def power(
    velocities: NDArrayFloat,
    air_density: float,
    power_interps: dict[str, Callable],
    yaw_angles: NDArrayFloat,
    tilt_angles: NDArrayFloat,
    tilt_interps: dict[str, interp1d],
    turbine_type_map: NDArrayObject,
    turbine_power_thrust_tables: dict,
    ix_filter: NDArrayInt | Iterable[int] | None = None,
    average_method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None,
    correct_cp_ct_for_tilt: bool = False,
    multidim_condition: tuple | None = None, # Assuming only one condition at a time?
) -> NDArrayFloat:
    """Power produced by a turbine adjusted for yaw and tilt. Value
    given in Watts.

    Args:
        velocities (NDArrayFloat[n_findex, n_turbines, n_grid, n_grid]): The velocities at a
            turbine.
        air_density (float): air density for simulation [kg/m^3]
        power_interp (dict[str, interp1d]): A dictionary of power interpolation functions for
            each turbine type.
        turbine_type_map: (NDArrayObject[wd, ws, turbines]): The Turbine type definition for
            each turbine.
        turbine_power_thrust_tables: Reference data for the power and thrust representation
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

    # Down-select inputs if ix_filter is given
    if ix_filter is not None:
        velocities = velocities[:, ix_filter]
        yaw_angles = yaw_angles[:, ix_filter]
        tilt_angles = tilt_angles[:, ix_filter]
        turbine_type_map = turbine_type_map[:, ix_filter]
        if type(correct_cp_ct_for_tilt) is bool:
            pass
        else:
            correct_cp_ct_for_tilt = correct_cp_ct_for_tilt[:, ix_filter]

    # Loop over each turbine type given to get power for all turbines
    p = np.zeros(np.shape(velocities)[0:2])
    turb_types = np.unique(turbine_type_map)
    for turb_type in turb_types:
        # Handle possible multidimensional power thrust tables
        if "power" in turbine_power_thrust_tables[turb_type]: # normal
            power_thrust_table = turbine_power_thrust_tables[turb_type]
        else: # assumed multidimensional, use multidim lookup
            # Currently, only works for single mutlidim condition. May need to
            # loop in the case where there are multiple conditions.
            power_thrust_table = turbine_power_thrust_tables[turb_type][multidim_condition]

        # Construct full set of possible keyword arguments for power()
        power_model_kwargs = {
            "power_thrust_table": power_thrust_table,
            "velocities": velocities,
            "air_density": air_density,
            "yaw_angles": yaw_angles,
            "tilt_angles": tilt_angles,
            "tilt_interp": tilt_interps[turb_type],
            "average_method": average_method,
            "cubature_weights": cubature_weights,
            "correct_cp_ct_for_tilt": correct_cp_ct_for_tilt,
        }

        # Using a masked array, apply the power for all turbines of the current
        # type to the main power
        p += power_interps[turb_type](**power_model_kwargs) * (turbine_type_map == turb_type)

    return p


def Ct(
    velocities: NDArrayFloat,
    yaw_angles: NDArrayFloat,
    tilt_angles: NDArrayFloat,
    fCt: dict,
    tilt_interps: NDArrayObject,
    correct_cp_ct_for_tilt: NDArrayBool,
    turbine_type_map: NDArrayObject,
    turbine_power_thrust_tables: dict,
    ix_filter: NDArrayFilter | Iterable[int] | None = None,
    average_method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None,
    multidim_condition: tuple | None = None, # Assuming only one condition at a time?
) -> NDArrayFloat:

    """Thrust coefficient of a turbine incorporating the yaw angle.
    The value is interpolated from the coefficient of thrust vs
    wind speed table using the rotor swept area average velocity.

    Args:
        velocities (NDArrayFloat[findex, turbines, grid1, grid2]): The velocity field at
            a turbine.
        yaw_angle (NDArrayFloat[findex, turbines]): The yaw angle for each turbine.
        tilt_angle (NDArrayFloat[findex, turbines]): The tilt angle for each turbine.
        ref_tilt (NDArrayFloat[findex, turbines]): The reference tilt angle for each turbine
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

    if isinstance(yaw_angles, list):
        yaw_angles = np.array(yaw_angles)

    if isinstance(tilt_angles, list):
        tilt_angles = np.array(tilt_angles)

    # Down-select inputs if ix_filter is given
    if ix_filter is not None:
        velocities = velocities[:, ix_filter]
        yaw_angles = yaw_angles[:, ix_filter]
        tilt_angles = tilt_angles[:, ix_filter]
        turbine_type_map = turbine_type_map[:, ix_filter]
        if type(correct_cp_ct_for_tilt) is bool:
            pass
        else:
            correct_cp_ct_for_tilt = correct_cp_ct_for_tilt[:, ix_filter]

    # Loop over each turbine type given to get thrust coefficient for all turbines
    thrust_coefficient = np.zeros(np.shape(velocities)[0:2])
    turb_types = np.unique(turbine_type_map)
    for turb_type in turb_types:
        # Handle possible multidimensional power thrust tables
        if "thrust_coefficient" in turbine_power_thrust_tables[turb_type]: # normal
            power_thrust_table = turbine_power_thrust_tables[turb_type]
        else: # assumed multidimensional, use multidim lookup
            # Currently, only works for single mutlidim condition. May need to
            # loop in the case where there are multiple conditions.
            power_thrust_table = turbine_power_thrust_tables[turb_type][multidim_condition]

        # Construct full set of possible keyword arguments for thrust_coefficient()
        thrust_model_kwargs = {
            "power_thrust_table": power_thrust_table,
            "velocities": velocities,
            "yaw_angles": yaw_angles,
            "tilt_angles": tilt_angles,
            "tilt_interp": tilt_interps[turb_type],
            "average_method": average_method,
            "cubature_weights": cubature_weights,
            "correct_cp_ct_for_tilt": correct_cp_ct_for_tilt,
        }

        # Using a masked array, apply the thrust coefficient for all turbines of the current
        # type to the main thrust coefficient array
        thrust_coefficient += (
            fCt[turb_type](**thrust_model_kwargs) * (turbine_type_map == turb_type)
        )

    return thrust_coefficient


def axial_induction(
    velocities: NDArrayFloat,  # (findex, turbines, grid, grid)
    yaw_angles: NDArrayFloat,  # (findex, turbines)
    tilt_angles: NDArrayFloat,  # (findex, turbines)
    ref_tilt: NDArrayFloat,
    fCt: dict,  # (turbines)
    tilt_interps: NDArrayObject,  # (turbines)
    correct_cp_ct_for_tilt: NDArrayBool, # (findex, turbines)
    turbine_type_map: NDArrayObject, # (findex, turbines)
    turbine_power_thrust_tables: dict, # (turbines)
    ix_filter: NDArrayFilter | Iterable[int] | None = None,
    average_method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None,
    multidim_condition: tuple | None = None, # Assuming only one condition at a time?
) -> NDArrayFloat:
    """Axial induction factor of the turbine incorporating
    the thrust coefficient and yaw angle.

    Args:
        velocities (NDArrayFloat): The velocity field at each turbine; should be shape:
            (number of turbines, ngrid, ngrid), or (ngrid, ngrid) for a single turbine.
        yaw_angle (NDArrayFloat[findex, turbines]): The yaw angle for each turbine.
        tilt_angle (NDArrayFloat[findex, turbines]): The tilt angle for each turbine.
        ref_tilt (NDArrayFloat[findex, turbines]): The reference tilt angle for each turbine
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

    # TODO: Should the axial induction factor be defined on the turbine submodel, as
    # thrust_coefficient() and power() are?

    if isinstance(yaw_angles, list):
        yaw_angles = np.array(yaw_angles)

    # TODO: Should the tilt_angle used for the return calculation be modified the same as the
    # tilt_angle in Ct, if the user has supplied a tilt/wind_speed table?
    if isinstance(tilt_angles, list):
        tilt_angles = np.array(tilt_angles)

    # Get Ct first before modifying any data
    thrust_coefficient = Ct(
        velocities,
        yaw_angles,
        tilt_angles,
        fCt,
        tilt_interps,
        correct_cp_ct_for_tilt,
        turbine_type_map,
        turbine_power_thrust_tables,
        ix_filter,
        average_method,
        cubature_weights,
        multidim_condition
    )

    # Then, process the input arguments as needed for this function
    if ix_filter is not None:
        yaw_angles = yaw_angles[:, ix_filter]
        tilt_angles = tilt_angles[:, ix_filter]
        ref_tilt = ref_tilt[:, ix_filter]

    # TODO: Cosine yaw loss hardcoded here? Is this what we want?
    # also, assumes the same ref_tilt throughout?
    return (
        0.5
        / (cosd(yaw_angles)
        * cosd(tilt_angles - ref_tilt))
        * (
            1 - np.sqrt(
                1 - thrust_coefficient * cosd(yaw_angles) * cosd(tilt_angles - ref_tilt)
            )
        )
    )

TURBINE_MODEL_MAP = {
    "power_thrust_model": {
        "simple": SimpleTurbine,
        "cosine-loss": CosineLossTurbine
    },
#     "velocity_averaging_method": {
#         "cubic-mean": cubic_mean
#     } # BUG/TODO: THIS IS NOT YET PASSED THROUGH!
}

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
        ref_air_density (float): The density at which the provided Cp and Ct curves are defined.
        ref_tilt (float): The implicit tilt of the turbine for which the Cp and Ct
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
    #pP: float = field()
    #pT: float = field()
    TSR: float = field()
    generator_efficiency: float = field()
    #ref_air_density: float = field()
    #ref_tilt: float = field()
    power_thrust_table: dict[str, NDArrayFloat] = field(converter=floris_numeric_dict_converter)
    power_thrust_model: str = field(default="cosine-loss")

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
    thrust_coefficient_function: Callable = field(init=False)
    power_function: Callable = field(init=False)
    tilt_interp: interp1d = field(init=False, default=None)

    def __attrs_post_init__(self) -> None:
        self._initialize_power_thrust_functions()
        self.__post_init__()

    def __post_init__(self) -> None:
        self._initialize_tilt_interpolation()

    def _initialize_power_thrust_functions(self) -> None:
        # TODO This validation for the power thrust tables should go in the turbine library
        # since it's preprocessing
        turbine_function_model = TURBINE_MODEL_MAP["power_thrust_model"][self.power_thrust_model]
        self.power_function = turbine_function_model.power
        self.thrust_coefficient_function = turbine_function_model.thrust_coefficient

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
        # if (len(value.keys()) != 3 or
        #     set(value.keys()) != {"wind_speed", "power", "thrust_coefficient"}):
        if not {"wind_speed", "power", "thrust_coefficient"} <= set(value.keys()):
            raise ValueError(
                """
                power_thrust_table dictionary must contain:
                    {
                        "wind_speed": List[float],
                        "power": List[float],
                        "thrust_coefficient": List[float],
                    }
                """
            )

        if any(e.ndim > 1 for e in
            (value["power"], value["thrust_coefficient"], value["wind_speed"])
            ):
            raise ValueError("power, thrust_coefficient, and wind_speed inputs must be 1-D.")

        if (len( {value["power"].size, value["thrust_coefficient"].size, value["wind_speed"].size} )
            > 1):
            raise ValueError(
                "power, thrust_coefficient, and wind_speed tables must be the same size."
            )

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