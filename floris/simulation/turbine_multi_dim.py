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
from pathlib import Path

import attrs
import numpy as np
import pandas as pd
from attrs import define, field
from flatten_dict import flatten
from scipy.interpolate import interp1d

from floris.simulation import (
    average_velocity,
    compute_tilt_angles_for_floating_turbines,
    Turbine,
)
from floris.type_dec import (
    convert_to_path,
    NDArrayBool,
    NDArrayFilter,
    NDArrayFloat,
    NDArrayInt,
    NDArrayObject,
)
from floris.utilities import cosd


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
        ref_air_density (:py:obj: float): The density at which the provided
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
        power_thrust_data_file (:py:obj:`str`): The path and name of the file containing the
            multidimensional power thrust curve. The path may be an absolute location or a relative
            path to where FLORIS is being run.
        multi_dimensional_cp_ct (:py:obj:`bool`, optional): Indicates if the turbine definition is
            single dimensional (False) or multidimensional (True).
        turbine_library_path (:py:obj:`pathlib.Path`, optional): The
            :py:attr:`Farm.turbine_library_path` or :py:attr:`Farm.internal_turbine_library_path`,
            whichever is being used to load turbine definitions.
            Defaults to the internal turbine library.
    """
    multi_dimensional_cp_ct: bool = field(default=False)
    power_thrust_table: dict = field(default={})
    # TODO power_thrust_data_file is actually required and should not default to None.
    # However, the super class has optional attributes so a required attribute here breaks
    power_thrust_data_file: str = field(default=None)
    power_thrust_data: MultiDimensionalPowerThrustTable = field(default=None)
    turbine_library_path: Path = field(
        default=Path(__file__).parents[1] / "turbine_library",
        converter=convert_to_path,
        validator=attrs.validators.instance_of(Path)
    )

    # Not to be provided by the user
    condition_keys: list[str] = field(init=False, factory=list)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self._initialize_power_thrust_table()

    def _initialize_power_thrust_table(self):
        # Collect reference information
        power_thrust_table_ref = copy.deepcopy(self.power_thrust_table)
        self.power_thrust_data_file = power_thrust_table_ref.pop("power_thrust_data_file")

        # Solidify the data file path and name
        self.power_thrust_data_file = self.turbine_library_path / self.power_thrust_data_file

        # Read in the multi-dimensional data supplied by the user.
        df = pd.read_csv(self.power_thrust_data_file)

        # Build the multi-dimensional power/thrust table
        self.power_thrust_data = MultiDimensionalPowerThrustTable.from_dataframe(df)

        # Down-select the DataFrame to have just the ws, Cp, and Ct values
        index_col = df.columns.values[:-3]
        self.condition_keys = index_col.tolist()
        df2 = df.set_index(index_col.tolist())

        # Loop over the multi-dimensional keys to get the correct ws/Cp/Ct data to make
        # the Ct and power interpolants.
        self.power_thrust_table = {} # Reset
        for key in df2.index.unique():
            # Select the correct ws/Cp/Ct data
            data = df2.loc[key]

            # Build the interpolants
            self.power_thrust_table.update({
                key: {
                    "wind_speed": data['ws'].values,
                    "power": (
                        0.5 * self.rotor_area * data['Cp'].values * self.generator_efficiency
                        * data['ws'].values ** 3 * power_thrust_table_ref["ref_air_density"] / 1000
                    ), # TODO: convert this to 'power' or 'P' in data tables, as per PR #765
                    "thrust_coefficient": data['Ct'].values,
                    **power_thrust_table_ref
                },
            })
            # Add reference information at the lower level
