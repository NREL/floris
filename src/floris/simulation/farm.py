# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import annotations
from typing import Any, Callable

import attrs
from attrs import define, field
import numpy as np

from floris.type_dec import (
    floris_array_converter,
    NDArrayFloat
)
from floris.utilities import Vec3
from floris.simulation import Turbine
from floris.simulation import BaseClass


def turbine_factory(value: Turbine | dict):
    if type(value) is Turbine:
        return value
    elif type(value) is dict:
        return Turbine.from_dict(value)
    else:
        raise ValueError


@define
class Farm(BaseClass):
    """Farm is where wind power plants should be instantiated from a YAML configuration
    file. The Farm will create a heterogenous set of turbines that compose a windfarm,
    validate the inputs, and then create a vectorized representation of the the turbine
    data.

    Farm is the container class of the FLORIS package. It brings
    together all of the component objects after input (i.e., Turbine,
    Wake, FlowField) and packages everything into the appropriate data
    type. Farm should also be used as an entry point to probe objects
    for generating output.
    """

    layout_x: NDArrayFloat = field(converter=floris_array_converter)
    layout_y: NDArrayFloat = field(converter=floris_array_converter)
    turbine: Turbine = field(converter=turbine_factory)
    # n_wind_directions: int = field(converter=int)
    # n_wind_speeds: int = field(converter=int)

    yaw_angles: NDArrayFloat = field(init=False)
    coordinates: list[Vec3] = field(init=False)
    # yaw_angles: NDArrayFloat = field(init=False)
    rotor_diameter: float = field(init=False)
    hub_height: float = field(init=False)
    pP: float = field(init=False)
    pT: float = field(init=False)
    generator_efficiency: float = field(init=False)
    TSR: float = field(init=False)
    fCt_interp: Callable = field(init=False)
    power_interp: Callable = field(init=False)

    @layout_x.validator
    def check_x(self, instance: attrs.Attribute, value: Any) -> None:
        if len(value) != len(self.layout_y):
            raise ValueError("layout_x and layout_y must have the same number of entries.")

    @layout_y.validator
    def check_y(self, instance: attrs.Attribute, value: Any) -> None:
        if len(value) != len(self.layout_x):
            raise ValueError("layout_x and layout_y must have the same number of entries.")

    def __attrs_post_init__(self) -> None:
        self.rotor_diameter = self.turbine.rotor_diameter
        self.hub_height = self.turbine.hub_height
        self.fCt_interp = self.turbine.fCt_interp
        self.power_interp = self.turbine.power_interp
        self.pP = self.turbine.pP
        self.pT = self.turbine.pT
        self.generator_efficiency = self.turbine.generator_efficiency
        self.TSR = self.turbine.TSR

        self.coordinates = [
            Vec3([x, y, self.hub_height]) for x, y in zip(self.layout_x, self.layout_y)
        ]

        # self.yaw_angles = np.zeros((self.n_wind_directions, self.n_wind_speeds, self.n_turbines))

    def set_yaw_angles(self, n_wind_directions: int, n_wind_speeds: int):
        self.yaw_angles = np.zeros((n_wind_directions, n_wind_speeds, self.n_turbines))

    @property
    def n_turbines(self):
        return len(self.layout_x)

    # def generate_farm_points(self) -> None:

        # def generate_turbine_tuple(turbine: Turbine) -> tuple:
        #     exclusions = ("power_thrust_table")
        #     return attr.astuple(turbine, filter=lambda attribute, value: attribute.name not in exclusions)

        # def generate_turbine_attribute_order(turbine: Turbine) -> List[str]:
        #     exclusions = ("power_thrust_table")
        #     mapping = attr.asdict(turbine, filter=lambda attribute, value: attribute.name not in exclusions)
        #     return list(mapping.keys())

        # self.rotor_diameter = [self.turbine_library[self.turbine_id[i]].rotor_diameter for i in range(self.n_turbines)]
        # self.hub_height = [self.turbine_library[self.turbine_id[i]].hub_height for i in range(self.n_turbines)]
        # self.fCt_interp = [self.turbine_library[self.turbine_id[i]].fCt_interp for i in range(self.n_turbines)]
        # self.fCp_interp = [self.turbine_library[self.turbine_id[i]].fCp_interp for i in range(self.n_turbines)]
        # self.power_interp = [self.turbine_library[self.turbine_id[i]].power_interp for i in range(self.n_turbines)]
        # self.pP = [self.turbine_library[self.turbine_id[i]].pP for i in range(self.n_turbines)]

        # # Create an array of turbine values and the column ordering
        # arbitrary_turbine = self.turbine_library[self.turbine_id[0]]
        # column_order = generate_turbine_attribute_order(arbitrary_turbine)

        # turbine_array = np.array([generate_turbine_tuple(self.turbine_library[t_id]) for t_id in self.turbine_id])
        # turbine_array = np.resize(turbine_array, (self.n_wind_directions, self.n_wind_speeds, *turbine_array.shape))

        # column_ix = {col: i for i, col in enumerate(column_order)}
        # self.rotor_diameter = turbine_array[:, :, :, column_ix["rotor_diameter"]].astype(float)
        # self.rotor_radius = turbine_array[:, :, :, column_ix["rotor_radius"]].astype(float)
        # self.rotor_area = turbine_array[:, :, :, column_ix["rotor_area"]].astype(float)
        # self.hub_height = turbine_array[:, :, :, column_ix["hub_height"]].astype(float)
        # self.pP = turbine_array[:, :, :, column_ix["pP"]].astype(float)
        # self.pT = turbine_array[:, :, :, column_ix["pT"]].astype(float)
        # self.generator_efficiency = turbine_array[:, :, :, column_ix["generator_efficiency"]].astype(float)
        # self.fCt_interp = turbine_array[:, :, :, column_ix["fCt_interp"]]
        # # TODO: should we have both fCp_interp and power_interp
        # self.fCp_interp = turbine_array[:, :, :, column_ix["fCp_interp"]]
        # self.power_interp = turbine_array[:, :, :, column_ix["power_interp"]]
