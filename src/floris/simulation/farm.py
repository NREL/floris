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

from typing import Any, Dict, List

import attr
import numpy as np
import numpy.typing as npt

from floris.utilities import (
    Vec3,
    attr_serializer,
    attr_floris_filter,
    attrs_array_converter,
)
from floris.simulation import Turbine
from floris.simulation import BaseClass


NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int_]


def _farm_filter(inst: attr.Attribute, value: Any) -> bool:
    if inst.name in ("wind_directions", "wind_speeds"):
        return False
    return attr_floris_filter(inst, value)


@attr.s(auto_attribs=True)
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

    n_wind_directions: int = attr.ib()
    n_wind_speeds: int = attr.ib()
    layout_x: NDArrayFloat = attr.ib(converter=attrs_array_converter)
    layout_y: NDArrayFloat = attr.ib(converter=attrs_array_converter)
    turbine: Turbine = attr.ib(converter=Turbine.from_dict)

    coordinates: list[Vec3] = attr.ib(init=False)
    yaw_angles: NDArrayFloat = attr.ib(init=False)

    rotor_diameter: NDArrayFloat = attr.ib(init=False)
    hub_height: NDArrayFloat = attr.ib(init=False)
    pP: NDArrayFloat = attr.ib(init=False)
    pT: NDArrayFloat = attr.ib(init=False)
    generator_efficiency: NDArrayFloat = attr.ib(init=False)
    TSR: NDArrayFloat = attr.ib(init=False)
    fCp_interp: NDArrayFloat = attr.ib(init=False)
    fCt_interp: NDArrayFloat = attr.ib(init=False)
    power_interp: NDArrayFloat = attr.ib(init=False)
    # rotor_radius: NDArrayFloat = attr.ib(init=False)
    # rotor_area: NDArrayFloat = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.coordinates = [
            Vec3([x, y, self.turbine.hub_height]) for x, y in zip(self.layout_x, self.layout_y)
        ]

        # TODO: assumes homogenous turbines; change this for heterogeneous turbines
        # TODO: maybe leave these on the turbine and reference from there?
        self.rotor_diameter = self.turbine.rotor_diameter
        self.hub_height = self.turbine.hub_height
        self.fCt_interp = self.turbine.fCt_interp
        self.fCp_interp = self.turbine.fCp_interp
        self.power_interp = self.turbine.power_interp
        self.pP = self.turbine.pP
        self.pT = self.turbine.pT
        self.generator_efficiency = self.turbine.generator_efficiency
        self.TSR = self.turbine.TSR

        # Turbine control settings indexed by the turbine ID
        self.yaw_angles = np.zeros((self.n_wind_directions, self.n_wind_speeds, self.n_turbines))

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

    @property
    def n_turbines(self):
        return len(self.layout_x)

    @property
    def reference_turbine_diameter(self):
        return self.rotor_diameter

    @property
    def reference_hub_height(self):
        return self.hub_height

    def _asdict(self) -> dict:
        """Creates a JSON and YAML friendly dictionary that can be save for future reloading.
        This dictionary will contain only `Python` types that can later be converted to their
        proper `Farm` formats.

        Returns:
            dict: All key, vaue pais required for class recreation.
        """
        return attr.asdict(self, filter=_farm_filter, value_serializer=attr_serializer)


    # Data validators

    # @layout_x.validator
    # def check_x_len(self, instance: attr.Attribute, value: list[float] | NDArrayFloat) -> None:
    #     if len(value) < len(self.turbine_id):
    #         raise ValueError("Not enough `layout_x` values to match the `turbine_id`s.")
    #     if len(value) > len(self.turbine_id):
    #         raise ValueError("Too many `layout_x` values to match the `turbine_id`s.")

    # @layout_y.validator
    # def check_y_len(self, instance: attr.Attribute, value: list[float] | NDArrayFloat) -> None:
    #     if len(value) < len(self.turbine_id):
    #         raise ValueError("Not enough `layout_y` values to match the `turbine_id`s")
    #     if len(value) > len(self.turbine_id):
    #         raise ValueError("Too many `layout_y` values to match the `turbine_id`s")

    # @turbine_id.validator
    # def check_turbine_id(self, instance: attr.Attribute, value: list[str]) -> None:
    #     if (
    #         len(value) != 0 and                # Not provided: use the first turbine definition
    #         len(value) != 1 and                # Single value: use the given turbine for all
    #         len(value) != len(self.layout_x)  # Otherwise, must match the number of turbines in the farm
    #     ):
    #         raise ValueError(f"Number of turbine ID's do not match the number of turbines defined.")

    # def sort_turbines(self, by: str) -> NDArrayInt:
    #     """Sorts the turbines by the given dimension.

    #     Args:
    #         by (str): The dimension to sort by; should be one of x or y.

    #     Returns:
    #         NDArrayInt: The index order for retrieving data from `data_array` or any
    #             other farm object.
    #     """
    #     if by == "x":
    #         return np.argsort(self.layout_x)
    #     elif by == "y":
    #         return np.argsort(self.layout_y)
    #     else:
    #         raise ValueError("`by` must be set to one of 'x' or 'y'.")
