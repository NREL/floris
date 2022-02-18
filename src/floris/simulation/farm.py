# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# from __future__ import annotations
from typing import Any, List

import attrs
from attrs import define, field
import numpy as np

from floris.type_dec import (
    floris_array_converter,
    NDArrayFloat
)
from floris.utilities import Vec3
from floris.simulation import BaseClass
from floris.simulation import Turbine


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
    turbine_type: List = field()
    turbine_definitions: dict = field()

    yaw_angles: NDArrayFloat = field(init=False)
    coordinates: List[Vec3] = field(init=False)
    hub_heights: NDArrayFloat = field(init=False)

    @layout_x.validator
    def check_x(self, instance: attrs.Attribute, value: Any) -> None:
        if len(value) != len(self.layout_y):
            raise ValueError("layout_x and layout_y must have the same number of entries.")

    @layout_y.validator
    def check_y(self, instance: attrs.Attribute, value: Any) -> None:
        if len(value) != len(self.layout_x):
            raise ValueError("layout_x and layout_y must have the same number of entries.")

    def initialize(self, sorted_indices):
        # Sort yaw angles from most upstream to most downstream wind turbine
        self.yaw_angles = np.take_along_axis(
            self.yaw_angles,
            sorted_indices[:, :, :, 0, 0],
            axis=2,
        )

    def construct_hub_heights(self):
        self.hub_heights = np.array([self.turbine_definitions[turb_type]['hub_height'] for turb_type in self.turbine_type])

    def construct_rotor_diameters(self):
        self.rotor_diameters = np.array([self.turbine_definitions[turb_type]['rotor_diameter'] for turb_type in self.turbine_type])#[None, None, :, None, None]

    def construct_turbine_TSRs(self):
        self.TSRs = np.array([self.turbine_definitions[turb_type]['TSR'] for turb_type in self.turbine_type])

    def construc_turbine_pPs(self):
        self.pPs = np.array([self.turbine_definitions[turb_type]['pP'] for turb_type in self.turbine_type])

    def construct_turbine_map(self):
        self.turbine_map = [Turbine.from_dict(self.turbine_definitions[turb_type]) for turb_type in self.turbine_type]

    def construct_turbine_fCts(self):
        self.turbine_fCts = [turb.fCt_interp for turb in self.turbine_map]

    def construct_turbine_fCps(self):
        self.turbine_fCps = [turb.fCp_interp for turb in self.turbine_map]

    def construct_turbine_power_interps(self):
        self.turbine_power_interps = [turb.power_interp for turb in self.turbine_map]

    def construct_coordinates(self, reference_z: float):
        self.coordinates = np.array(
            [Vec3([x, y, z]) for x, y, z in zip(self.layout_x, self.layout_y, self.hub_heights)]
        )

    def expand_farm_properties(self, n_wind_directions: int, n_wind_speeds: int, sorted_coord_indices):
        template_shape = np.ones_like(sorted_coord_indices)
        self.hub_heights = np.take_along_axis(self.hub_heights * template_shape, sorted_coord_indices, axis=2)
        self.rotor_diameters = np.take_along_axis(self.rotor_diameters * template_shape, sorted_coord_indices, axis=2)
        self.TSRs = np.take_along_axis(self.TSRs * template_shape, sorted_coord_indices, axis=2)
        self.pPs = np.take_along_axis(self.pPs * template_shape, sorted_coord_indices, axis=2)
        self.turbine_fCts = np.take_along_axis(
            np.reshape(self.turbine_fCts * n_wind_directions * n_wind_speeds, np.shape(sorted_coord_indices)),
            sorted_coord_indices,
            axis=2
        )
        self.turbine_fCps = np.take_along_axis(
            np.reshape(self.turbine_fCps * n_wind_directions * n_wind_speeds, np.shape(sorted_coord_indices)),
            sorted_coord_indices,
            axis=2
        )
        self.turbine_power_interps = np.take_along_axis(
            np.reshape(self.turbine_power_interps * n_wind_directions * n_wind_speeds, np.shape(sorted_coord_indices)),
            sorted_coord_indices,
            axis=2
        )

    def set_yaw_angles(self, n_wind_directions: int, n_wind_speeds: int):
        # TODO Is this just for initializing yaw angles to zero?
        self.yaw_angles = np.zeros((n_wind_directions, n_wind_speeds, self.n_turbines))

    def finalize(self, unsorted_indices):
        self.yaw_angles = np.take_along_axis(self.yaw_angles, unsorted_indices[:,:,:,0,0], axis=2)
        self.hub_heights = np.take_along_axis(self.hub_heights , unsorted_indices[:,:,:,0,0], axis=2)
        self.rotor_diameters = np.take_along_axis(self.rotor_diameters, unsorted_indices[:,:,:,0,0], axis=2)
        self.TSRs = np.take_along_axis(self.TSRs, unsorted_indices[:,:,:,0,0], axis=2)
        self.pPs = np.take_along_axis(self.pPs, unsorted_indices[:,:,:,0,0], axis=2)
        self.turbine_fCts = np.take_along_axis(self.turbine_fCts, unsorted_indices[:,:,:,0,0], axis=2)
        self.turbine_fCps = np.take_along_axis(self.turbine_fCps, unsorted_indices[:,:,:,0,0], axis=2)
        self.turbine_power_interps = np.take_along_axis(self.turbine_power_interps, unsorted_indices[:,:,:,0,0], axis=2)

    @property
    def n_turbines(self):
        return len(self.layout_x)
