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

import copy
from pathlib import Path
from typing import Any, List

import attrs
import numpy as np
import yaml
from attrs import define, field

from floris.simulation import (
    BaseClass,
    State,
    Turbine,
)
from floris.type_dec import (
    convert_to_path,
    floris_array_converter,
    NDArrayFloat,
    NDArrayObject,
)
from floris.utilities import load_yaml, Vec3


default_turbine_library_path = Path(__file__).parents[1] / "turbine_library"


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
    turbine_library_path: Path = field(
        default=default_turbine_library_path, converter=convert_to_path
    )

    turbine_definitions: dict = field(init=False)
    yaw_angles: NDArrayFloat = field(init=False)
    yaw_angles_sorted: NDArrayFloat = field(init=False)
    coordinates: List[Vec3] = field(init=False)
    hub_heights: NDArrayFloat = field(init=False)
    hub_heights_sorted: NDArrayFloat = field(init=False, default=[])
    turbine_fCts: tuple = field(init=False, default=[])
    turbine_type_map_sorted: NDArrayObject = field(init=False, default=[])
    rotor_diameters_sorted: NDArrayFloat = field(init=False, default=[])
    TSRs_sorted: NDArrayFloat = field(init=False, default=[])
    pPs_sorted: NDArrayFloat = field(init=False, default=[])

    def __attrs_post_init__(self) -> None:
        self.check_turbine_type()

    @layout_x.validator
    def check_x(self, attribute: attrs.Attribute, value: Any) -> None:
        if len(value) != len(self.layout_y):
            raise ValueError("layout_x and layout_y must have the same number of entries.")

    @layout_y.validator
    def check_y(self, attribute: attrs.Attribute, value: Any) -> None:
        if len(value) != len(self.layout_x):
            raise ValueError("layout_x and layout_y must have the same number of entries.")

    @turbine_library_path.validator
    def check_library_path(self, attribute: attrs.Attribute, value: Path) -> None:
        """Ensures that the input to `library_path` exists and is a directory."""
        if not value.is_dir():
            raise FileExistsError(f"The input file path: {str(value)} is not a valid directory.")

    def check_turbine_type(self) -> None:
        if len(self.turbine_type) != len(self.layout_x):
            if len(self.turbine_type) == 1:
                self.turbine_type *= len(self.layout_x)
            elif np.unique(self.turbine_type).size == 1:
                self.turbine_type = [self.turbine_type[0]] * len(self.layout_x)
            else:
                raise ValueError(
                    "turbine_type must have the same number of entries as layout_x/layout_y or have"
                    " a single turbine_type value."
                )

        # If the user specified the default location, do not check against duplicated definitions
        turbine_map = {}
        unique_turbines = [
            yaml.safe_load(el_j)
            for el_j in {yaml.dump(el_i, default_flow_style=False)
            for el_i in self.turbine_type}
        ]
        for turbine in unique_turbines:

            # If the passed data are already turbine dictionaries, skip the file loading
            if isinstance(turbine, str):
                # Check if the file exists in the internal and/or external libary
                internal_fn = (default_turbine_library_path / turbine).with_suffix(".yaml")
                external_fn = (self.turbine_library_path / turbine).with_suffix(".yaml")
                in_internal = internal_fn.exists()
                in_external = external_fn.exists()

                # If an external library is used, and the file is a duplicate with what already
                # exists, then raise an error
                is_separate_path = self.turbine_library_path != default_turbine_library_path
                if is_separate_path and in_external and in_internal:
                    raise ValueError(
                        f"The turbine type: {turbine} exists in both the internal and external"
                        " turbine library."
                    )
                if in_internal:
                    full_path = internal_fn
                elif in_external:
                    full_path = external_fn
                else:
                    raise ValueError(
                        f"The turbine type: {turbine} exists in both the internal and external"
                        " turbine library."
                    )
                turbine_map[turbine] = turbine = load_yaml(full_path)
            elif isinstance(turbine, dict):
                turbine_map[turbine["turbine_type"]] = turbine
                full_path = turbine["turbine_type"]

            # Log a warning if the reference air density doesn't exist
            if "ref_density_cp_ct" not in turbine:
                self.logger.warning(
                    f"The value ref_density_cp_ct is not defined in the file {full_path}."
                    "This value is not the simulated air density but is the density "
                    "at which the cp/ct curves are defined. In previous versions, this "
                    "was assumed to be 1.225. Future versions of FLORIS will give an error "
                    "if this value is not explicitly defined. Currently, this value is "
                    "being set to the prior default value of 1.225."
                )
                turbine['ref_density_cp_ct'] = 1.225
                turbine_map[turbine["turbine_type"]] = turbine

        self.turbine_definitions = [
            copy.deepcopy(turbine_map[el]) if isinstance(el, str)
            else copy.deepcopy(turbine_map[el["turbine_type"]])
            for el in self.turbine_type
        ]

    def initialize(self, sorted_indices):
        # Sort yaw angles from most upstream to most downstream wind turbine
        self.yaw_angles_sorted = np.take_along_axis(
            self.yaw_angles,
            sorted_indices[:, :, :, 0, 0],
            axis=2,
        )
        self.state = State.INITIALIZED

    def construct_hub_heights(self):
        self.hub_heights = np.array([turb['hub_height'] for turb in self.turbine_definitions])

    def construct_rotor_diameters(self):
        self.rotor_diameters = np.array([
            turb['rotor_diameter'] for turb in self.turbine_definitions
        ])

    def construct_turbine_TSRs(self):
        self.TSRs = np.array([turb['TSR'] for turb in self.turbine_definitions])

    def construc_turbine_pPs(self):
        self.pPs = np.array([turb['pP'] for turb in self.turbine_definitions])

    def construc_turbine_ref_density_cp_cts(self):
        self.ref_density_cp_cts = np.array([
            turb['ref_density_cp_ct'] for turb in self.turbine_definitions
        ])

    def construct_turbine_map(self):
        self.turbine_map = [Turbine.from_dict(turb) for turb in self.turbine_definitions]

    def construct_turbine_fCts(self):
        self.turbine_fCts = [(turb.turbine_type, turb.fCt_interp) for turb in self.turbine_map]

    def construct_turbine_fCps(self):
        self.turbine_fCps = [(turb.turbine_type, turb.fCp_interp) for turb in self.turbine_map]

    def construct_turbine_power_interps(self):
        self.turbine_power_interps = [
            (turb.turbine_type, turb.power_interp) for turb in self.turbine_map
        ]

    def construct_coordinates(self):
        self.coordinates = np.array([
            Vec3([x, y, z]) for x, y, z in zip(self.layout_x, self.layout_y, self.hub_heights)
        ])

    def expand_farm_properties(
        self,
        n_wind_directions: int,
        n_wind_speeds: int,
        sorted_coord_indices
    ):
        template_shape = np.ones_like(sorted_coord_indices)
        self.hub_heights_sorted = np.take_along_axis(
            self.hub_heights * template_shape,
            sorted_coord_indices,
            axis=2
        )
        self.rotor_diameters_sorted = np.take_along_axis(
            self.rotor_diameters * template_shape,
            sorted_coord_indices,
            axis=2
        )
        self.TSRs_sorted = np.take_along_axis(
            self.TSRs * template_shape,
            sorted_coord_indices,
            axis=2
        )
        self.pPs_sorted = np.take_along_axis(
            self.pPs * template_shape,
            sorted_coord_indices,
            axis=2
        )
        self.turbine_type_names_sorted = [turb["turbine_type"] for turb in self.turbine_definitions]
        self.turbine_type_map_sorted = np.take_along_axis(
            np.reshape(
                self.turbine_type_names_sorted * n_wind_directions,
                np.shape(sorted_coord_indices)
            ),
            sorted_coord_indices,
            axis=2
        )

    def set_yaw_angles(self, n_wind_directions: int, n_wind_speeds: int):
        # TODO Is this just for initializing yaw angles to zero?
        self.yaw_angles = np.zeros((n_wind_directions, n_wind_speeds, self.n_turbines))
        self.yaw_angles_sorted = np.zeros((n_wind_directions, n_wind_speeds, self.n_turbines))

    def finalize(self, unsorted_indices):
        self.yaw_angles = np.take_along_axis(
            self.yaw_angles_sorted,
            unsorted_indices[:,:,:,0,0],
            axis=2
        )
        self.hub_heights = np.take_along_axis(
            self.hub_heights_sorted,
            unsorted_indices[:,:,:,0,0],
            axis=2
        )
        self.rotor_diameters = np.take_along_axis(
            self.rotor_diameters_sorted,
            unsorted_indices[:,:,:,0,0],
            axis=2
        )
        self.TSRs = np.take_along_axis(
            self.TSRs_sorted,
            unsorted_indices[:,:,:,0,0],
            axis=2
        )
        self.pPs = np.take_along_axis(
            self.pPs_sorted,
            unsorted_indices[:,:,:,0,0],
            axis=2
        )
        self.turbine_type_map = np.take_along_axis(
            self.turbine_type_map_sorted,
            unsorted_indices[:,:,:,0,0],
            axis=2
        )
        self.state.USED

    @property
    def n_turbines(self):
        return len(self.layout_x)
