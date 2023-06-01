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
from attrs import define, field

from floris.simulation import (
    BaseClass,
    State,
    Turbine,
)
from floris.simulation.turbine import compute_tilt_angles_for_floating_turbines
from floris.type_dec import (
    convert_to_path,
    floris_array_converter,
    iter_validator,
    NDArrayFloat,
    NDArrayObject,
)
from floris.utilities import load_yaml, Vec3


default_turbine_library_path = Path(__file__).parents[1] / "turbine_library"


@define
class Farm(BaseClass):
    """Farm is where wind power plants should be instantiated from a YAML configuration
    file. The Farm will create a heterogenous set of turbines that compose a wind farm,
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
    # TODO: turbine_type should be immutable
    turbine_type: List = field(validator=iter_validator(list, (dict, str)))
    turbine_library_path: Path = field(
        default=default_turbine_library_path, converter=convert_to_path
    )

    turbine_definitions: list = field(init=False, validator=iter_validator(list, dict))
    coordinates: List[Vec3] = field(init=False)
    turbine_fCts: tuple = field(init=False, default=[])
    turbine_fTilts: list = field(init=False, default=[])

    yaw_angles: NDArrayFloat = field(init=False)
    yaw_angles_sorted: NDArrayFloat = field(init=False)

    tilt_angles: NDArrayFloat = field(init=False)
    tilt_angles_sorted: NDArrayFloat = field(init=False)

    hub_heights: NDArrayFloat = field(init=False)
    hub_heights_sorted: NDArrayFloat = field(init=False, default=[])

    turbine_type_map: NDArrayObject = field(init=False, default=[])
    turbine_type_map_sorted: NDArrayObject = field(init=False, default=[])

    rotor_diameters: NDArrayFloat = field(init=False, default=[])
    rotor_diameters_sorted: NDArrayFloat = field(init=False, default=[])

    TSRs: NDArrayFloat = field(init=False, default=[])
    TSRs_sorted: NDArrayFloat = field(init=False, default=[])

    pPs: NDArrayFloat = field(init=False, default=[])
    pPs_sorted: NDArrayFloat = field(init=False, default=[])

    pTs: NDArrayFloat = field(init=False, default=[])
    pTs_sorted: NDArrayFloat = field(init=False, default=[])

    ref_tilt_cp_cts: NDArrayFloat = field(init=False, default=[])
    ref_tilt_cp_cts_sorted: NDArrayFloat = field(init=False, default=[])

    correct_cp_ct_for_tilt: NDArrayFloat = field(init=False, default=[])
    correct_cp_ct_for_tilt_sorted: NDArrayFloat = field(init=False, default=[])

    def __attrs_post_init__(self) -> None:
        # Turbine definitions can be supplied in three ways:
        # - A string selecting a turbine in the floris turbine library
        # - A Python dict representation of a turbine definition
        #   - There's an option to use the yaml keyword "!include" which results in the yaml
        #     library preprocessing the inputs and loading the specified file directly into
        #     the main input file. The result is that floris sees the turbine definition as a dict.
        # - A string selecting an turbine that exists in an external turbine library
        #   specified in `turbine_library_path`

        # Load all the turbine types into a cache to be mapped to specific turbine indices later.
        # This allows to read the yaml input files once rather than every time they're given.
        # In other words, if the turbine type is already in the cache, skip that iteration of
        # the for-loop.
        turbine_definition_cache = {}
        for t in self.turbine_type:
            # If a turbine type is a dict, then it was either preprocessed by the yaml
            # library to resolve the "!include" or it was set in a script as a dict. In either case,
            # add an entry to the cache
            if isinstance(t, dict):
                if t["turbine_type"] in turbine_definition_cache:
                    continue
                turbine_definition_cache[t["turbine_type"]] = t

            # If a turbine type is a string, then it is expected in the internal or external
            # turbine library
            if isinstance(t, str):
                if t in turbine_definition_cache:
                    continue

                # Check if the file exists in the internal and/or external library
                internal_fn = (default_turbine_library_path / t).with_suffix(".yaml")
                external_fn = (self.turbine_library_path / t).with_suffix(".yaml")
                in_internal = internal_fn.exists()
                in_external = external_fn.exists()

                # If an external library is used and there's a duplicate of an internal
                # definition, then raise an error
                is_unique_path = self.turbine_library_path != default_turbine_library_path
                if is_unique_path and in_external and in_internal:
                    raise ValueError(
                        f"The turbine type: {t} exists in both the internal and external"
                        " turbine library."
                    )

                if in_internal:
                    full_path = internal_fn
                elif in_external:
                    full_path = external_fn
                else:
                    raise FileNotFoundError(
                        f"The turbine type: {t} does not exist in either the internal or"
                        " external turbine library."
                    )
                turbine_definition_cache[t] = load_yaml(full_path)

        # Convert any dict entries in the turbine_type list to the type string. Since the
        # definition is saved above, we can make the whole list consistent now to use it
        # for mapping turbines later.
        # We use a private variable here instead of self.turbine_type because self.turbine_type
        # should always retain the input data. When this class is exported as_dict, the input
        # types must be used. If we modify that directly and change its shape, recreating this
        # class with a different layout but not a new self.turbine_type could cause the data
        # to be out of sync.
        _turbine_types = [
            copy.deepcopy(t["turbine_type"]) if isinstance(t, dict) else t
            for t in self.turbine_type
        ]

        # If 1 turbine definition is given, expand to N turbines; this covers a 1-turbine
        # farm and 1 definition for multiple turbines
        if len(_turbine_types) == 1:
            _turbine_types *= self.n_turbines

        # Map each turbine definition to its index in this list
        self.turbine_definitions = [
            copy.deepcopy(turbine_definition_cache[t]) for t in _turbine_types
        ]

    @layout_x.validator
    def check_x(self, attribute: attrs.Attribute, value: Any) -> None:
        if len(value) != len(self.layout_y):
            raise ValueError("layout_x and layout_y must have the same number of entries.")

    @layout_y.validator
    def check_y(self, attribute: attrs.Attribute, value: Any) -> None:
        if len(value) != len(self.layout_x):
            raise ValueError("layout_x and layout_y must have the same number of entries.")

    @turbine_type.validator
    def check_turbine_type(self, attribute: attrs.Attribute, value: Any) -> None:
        # Check that the list of turbines is either of length 1 or N turbines
        if len(value) != 1 and len(value) != self.n_turbines:
            raise ValueError(
                "turbine_type must have the same number of entries as layout_x/layout_y or have "
                "a single turbine_type value."
            )

    @turbine_library_path.validator
    def check_library_path(self, attribute: attrs.Attribute, value: Path) -> None:
        """Ensures that the input to `library_path` exists and is a directory."""
        if not value.is_dir():
            raise FileExistsError(f"The input file path: {str(value)} is not a valid directory.")

    def initialize(self, sorted_indices):
        # Sort yaw angles from most upstream to most downstream wind turbine
        self.yaw_angles_sorted = np.take_along_axis(
            self.yaw_angles,
            sorted_indices[:, :, :, 0, 0],
            axis=2,
        )
        self.tilt_angles_sorted = np.take_along_axis(
            self.tilt_angles,
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

    def construct_turbine_pPs(self):
        self.pPs = np.array([turb['pP'] for turb in self.turbine_definitions])

    def construct_turbine_pTs(self):
        self.pTs = np.array([turb['pT'] for turb in self.turbine_definitions])

    def construct_turbine_ref_density_cp_cts(self):
        self.ref_density_cp_cts = np.array([
            turb['ref_density_cp_ct'] for turb in self.turbine_definitions
        ])

    def construct_turbine_ref_tilt_cp_cts(self):
        self.ref_tilt_cp_cts = np.array(
            [turb['ref_tilt_cp_ct'] for turb in self.turbine_definitions]
        )

    def construct_turbine_correct_cp_ct_for_tilt(self):
        self.correct_cp_ct_for_tilt = np.array(
            [turb.correct_cp_ct_for_tilt for turb in self.turbine_map]
        )

    def construct_turbine_map(self):
        self.turbine_map = [Turbine.from_dict(turb) for turb in self.turbine_definitions]

    def construct_turbine_fCts(self):
        self.turbine_fCts = {
            turb.turbine_type: turb.fCt_interp for turb in self.turbine_map
        }

    def construct_turbine_fTilts(self):
        self.turbine_fTilts = [(turb.turbine_type, turb.fTilt_interp) for turb in self.turbine_map]

    def construct_turbine_power_interps(self):
        self.turbine_power_interps = {
            turb.turbine_type: turb.power_interp for turb in self.turbine_map
        }

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
        self.ref_density_cp_cts_sorted = np.take_along_axis(
            self.ref_density_cp_cts * template_shape,
            sorted_coord_indices,
            axis=2
        )
        self.ref_tilt_cp_cts_sorted = np.take_along_axis(
            self.ref_tilt_cp_cts * template_shape,
            sorted_coord_indices,
            axis=2
        )
        self.correct_cp_ct_for_tilt_sorted = np.take_along_axis(
            self.correct_cp_ct_for_tilt * template_shape,
            sorted_coord_indices,
            axis=2
        )
        self.pPs_sorted = np.take_along_axis(
            self.pPs * template_shape,
            sorted_coord_indices,
            axis=2
        )
        self.pTs_sorted = np.take_along_axis(
            self.pTs * template_shape,
            sorted_coord_indices,
            axis=2
        )

        # NOTE: Tilt angles are sorted twice - here and in initialize()
        self.tilt_angles_sorted = np.take_along_axis(
            self.tilt_angles * template_shape,
            sorted_coord_indices,
            axis=2
        )
        self.turbine_type_map_sorted = np.take_along_axis(
            np.reshape(
                [turb["turbine_type"] for turb in self.turbine_definitions] * n_wind_directions,
                np.shape(sorted_coord_indices)
            ),
            sorted_coord_indices,
            axis=2
        )

    def set_yaw_angles(self, n_wind_directions: int, n_wind_speeds: int):
        # TODO Is this just for initializing yaw angles to zero?
        self.yaw_angles = np.zeros((n_wind_directions, n_wind_speeds, self.n_turbines))
        self.yaw_angles_sorted = np.zeros((n_wind_directions, n_wind_speeds, self.n_turbines))

    def set_tilt_to_ref_tilt(self, n_wind_directions: int, n_wind_speeds: int):
        self.tilt_angles = (
            np.ones((n_wind_directions, n_wind_speeds, self.n_turbines))
            * self.ref_tilt_cp_cts
        )
        self.tilt_angles_sorted = (
            np.ones((n_wind_directions, n_wind_speeds, self.n_turbines))
            * self.ref_tilt_cp_cts
        )

    def calculate_tilt_for_eff_velocities(self, rotor_effective_velocities):
        tilt_angles = compute_tilt_angles_for_floating_turbines(
            self.turbine_type_map_sorted,
            self.tilt_angles_sorted,
            self.turbine_fTilts,
            rotor_effective_velocities,
        )
        return tilt_angles

    def finalize(self, unsorted_indices):
        self.yaw_angles = np.take_along_axis(
            self.yaw_angles_sorted,
            unsorted_indices[:,:,:,0,0],
            axis=2
        )
        self.tilt_angles = np.take_along_axis(
            self.tilt_angles_sorted,
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
        # TODO: do these need to be unsorted? Maybe we should just for completeness...
        self.ref_density_cp_cts = np.take_along_axis(
            self.ref_density_cp_cts_sorted,
            unsorted_indices[:,:,:,0,0],
            axis=2
        )
        self.ref_tilt_cp_cts = np.take_along_axis(
            self.ref_tilt_cp_cts_sorted,
            unsorted_indices[:,:,:,0,0],
            axis=2
        )
        self.correct_cp_ct_for_tilt = np.take_along_axis(
            self.correct_cp_ct_for_tilt_sorted,
            unsorted_indices[:,:,:,0,0],
            axis=2
        )
        self.pPs = np.take_along_axis(
            self.pPs_sorted,
            unsorted_indices[:,:,:,0,0],
            axis=2
        )
        self.pTs = np.take_along_axis(
            self.pTs_sorted,
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
