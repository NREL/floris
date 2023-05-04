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

import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from floris.simulation import Farm
from floris.utilities import load_yaml, Vec3
from tests.conftest import (
    N_TURBINES,
    N_WIND_DIRECTIONS,
    N_WIND_SPEEDS,
    SampleInputs,
)


def test_farm_init_homogenous_turbines():
    farm_data = SampleInputs().farm
    turbine_data = SampleInputs().turbine

    layout_x = farm_data["layout_x"]
    layout_y = farm_data["layout_y"]
    coordinates = np.array([
        Vec3([x, y, turbine_data["hub_height"]])
        for x, y in zip(layout_x, layout_y)
    ])

    farm = Farm(
        layout_x=layout_x,
        layout_y=layout_y,
        turbine_type=[turbine_data]
    )
    # TODO: these all pass on mac and fail on linux
    # turbine_type=[turbine_data]
    # turbine_type=[turbine_data["turbine_type"]]

    farm.construct_hub_heights()
    farm.construct_coordinates()
    farm.set_yaw_angles(N_WIND_DIRECTIONS, N_WIND_SPEEDS)

    # Check initial values
    np.testing.assert_array_equal(farm.coordinates, coordinates)
    assert isinstance(farm.layout_x, np.ndarray)
    assert isinstance(farm.layout_y, np.ndarray)


def test_asdict(sample_inputs_fixture: SampleInputs):
    farm = Farm.from_dict(sample_inputs_fixture.farm)
    farm.construct_hub_heights()
    farm.construct_coordinates()
    farm.construct_turbine_ref_tilt_cp_cts()
    farm.set_yaw_angles(N_WIND_DIRECTIONS, N_WIND_SPEEDS)
    farm.set_tilt_to_ref_tilt(N_WIND_DIRECTIONS, N_WIND_SPEEDS)
    dict1 = farm.as_dict()

    new_farm = farm.from_dict(dict1)
    new_farm.construct_hub_heights()
    new_farm.construct_coordinates()
    new_farm.construct_turbine_ref_tilt_cp_cts()
    new_farm.set_yaw_angles(N_WIND_DIRECTIONS, N_WIND_SPEEDS)
    new_farm.set_tilt_to_ref_tilt(N_WIND_DIRECTIONS, N_WIND_SPEEDS)
    dict2 = new_farm.as_dict()

    assert dict1 == dict2


def test_farm_external_library(sample_inputs_fixture: SampleInputs):
    external_library = Path(__file__).parent / "data"

    # Demonstrate a passing case
    farm_data = deepcopy(SampleInputs().farm)
    farm_data["turbine_library_path"] = external_library
    farm_data["turbine_type"] = ["nrel_5MW_custom"] * N_TURBINES
    farm = Farm.from_dict(farm_data)
    assert farm.turbine_library_path == external_library

    # Demonstrate a file not existing in the user library, but exists in the internal library, so
    # the loading is successful
    farm_data["turbine_library_path"] = external_library
    farm_data["turbine_type"] = ["iea_10MW"] * N_TURBINES
    farm = Farm.from_dict(farm_data)
    assert farm.turbine_definitions[0]["turbine_type"] == "iea_10MW"

    # Demonstrate a failing case with an incorrect library location
    farm_data["turbine_library_path"] = external_library / "turbine_library_path"
    with pytest.raises(FileExistsError):
        Farm.from_dict(farm_data)

    # Demonstrate a failing case where there is a duplicated turbine between the internal
    # and external turbine libraries
    farm_data = deepcopy(SampleInputs().farm)
    farm_data["turbine_library_path"] = external_library
    farm_data["turbine_type"] = ["nrel_5MW"] * N_TURBINES
    with pytest.raises(ValueError):
        Farm.from_dict(farm_data)

    # Demonstrate a failing case where there a turbine does not exist in either
    farm_data = deepcopy(SampleInputs().farm)
    farm_data["turbine_library_path"] = external_library
    farm_data["turbine_type"] = ["FAKE_TURBINE"] * N_TURBINES
    with pytest.raises(FileNotFoundError):
        Farm.from_dict(farm_data)
