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


from pathlib import Path

import yaml

from floris.simulation import (
    Farm,
    Floris,
    FlowField,
    TurbineGrid,
    WakeModelManager,
)


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full_v3.yaml"
DICT_INPUT = yaml.load(open(YAML_INPUT, "r"), Loader=yaml.SafeLoader)


def test_read_yaml():
    fi = Floris.from_file(YAML_INPUT)
    assert isinstance(fi, Floris)


def test_read_dict():
    fi = Floris.from_dict(DICT_INPUT)
    assert isinstance(fi, Floris)


def test_init():
    fi = Floris.from_dict(DICT_INPUT)
    assert isinstance(fi.farm, Farm)
    assert isinstance(fi.wake, WakeModelManager)
    assert isinstance(fi.flow_field, FlowField)


def test_asdict(turbine_grid_fixture: TurbineGrid):

    floris = Floris.from_dict(DICT_INPUT)
    floris.flow_field.initialize_velocity_field(turbine_grid_fixture)
    dict1 = floris.as_dict()

    new_floris = Floris.from_dict(dict1)
    new_floris.flow_field.initialize_velocity_field(turbine_grid_fixture)
    dict2 = new_floris.as_dict()

    assert dict1 == dict2
