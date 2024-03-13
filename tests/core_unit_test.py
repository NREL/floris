
from pathlib import Path

import yaml

from floris.core import (
    Core,
    Farm,
    FlowField,
    TurbineGrid,
    WakeModelManager,
)


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"
DICT_INPUT = yaml.load(open(YAML_INPUT, "r"), Loader=yaml.SafeLoader)


def test_read_yaml():
    fmodel = Core.from_file(YAML_INPUT)
    assert isinstance(fmodel, Core)


def test_read_dict():
    fmodel = Core.from_dict(DICT_INPUT)
    assert isinstance(fmodel, Core)


def test_init():
    fmodel = Core.from_dict(DICT_INPUT)
    assert isinstance(fmodel.farm, Farm)
    assert isinstance(fmodel.wake, WakeModelManager)
    assert isinstance(fmodel.flow_field, FlowField)


def test_asdict(turbine_grid_fixture: TurbineGrid):

    floris = Core.from_dict(DICT_INPUT)
    floris.flow_field.initialize_velocity_field(turbine_grid_fixture)
    dict1 = floris.as_dict()

    new_floris = Core.from_dict(dict1)
    new_floris.flow_field.initialize_velocity_field(turbine_grid_fixture)
    dict2 = new_floris.as_dict()

    assert dict1 == dict2
