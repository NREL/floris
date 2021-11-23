from pathlib import Path

import yaml
import numpy as np

from floris.simulation.farm import Farm
from floris.simulation.wake import Wake
from floris.simulation.floris import Floris
from floris.simulation.turbine import Turbine
from floris.simulation.flow_field import FlowField
from floris.tools.floris_interface import FlorisInterface


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full_v3.yaml"
JSON_INPUT = TEST_DATA / "input_full_v3.json"
DICT_INPUT = yaml.load(open(YAML_INPUT, "r"), Loader=yaml.SafeLoader)


def test_read_json():
    fi = Floris.from_json(JSON_INPUT)
    assert isinstance(fi, Floris)


def test_read_yaml():
    fi = Floris.from_yaml(YAML_INPUT)
    assert isinstance(fi, Floris)


def test_read_dict():
    fi = Floris.from_dict(DICT_INPUT)
    assert isinstance(fi, Floris)


def test_init():
    fi = Floris.from_dict(DICT_INPUT)
    print(fi)
    assert isinstance(fi.farm, Farm)
    assert isinstance(fi.logging, dict)
    for turb in fi.turbine.values():
        assert isinstance(turb, Turbine)
    assert isinstance(fi.wake, Wake)
    assert isinstance(fi.flow_field, FlowField)


def test_prepare_for_save():
    # Need to define some __eq__ methods for this to work out correctly
    fi = Floris.from_dict(DICT_INPUT)
    new_input = fi._prepare_for_save()
    new_fi = Floris.from_dict(new_input)
    # assert fi == new_fi


def test_annual_energy_production():
    pass


def test_steady_state_atmospheric_condition():
    pass
