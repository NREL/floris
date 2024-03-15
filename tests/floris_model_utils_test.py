
from pathlib import Path

from floris import FlorisModel
from floris.floris_model_utils import (
    get_fmodel_param,
    get_power_thrust_model,
    nested_get,
    nested_set,
    set_fmodel_param,
    set_power_thrust_model,
)


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"

def test_nested_get():
    example_dict = {
        'a': {
            'b': {
                'c': 10
            }
        }
    }

    assert nested_get(example_dict, ['a', 'b', 'c']) == 10

def test_nested_set():
    example_dict = {
        'a': {
            'b': {
                'c': 10
            }
        }
    }

    nested_set(example_dict, ['a', 'b', 'c'], 20)
    assert nested_get(example_dict, ['a', 'b', 'c']) == 20


def test_get_and_set_fmodel_param():


    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Get the wind speed
    wind_speeds = get_fmodel_param(fmodel, ['flow_field', 'wind_speeds'])
    assert wind_speeds[0] == 8.0

    # Set the wind speed
    fmodel = set_fmodel_param(fmodel, ['flow_field', 'wind_speeds'], 10.0, param_idx=0)
    wind_speed = get_fmodel_param(fmodel, ['flow_field', 'wind_speeds'], param_idx=0  )
    assert wind_speed == 10.0

    # Repeat with wake parameter
    fmodel = set_fmodel_param(fmodel, ['wake', 'wake_velocity_parameters', 'gauss', 'alpha'], 0.1)
    alpha = get_fmodel_param(fmodel, ['wake', 'wake_velocity_parameters', 'gauss', 'alpha'])
    assert alpha == 0.1

def test_get_power_thrust_model():
    fmodel = FlorisModel(configuration=YAML_INPUT)
    assert get_power_thrust_model(fmodel) == "cosine-loss"

def test_set_power_thrust_model():

    fmodel = FlorisModel(configuration=YAML_INPUT)
    fmodel = set_power_thrust_model(fmodel, "simple-derating")
    assert get_power_thrust_model(fmodel) == "simple-derating"
