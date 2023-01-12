
import numpy as np
from pathlib import Path

from floris.tools.floris_interface import FlorisInterface


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full_v3.yaml"
JSON_INPUT = TEST_DATA / "input_full_v3.json"


def test_read_yaml():
    fi = FlorisInterface(configuration=YAML_INPUT)
    assert isinstance(fi, FlorisInterface)


def test_calculate_wake():

    """
    In FLORIS v3.2, running calculate_wake twice incorrectly set the yaw angles when the first time
    has non-zero yaw settings but the second run had all-zero yaw settings. The test below asserts
    that the yaw angles are correctly set in subsequent calls to calculate_wake.
    """
    fi = FlorisInterface(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones((fi.floris.flow_field.n_wind_directions, fi.floris.flow_field.n_wind_speeds, fi.floris.farm.n_turbines))
    fi.calculate_wake(yaw_angles=yaw_angles)
    assert fi.floris.farm.yaw_angles == yaw_angles

    yaw_angles = np.zeros((fi.floris.flow_field.n_wind_directions, fi.floris.flow_field.n_wind_speeds, fi.floris.farm.n_turbines))
    fi.calculate_wake(yaw_angles=yaw_angles)
    assert fi.floris.farm.yaw_angles == yaw_angles


def test_reinitialize():
    pass

