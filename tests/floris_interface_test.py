
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
    yaw_angles = 20 * np.ones(
        (
            fi.floris.flow_field.n_wind_directions,
            fi.floris.flow_field.n_wind_speeds,
            fi.floris.farm.n_turbines
        )
    )
    fi.calculate_wake(yaw_angles=yaw_angles)
    assert fi.floris.farm.yaw_angles == yaw_angles

    yaw_angles = np.zeros(
        (
            fi.floris.flow_field.n_wind_directions,
            fi.floris.flow_field.n_wind_speeds,
            fi.floris.farm.n_turbines
        )
    )
    fi.calculate_wake(yaw_angles=yaw_angles)
    assert fi.floris.farm.yaw_angles == yaw_angles


def test_calculate_no_wake():
    """
    In FLORIS v3.2, running calculate_no_wake twice incorrectly set the yaw angles when the first
    time has non-zero yaw settings but the second run had all-zero yaw settings. The test below
    asserts that the yaw angles are correctly set in subsequent calls to calculate_wake.
    """
    fi = FlorisInterface(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones(
        (
            fi.floris.flow_field.n_wind_directions,
            fi.floris.flow_field.n_wind_speeds,
            fi.floris.farm.n_turbines
        )
    )
    fi.calculate_no_wake(yaw_angles=yaw_angles)
    assert fi.floris.farm.yaw_angles == yaw_angles

    yaw_angles = np.zeros(
        (
            fi.floris.flow_field.n_wind_directions,
            fi.floris.flow_field.n_wind_speeds,
            fi.floris.farm.n_turbines
        )
    )
    fi.calculate_no_wake(yaw_angles=yaw_angles)
    assert fi.floris.farm.yaw_angles == yaw_angles

    # check finalized values of calculate_no_wake
    fi_base = FlorisInterface(configuration=YAML_INPUT)
    yaw_angles = np.zeros(
        (
            fi_base.floris.flow_field.n_wind_directions,
            fi_base.floris.flow_field.n_wind_speeds,
            fi_base.floris.farm.n_turbines
        )
    )
    fi_base.floris.farm.yaw_angles = yaw_angles
    fi_base.floris.flow_field.initialize_velocity_field(fi_base.floris.grid)
    fi_base.floris.farm.initialize(fi_base.floris.grid.sorted_indices)
    fi_base.floris.state.INITIALIZED
    fi_base.floris.finalize()

    fi = FlorisInterface(configuration=YAML_INPUT)
    fi.calculate_no_wake()

    # check flow_field finalized values
    assert np.allclose(fi_base.floris.flow_field.u, fi.floris.flow_field.u)
    assert np.allclose(fi_base.floris.flow_field.v, fi.floris.flow_field.v)
    assert np.allclose(fi_base.floris.flow_field.w, fi.floris.flow_field.w)
    assert np.allclose(
        fi_base.floris.flow_field.turbulence_intensity_field,
        fi.floris.flow_field.turbulence_intensity_field
    )

    # check farm finalized values
    assert np.allclose(fi_base.floris.farm.yaw_angles, fi.floris.farm.yaw_angles)
    assert np.allclose(fi_base.floris.farm.hub_heights, fi.floris.farm.hub_heights)
    assert np.allclose(fi_base.floris.farm.rotor_diameters, fi.floris.farm.rotor_diameters)
    assert np.allclose(fi_base.floris.farm.TSRs, fi.floris.farm.TSRs)
    assert np.allclose(fi_base.floris.farm.pPs, fi.floris.farm.pPs)
    assert fi_base.floris.farm.turbine_type_map == fi.floris.farm.turbine_type_map


def test_reinitialize():
    pass

