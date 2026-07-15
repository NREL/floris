
from pathlib import Path

import numpy as np
import pytest

from floris.core import (
    Turbine,
)
from floris.core.turbine.operation_models import POWER_SETPOINT_DEFAULT
from floris.core.turbine.turbine import (
    axial_induction,
    power,
    thrust_coefficient,
)
from tests.conftest import SampleInputs, WIND_SPEEDS


# size 16 x 1 x 1 x 1
# 16 wind speed and wind direction combinations from conftest
WIND_CONDITION_BROADCAST = np.reshape(np.array(WIND_SPEEDS), (-1, 1, 1, 1))

INDEX_FILTER = [0, 2]


def test_turbine_init():
    turbine_data = SampleInputs().turbine_multi_dim
    turbine = Turbine.from_dict(turbine_data)
    condition_tuple = (2, 1)
    assert turbine.rotor_diameter == turbine_data["rotor_diameter"]
    assert turbine.hub_height == turbine_data["hub_height"]
    assert (
        turbine.power_thrust_table[condition_tuple]["cosine_loss_exponent_yaw"]
        == turbine_data["power_thrust_table"]["cosine_loss_exponent_yaw"]
    )
    assert (
        turbine.power_thrust_table[condition_tuple]["cosine_loss_exponent_tilt"]
        == turbine_data["power_thrust_table"]["cosine_loss_exponent_tilt"]
    )

    assert isinstance(turbine.power_thrust_table, dict)
    assert callable(turbine.thrust_coefficient_function)
    assert callable(turbine.power_function)
    assert turbine.rotor_radius == turbine_data["rotor_diameter"] / 2.0


def test_ct():
    N_TURBINES = 4

    turbine_data = SampleInputs().turbine_multi_dim
    turbine = Turbine.from_dict(turbine_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]
    condition = {"Tp":2, "Hs":1}

    # Force a 5 degree yaw angle
    turbine.ref_tilt = 5.0
    turbine.correct_cp_ct_for_tilt = False

    # Single turbine
    # yaw angle / fCt are (n wind direction, n wind speed, n turbine)
    wind_speed = 10.0
    thrust = thrust_coefficient(
        turbines=[turbine] * N_TURBINES,
        velocities=wind_speed * np.ones((1, 1, 3, 3)),
        turbulence_intensities=0.06 * np.ones((1, 1, 3, 3)),
        air_density=None,
        yaw_angles=np.zeros((1, 1)),
        power_setpoints=np.ones((1, 1)) * POWER_SETPOINT_DEFAULT,\
        awc_modes=np.array([["baseline"]*N_TURBINES]*1),
        awc_amplitudes=np.zeros((1, 1)),
        turbine_type_map=turbine_type_map[:,0],
        multidim_condition=condition
    )

    np.testing.assert_allclose(thrust, np.array([[0.77958497]]))

    # Multiple turbines with index filter
    # 4 turbines with 3 x 3 grid arrays
    thrusts = thrust_coefficient(
        turbines=[turbine] * N_TURBINES,
        velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 16 x 4 x 3 x 3
        turbulence_intensities=(
            0.06 * np.ones((N_TURBINES, 3, 3))
            * np.ones_like(WIND_CONDITION_BROADCAST)
        ),
        air_density=None,
        yaw_angles=np.zeros((1, N_TURBINES)),
        power_setpoints=np.ones((1, N_TURBINES)) * POWER_SETPOINT_DEFAULT,
        awc_modes=np.array([["baseline"]*N_TURBINES]*1),
        awc_amplitudes=np.zeros((1, N_TURBINES)),
        turbine_type_map=turbine_type_map,
        ix_filter=INDEX_FILTER,
        multidim_condition=condition
    )
    assert len(thrusts[0]) == len(INDEX_FILTER)

    thrusts_truth = np.array(
        [
            [0.77958497, 0.77958497],
            [0.77958497, 0.77958497],
            [0.77958497, 0.77958497],
            [0.66749069, 0.66749069],
            [0.77958497, 0.77958497],
            [0.77958497, 0.77958497],
            [0.77958497, 0.77958497],
            [0.66749069, 0.66749069],
            [0.77958497, 0.77958497],
            [0.77958497, 0.77958497],
            [0.77958497, 0.77958497],
            [0.66749069, 0.66749069],
            [0.77958497, 0.77958497],
            [0.77958497, 0.77958497],
            [0.77958497, 0.77958497],
            [0.66749069, 0.66749069]
        ]
    )
    np.testing.assert_allclose(thrusts, thrusts_truth)

def test_power():
    N_TURBINES = 4
    AIR_DENSITY = 1.225

    turbine_data = SampleInputs().turbine_multi_dim
    turbine = Turbine.from_dict(turbine_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]
    condition = {"Tp":2, "Hs":1}
    condition_tuple = tuple(condition[k] for k in condition.keys())

    # Use reference tilt angle for testing power
    turbine.correct_cp_ct_for_tilt = False

    # Single turbine
    wind_speed = 10.0
    p = power(
        turbines=[turbine] * N_TURBINES,
        velocities=wind_speed * np.ones((1, 1, 3, 3)),
        turbulence_intensities=0.06 * np.ones((1, 1, 3, 3)),
        air_density=AIR_DENSITY,
        yaw_angles=np.zeros((1, 1)), # 1 findex, 1 turbine
        power_setpoints=np.ones((1, 1)) * POWER_SETPOINT_DEFAULT,
        awc_modes=np.array([["baseline"]*N_TURBINES]*1),
        awc_amplitudes=np.zeros((1, 1)),
        turbine_type_map=turbine_type_map[:,0],
        multidim_condition=condition,
    )

    power_truth = 12424759.67683091

    np.testing.assert_allclose(p, power_truth)

    # Multiple turbines with ix filter, switch to different tilt angle
    turbine.ref_tilt = 5.0
    velocities = np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST
    p = power(
        turbines=[turbine] * N_TURBINES,
        velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 16 x 4 x 3 x 3
        turbulence_intensities=(
            0.06 * np.ones((N_TURBINES, 3, 3))
            * np.ones_like(WIND_CONDITION_BROADCAST)
        ),
        air_density=AIR_DENSITY,
        yaw_angles=np.zeros((1, N_TURBINES)),
        power_setpoints=np.ones((1, N_TURBINES)) * POWER_SETPOINT_DEFAULT,
        awc_modes=np.array([["baseline"]*N_TURBINES]*1),
        awc_amplitudes=np.zeros((1, N_TURBINES)),
        turbine_type_map=turbine_type_map,
        ix_filter=INDEX_FILTER,
        multidim_condition=condition
    )
    assert len(p[0]) == len(INDEX_FILTER)

    power_truth = turbine.power_function(
        power_thrust_table=turbine.power_thrust_table[condition_tuple],
        velocities=velocities,
        air_density=AIR_DENSITY,
        yaw_angles=np.zeros((1, N_TURBINES)),
        tilt_angles=np.ones((1, N_TURBINES)) * 5.0,
        tilt_interp=turbine.tilt_interp,
    )
    np.testing.assert_allclose(p, power_truth[:, INDEX_FILTER[0]:INDEX_FILTER[1]])


def test_axial_induction():

    N_TURBINES = 4

    turbine_data = SampleInputs().turbine_multi_dim
    turbine = Turbine.from_dict(turbine_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]
    condition = {"Tp":2, "Hs":1}

    # Force a 5 degree yaw angle
    turbine.ref_tilt = 5.0
    turbine.correct_cp_ct_for_tilt = False

    baseline_ai = np.array([[0.26551081]])

    # Single turbine
    wind_speed = 10.0
    ai = axial_induction(
        turbines=[turbine] * N_TURBINES,
        velocities=wind_speed * np.ones((1, 1, 3, 3)),
        turbulence_intensities=0.06 * np.ones((1, 1, 3, 3)),
        air_density=None,
        yaw_angles=np.zeros((1, 1)),
        power_setpoints = np.ones((1, 1)) * POWER_SETPOINT_DEFAULT,
        awc_modes=np.array([["baseline"]*N_TURBINES]*1),
        awc_amplitudes=np.zeros((1, 1)),
        turbine_type_map=turbine_type_map[0,0],
        multidim_condition=condition
    )
    np.testing.assert_allclose(ai, baseline_ai)

    # Multiple turbines with ix filter
    ai = axial_induction(
        turbines=[turbine] * N_TURBINES,
        velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 16 x 4 x 3 x 3
        turbulence_intensities=(
            0.06 * np.ones((N_TURBINES, 3, 3))
            * np.ones_like(WIND_CONDITION_BROADCAST)
        ),
        air_density=None,
        yaw_angles=np.zeros((1, N_TURBINES)),
        power_setpoints=np.ones((1, N_TURBINES)) * POWER_SETPOINT_DEFAULT,
        awc_modes=np.array([["baseline"]*N_TURBINES]*1),
        awc_amplitudes=np.zeros((1, N_TURBINES)),
        turbine_type_map=turbine_type_map,
        ix_filter=INDEX_FILTER,
        multidim_condition=condition
    )

    assert len(ai[0]) == len(INDEX_FILTER)

    # Test the 10 m/s wind speed to use the same baseline as above
    np.testing.assert_allclose(ai[2][0], baseline_ai)


def test_asdict(sample_inputs_fixture: SampleInputs):

    turbine = Turbine.from_dict(sample_inputs_fixture.turbine)
    dict1 = turbine.as_dict()

    new_turb = Turbine.from_dict(dict1)
    dict2 = new_turb.as_dict()

    assert dict1 == dict2

def test_multiple_conditions():

    N_TURBINES = 4
    N_CONDITIONS = 2

    turbine_data = SampleInputs().turbine_multi_dim
    turbine = Turbine.from_dict(turbine_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]

    # Force a 5 degree yaw angle
    ref_tilt_orig = turbine.power_thrust_table[(2,1)]["ref_tilt"]
    ref_tilt_test = 5.0
    turbine.ref_tilt = ref_tilt_test
    turbine.correct_cp_ct_for_tilt = False

    # First, test the same condition repeated
    conditions = {"Tp":[2, 2], "Hs":[1, 1]}

    # Single turbine
    wind_speed = 10.0
    thrust = thrust_coefficient(
        turbines=[turbine] * N_TURBINES,
        velocities=wind_speed * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        turbulence_intensities=0.06 * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        air_density=None,
        yaw_angles=np.zeros((N_CONDITIONS, N_TURBINES)),
        power_setpoints=np.ones((N_CONDITIONS, N_TURBINES)) * POWER_SETPOINT_DEFAULT,
        awc_modes=np.array([["baseline"]*N_TURBINES]*N_CONDITIONS),
        awc_amplitudes=np.zeros((N_CONDITIONS, N_TURBINES)),
        turbine_type_map=turbine_type_map,
        multidim_condition=conditions
    )
    assert np.allclose(thrust, 0.77958497)

    ai = axial_induction(
        turbines=[turbine] * N_TURBINES,
        velocities=wind_speed * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        turbulence_intensities=0.06 * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        air_density=None,
        yaw_angles=np.zeros((N_CONDITIONS, N_TURBINES)),
        power_setpoints=np.ones((N_CONDITIONS, N_TURBINES)) * POWER_SETPOINT_DEFAULT,
        awc_modes=np.array([["baseline"]*N_TURBINES]*N_CONDITIONS),
        awc_amplitudes=np.zeros((N_CONDITIONS, N_TURBINES)),
        turbine_type_map=turbine_type_map,
        multidim_condition=conditions
    )
    assert np.allclose(ai, 0.26551081)

    # Set original reference tilt angle
    turbine.ref_tilt = ref_tilt_orig
    p = power(
        turbines=[turbine] * N_TURBINES,
        velocities=wind_speed * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        turbulence_intensities=0.06 * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        air_density=1.225,
        yaw_angles=np.zeros((N_CONDITIONS, N_TURBINES)),
        power_setpoints=np.ones((N_CONDITIONS, N_TURBINES)) * POWER_SETPOINT_DEFAULT,
        awc_modes=np.array([["baseline"]*N_TURBINES]*N_CONDITIONS),
        awc_amplitudes=np.zeros((N_CONDITIONS, N_TURBINES)),
        turbine_type_map=turbine_type_map,
        multidim_condition=conditions,
    )
    assert np.allclose(p, 12424759.67683091)

    # Next, test different conditions (one which must be inferred)
    turbine.ref_tilt = ref_tilt_test
    conditions = {"Tp":[2, 4], "Hs":[1, 4]}
    thrust = thrust_coefficient(
        turbines=[turbine] * N_TURBINES,
        velocities=wind_speed * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        turbulence_intensities=0.06 * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        air_density=None,
        yaw_angles=np.zeros((N_CONDITIONS, N_TURBINES)),
        power_setpoints=np.ones((N_CONDITIONS, N_TURBINES)) * POWER_SETPOINT_DEFAULT,
        awc_modes=np.array([["baseline"]*N_TURBINES]*N_CONDITIONS),
        awc_amplitudes=np.zeros((N_CONDITIONS, N_TURBINES)),
        turbine_type_map=turbine_type_map,
        multidim_condition=conditions
    )
    assert np.allclose(thrust, np.array([[0.77958497], [0.09744812]]))

    ai = axial_induction(
        turbines=[turbine] * N_TURBINES,
        velocities=wind_speed * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        turbulence_intensities=0.06 * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        air_density=None,
        yaw_angles=np.zeros((N_CONDITIONS, N_TURBINES)),
        power_setpoints=np.ones((N_CONDITIONS, N_TURBINES)) * POWER_SETPOINT_DEFAULT,
        awc_modes=np.array([["baseline"]*N_TURBINES]*N_CONDITIONS),
        awc_amplitudes=np.zeros((N_CONDITIONS, N_TURBINES)),
        turbine_type_map=turbine_type_map,
        multidim_condition=conditions
    )
    assert np.allclose(ai, np.array([[0.26551081], [0.02498745]]))

    turbine.ref_tilt = ref_tilt_orig
    p = power(
        turbines=[turbine]*N_TURBINES,
        velocities=wind_speed * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        turbulence_intensities=0.06 * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        air_density=1.225,
        yaw_angles=np.zeros((N_CONDITIONS, N_TURBINES)),
        power_setpoints=np.ones((N_CONDITIONS, N_TURBINES)) * POWER_SETPOINT_DEFAULT,
        awc_modes=np.array([["baseline"]*N_TURBINES]*N_CONDITIONS),
        awc_amplitudes=np.zeros((N_CONDITIONS, N_TURBINES)),
        turbine_type_map=turbine_type_map,
        multidim_condition=conditions,
    )
    assert np.allclose(p, np.array([[12424759.67683091], [ 1553094.95985386]]))

    # Multiple findices with broadcast multidim conditions
    wind_speeds = np.array([10., 11.])
    conditions = {"Tp":2, "Hs":1}
    turbine.ref_tilt = ref_tilt_test
    thrust = thrust_coefficient(
        turbines=[turbine] * N_TURBINES,
        velocities=np.tile(wind_speeds[:,None,None,None], (1, N_TURBINES, 3, 3)),
        turbulence_intensities=0.06 * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        air_density=None,
        yaw_angles=np.zeros((N_CONDITIONS, N_TURBINES)),
        power_setpoints=np.ones((N_CONDITIONS, N_TURBINES)) * POWER_SETPOINT_DEFAULT,
        awc_modes=np.array([["baseline"]*N_TURBINES]*N_CONDITIONS),
        awc_amplitudes=np.zeros((N_CONDITIONS, N_TURBINES)),
        turbine_type_map=turbine_type_map,
        multidim_condition=conditions
    )
    assert np.allclose(thrust, np.array([[0.77958497], [0.66749069]]))

    ai = axial_induction(
        turbines=[turbine] * N_TURBINES,
        velocities=np.tile(wind_speeds[:,None,None,None], (1, N_TURBINES, 3, 3)),
        turbulence_intensities=0.06 * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        air_density=None,
        yaw_angles=np.zeros((N_CONDITIONS, N_TURBINES)),
        power_setpoints=np.ones((N_CONDITIONS, N_TURBINES)) * POWER_SETPOINT_DEFAULT,
        awc_modes=np.array([["baseline"]*N_TURBINES]*N_CONDITIONS),
        awc_amplitudes=np.zeros((N_CONDITIONS, N_TURBINES)),
        turbine_type_map=turbine_type_map,
        multidim_condition=conditions
    )
    assert np.allclose(ai, np.array([[0.26551081], [0.2118128]]))

    turbine.ref_tilt = ref_tilt_orig
    p = power(
        turbines=[turbine]*N_TURBINES,
        velocities=np.tile(wind_speeds[:,None,None,None], (1, N_TURBINES, 3, 3)),
        turbulence_intensities=0.06 * np.ones((N_CONDITIONS, N_TURBINES, 3, 3)),
        air_density=1.225,
        yaw_angles=np.zeros((N_CONDITIONS, N_TURBINES)),
        power_setpoints=np.ones((N_CONDITIONS, N_TURBINES)) * POWER_SETPOINT_DEFAULT,
        awc_modes=np.array([["baseline"]*N_TURBINES]*N_CONDITIONS),
        awc_amplitudes=np.zeros((N_CONDITIONS, N_TURBINES)),
        turbine_type_map=turbine_type_map,
        multidim_condition=conditions,
    )
    assert np.allclose(p, np.array([[12424759.67683091], [15000000.0]]))
