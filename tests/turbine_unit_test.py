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


import attr
import numpy as np
import pytest
from scipy.interpolate import interp1d

from floris.simulation import (
    average_velocity,
    axial_induction,
    Ct,
    power,
    Turbine,
)
from floris.simulation.turbine import (
    _filter_convert,
    _rotor_velocity_tilt_correction,
    _rotor_velocity_yaw_correction,
    compute_tilt_angles_for_floating_turbines,
    PowerThrustTable,
)
from tests.conftest import SampleInputs, WIND_SPEEDS


# size 3 x 4 x 1 x 1 x 1
WIND_CONDITION_BROADCAST = np.stack(
    (
        np.reshape(np.array(WIND_SPEEDS), (-1, 1, 1, 1)),  # Wind direction 0
        np.reshape(np.array(WIND_SPEEDS), (-1, 1, 1, 1)),  # Wind direction 1
        np.reshape(np.array(WIND_SPEEDS), (-1, 1, 1, 1)),  # Wind direction 2
    ),
    axis=0,
)
INDEX_FILTER = [0, 2]


def test_power_thrust_table():
    turbine_data = SampleInputs().turbine
    table = PowerThrustTable.from_dict(turbine_data["power_thrust_table"])

    # Test data conversion is correct
    assert isinstance(table.power, np.ndarray)
    assert isinstance(table.thrust, np.ndarray)
    assert isinstance(table.wind_speed, np.ndarray)

    # Test for initialization errors
    for el in ("power", "thrust", "wind_speed"):
        pt_table = SampleInputs().turbine["power_thrust_table"]
        pt_table[el] = pt_table[el][:-1]
        with pytest.raises(ValueError):
            PowerThrustTable.from_dict(pt_table)

        pt_table = SampleInputs().turbine["power_thrust_table"]
        pt_table[el] = np.array(pt_table[el]).reshape(2, -1)
        with pytest.raises(ValueError):
            PowerThrustTable.from_dict(pt_table)


def test_turbine_init():
    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)
    assert turbine.rotor_diameter == turbine_data["rotor_diameter"]
    assert turbine.hub_height == turbine_data["hub_height"]
    assert turbine.pP == turbine_data["pP"]
    assert turbine.pT == turbine_data["pT"]
    assert turbine.generator_efficiency == turbine_data["generator_efficiency"]

    pt_data = turbine_data["power_thrust_table"]
    assert isinstance(turbine.power_thrust_table, PowerThrustTable)
    np.testing.assert_allclose(
        turbine.power_thrust_table.power,
        np.array(pt_data["power"])
    )
    np.testing.assert_allclose(
        turbine.power_thrust_table.thrust,
        np.array(pt_data["thrust"])
    )
    np.testing.assert_allclose(
        turbine.power_thrust_table.wind_speed,
        np.array(pt_data["wind_speed"])
    )

    assert isinstance(turbine.fCp_interp, interp1d)
    assert isinstance(turbine.fCt_interp, interp1d)
    assert isinstance(turbine.power_interp, interp1d)
    assert turbine.rotor_radius == turbine_data["rotor_diameter"] / 2.0


def test_rotor_radius():

    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)

    # Test that the radius is set correctly from the input file
    assert turbine.rotor_radius == turbine_data["rotor_diameter"] / 2.0

    # Test the radius setter method since it actually sets the diameter
    turbine.rotor_radius = 200.0
    assert turbine.rotor_diameter == 400.0

    # Test the getter-method again
    assert turbine.rotor_radius == 200.0


def test_rotor_area():

    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)

    # Test that the area is set correctly from the input file
    assert turbine.rotor_area == np.pi * (turbine_data["rotor_diameter"] / 2.0) ** 2

    # Test the area setter method since it actually sets the radius and then the diameter
    turbine.rotor_area = np.pi
    assert turbine.rotor_radius == 1
    assert turbine.rotor_diameter == 2

    # Test the getter-method again
    assert turbine.rotor_area == np.pi


def test_filter_convert():
    N = 4

    # When the index filter is not None or a Numpy array,
    # the function should return None
    ix_filter = 1
    sample_arg = np.arange(N)
    with pytest.raises(TypeError):
        _filter_convert(ix_filter, sample_arg)

    # When the sample_arg is not a Numpy array, the function
    # should return None
    ix_filter = None
    sample_arg = [1, 2, 3]
    with pytest.raises(TypeError):
        _filter_convert(ix_filter, sample_arg)

    # When the sample_arg is a Numpy array and the index filter
    # is None, a boolean array containing all True should be
    # returned with the same shape as the sample_arg.
    ix_filter = None
    sample_arg = np.arange(N)
    ix_filter = _filter_convert(ix_filter, sample_arg)
    assert ix_filter.sum() == N
    assert ix_filter.shape == (N,)

    # When the index filter is given as a Python list, the
    # function should return the values cast to a Numpy array
    ix_filter = [1, 2]
    sample_arg = np.arange(N).reshape(1, 1, N)
    ix_filter = _filter_convert(ix_filter, sample_arg)
    np.testing.assert_array_equal(ix_filter, np.array([1, 2]))

    # Test that a 1-D boolean truth array is returned
    # When the index filter is None and the sample_arg
    # is a Numpy array of values, the returned filter indices
    # should be all True and have the shape of the turbine-dimension
    ix_filter = None
    sample_arg = np.arange(N).reshape(1, 1, N)
    ix_filter = _filter_convert(ix_filter, sample_arg)
    assert ix_filter.sum() == N
    assert ix_filter.shape == (N,)


def test_average_velocity():
    # TODO: why do we use cube root - mean - cube (like rms) instead of a simple average (np.mean)?
    # Dimensions are (n wind directions, n wind speeds, n turbines, grid x, grid y)
    velocities = np.ones((1, 1, 1, 5, 5))
    assert average_velocity(velocities, method="cubic-mean") == 1

    # Constructs an array of shape 1 x 1 x 2 x 3 x 3 with finrst turbie all 1, second turbine all 2
    velocities = np.stack(
        (
            np.ones((1, 1, 3, 3)),  # The first dimension here is the wind direction and the second
            2 * np.ones((1, 1, 3, 3)),  # is the wind speed since we are stacking on axis=2
        ),
        axis=2,
    )

    # Pull out the first wind speed for the test
    np.testing.assert_array_equal(
        average_velocity(velocities, method="cubic-mean")[0, 0],
        np.array([1, 2])
    )

    # Test boolean filter
    ix_filter = [True, False, True, False]
    velocities = np.stack(  # 4 turbines with 3 x 3 velocity array; shape (1,1,4,3,3)
        [i * np.ones((1, 1, 3, 3)) for i in range(1,5)],
        # (
        #     # The first dimension here is the wind direction
        #     # and second is the wind speed since we are stacking on axis=2
        #     np.ones(
        #         (1, 1, 3, 3)
        #     ),
        #     2 * np.ones((1, 1, 3, 3)),
        #     3 * np.ones((1, 1, 3, 3)),
        #     4 * np.ones((1, 1, 3, 3)),
        # ),
        axis=2,
    )
    avg = average_velocity(velocities, ix_filter, method="cubic-mean")
    assert avg.shape == (1, 1, 2)  # 1 wind direction, 1 wind speed, 2 turbines filtered

    # Pull out the first wind direction and wind speed for the comparison
    assert np.allclose(avg[0, 0], np.array([1.0, 3.0]))
    # This fails in GitHub Actions due to a difference in precision:
    # E           assert 3.0000000000000004 == 3.0
    # np.testing.assert_array_equal(avg[0], np.array([1.0, 3.0]))

    # Test integer array filter
    # np.arange(1, 5).reshape((-1,1,1)) * np.ones((1, 1, 3, 3))
    velocities = np.stack(  # 4 turbines with 3 x 3 velocity array; shape (1,1,4,3,3)
        [i * np.ones((1, 1, 3, 3)) for i in range(1,5)],
        axis=2,
    )
    avg = average_velocity(velocities, INDEX_FILTER, method="cubic-mean")
    assert avg.shape == (1, 1, 2)  # 1 wind direction, 1 wind speed, 2 turbines filtered

    # Pull out the first wind direction and wind speed for the comparison
    assert np.allclose(avg[0, 0], np.array([1.0, 3.0]))


def test_ct():
    N_TURBINES = 4

    turbine_data = SampleInputs().turbine
    turbine_floating_data = SampleInputs().turbine_floating
    turbine = Turbine.from_dict(turbine_data)
    turbine_floating = Turbine.from_dict(turbine_floating_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, None, :]

    # Single turbine
    # yaw angle / fCt are (n wind direction, n wind speed, n turbine)
    wind_speed = 10.0
    thrust = Ct(
        velocities=wind_speed * np.ones((1, 1, 1, 3, 3)),
        yaw_angle=np.zeros((1, 1, 1)),
        tilt_angle=np.ones((1, 1, 1)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, 1, 1)) * 5.0,
        fCt={turbine.turbine_type: turbine.fCt_interp},
        tilt_interp=np.array([(turbine.turbine_type, None)]),
        correct_cp_ct_for_tilt=np.array([[[False]]]),
        turbine_type_map=turbine_type_map[:,:,0]
    )

    truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    np.testing.assert_allclose(thrust, turbine_data["power_thrust_table"]["thrust"][truth_index])

    # Multiple turbines with index filter
    # 4 turbines with 3 x 3 grid arrays
    thrusts = Ct(
        velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 3 x 4 x 4 x 3 x 3
        yaw_angle=np.zeros((1, 1, N_TURBINES)),
        tilt_angle=np.ones((1, 1, N_TURBINES)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, 1, N_TURBINES)) * 5.0,
        fCt={turbine.turbine_type: turbine.fCt_interp},
        tilt_interp=np.array([(turbine.turbine_type, None)]),
        correct_cp_ct_for_tilt=np.array([[[False] * N_TURBINES]]),
        turbine_type_map=turbine_type_map,
        ix_filter=INDEX_FILTER,
    )
    assert len(thrusts[0, 0]) == len(INDEX_FILTER)

    for i in range(len(INDEX_FILTER)):
        truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(WIND_SPEEDS[0])
        np.testing.assert_allclose(
            thrusts[0, 0, i],
            turbine_data["power_thrust_table"]["thrust"][truth_index]
        )

    # Single floating turbine; note that 'tilt_interp' is not set to None
    thrust = Ct(
        velocities=wind_speed * np.ones((1, 1, 1, 3, 3)),
        yaw_angle=np.zeros((1, 1, 1)),
        tilt_angle=np.ones((1, 1, 1)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, 1, 1)) * 5.0,
        fCt={turbine.turbine_type: turbine_floating.fCt_interp},
        tilt_interp=np.array([(turbine_floating.turbine_type, turbine_floating.fTilt_interp)]),
        correct_cp_ct_for_tilt=np.array([[[True]]]),
        turbine_type_map=turbine_type_map[:,:,0]
    )

    truth_index = turbine_floating_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    np.testing.assert_allclose(
        thrust,
        turbine_floating_data["power_thrust_table"]["thrust"][truth_index]
    )


def test_power():
    N_TURBINES = 4
    AIR_DENSITY = 1.225

    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, None, :]

    # Single turbine
    wind_speed = 10.0
    p = power(
        ref_density_cp_ct=AIR_DENSITY,
        rotor_effective_velocities=wind_speed * np.ones((1, 1, 1, 3, 3)),
        power_interp={turbine.turbine_type: turbine.fCp_interp},
        turbine_type_map=turbine_type_map[:,:,0]
    )

    # calculate power again
    truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    cp_truth = turbine_data["power_thrust_table"]["power"][truth_index]
    power_truth = (
        0.5
        * turbine.rotor_area
        * cp_truth
        * turbine.generator_efficiency
        * wind_speed ** 3
    )
    np.testing.assert_allclose(p,cp_truth,power_truth )

    # # Multiple turbines with ix filter
    # p = power(
    #     air_density=AIR_DENSITY,
    #     velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 3 x 4 x 4 x 3 x 3
    #     yaw_angle=np.zeros((1, 1, N_TURBINES)),
    #     pP=turbine.pP * np.ones((3, 4, N_TURBINES)),
    #     power_interp={turbine.turbine_type: turbine.fCp_interp},
    #     turbine_type_map=turbine_type_map,
    #     ix_filter=INDEX_FILTER,
    # )
    # assert len(p[0, 0]) == len(INDEX_FILTER)

    # for i in range(len(INDEX_FILTER)):
    #     effective_velocity_trurth = ((AIR_DENSITY/1.225)**(1/3)) * WIND_SPEEDS[0]
    #     truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(
    #         effective_velocity_trurth
    #     )
    #     cp_truth = turbine_data["power_thrust_table"]["power"][truth_index]
    #     power_truth = (
    #         0.5
    #         * turbine.rotor_area
    #         * cp_truth
    #         * turbine.generator_efficiency
    #         * effective_velocity_trurth ** 3
    #     )
    #     print(i,WIND_SPEEDS, effective_velocity_trurth, cp_truth, p[0, 0, i], power_truth)
    #     np.testing.assert_allclose(p[0, 0, i], power_truth)


def test_axial_induction():

    N_TURBINES = 4

    turbine_data = SampleInputs().turbine
    turbine_floating_data = SampleInputs().turbine_floating
    turbine = Turbine.from_dict(turbine_data)
    turbine_floating = Turbine.from_dict(turbine_floating_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, None, :]

    baseline_ai = 0.25116283939089806

    # Single turbine
    wind_speed = 10.0
    ai = axial_induction(
        velocities=wind_speed * np.ones((1, 1, 1, 3, 3)),
        yaw_angle=np.zeros((1, 1, 1)),
        tilt_angle=np.ones((1, 1, 1)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, 1, 1)) * 5.0,
        fCt={turbine.turbine_type: turbine.fCt_interp},
        tilt_interp=np.array([(turbine.turbine_type, None)]),
        correct_cp_ct_for_tilt=np.array([[[False]]]),
        turbine_type_map=turbine_type_map[0,0,0],
    )
    np.testing.assert_allclose(ai, baseline_ai)

    # Multiple turbines with ix filter
    ai = axial_induction(
        velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 3 x 4 x 4 x 3 x 3
        yaw_angle=np.zeros((1, 1, N_TURBINES)),
        tilt_angle=np.ones((1, 1, N_TURBINES)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, 1, N_TURBINES)) * 5.0,
        fCt={turbine.turbine_type: turbine.fCt_interp},
        tilt_interp=np.array([(turbine.turbine_type, None)] * N_TURBINES),
        correct_cp_ct_for_tilt=np.array([[[False] * N_TURBINES]]),
        turbine_type_map=turbine_type_map,
        ix_filter=INDEX_FILTER,
    )

    assert len(ai[0, 0]) == len(INDEX_FILTER)

    # Test the 10 m/s wind speed to use the same baseline as above
    np.testing.assert_allclose(ai[0,2], baseline_ai)

    # Single floating turbine; note that 'tilt_interp' is not set to None
    ai = axial_induction(
        velocities=wind_speed * np.ones((1, 1, 1, 3, 3)),
        yaw_angle=np.zeros((1, 1, 1)),
        tilt_angle=np.ones((1, 1, 1)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, 1, 1)) * 5.0,
        fCt={turbine.turbine_type: turbine_floating.fCt_interp},
        tilt_interp=np.array([(turbine_floating.turbine_type, turbine_floating.fTilt_interp)]),
        correct_cp_ct_for_tilt=np.array([[[True]]]),
        turbine_type_map=turbine_type_map[0,0,0],
    )
    np.testing.assert_allclose(ai, baseline_ai)


def test_rotor_velocity_yaw_correction():
    N_TURBINES = 4

    wind_speed = average_velocity(10.0 * np.ones((1, 1, 1, 3, 3)))
    wind_speed_N_TURBINES = average_velocity(10.0 * np.ones((1, 1, N_TURBINES, 3, 3)))

    # Test a single turbine for zero yaw
    yaw_corrected_velocities = _rotor_velocity_yaw_correction(
        pP=3.0,
        yaw_angle=0.0,
        rotor_effective_velocities=wind_speed,
    )
    np.testing.assert_allclose(yaw_corrected_velocities, wind_speed)

    # Test a single turbine for non-zero yaw
    yaw_corrected_velocities = _rotor_velocity_yaw_correction(
        pP=3.0,
        yaw_angle=60.0,
        rotor_effective_velocities=wind_speed,
    )
    np.testing.assert_allclose(yaw_corrected_velocities, 0.5 * wind_speed)

    # Test multiple turbines for zero yaw
    yaw_corrected_velocities = _rotor_velocity_yaw_correction(
        pP=3.0,
        yaw_angle=np.zeros((1, 1, N_TURBINES)),
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )
    np.testing.assert_allclose(yaw_corrected_velocities, wind_speed_N_TURBINES)

    # Test multiple turbines for non-zero yaw
    yaw_corrected_velocities = _rotor_velocity_yaw_correction(
        pP=3.0,
        yaw_angle=np.ones((1, 1, N_TURBINES)) * 60.0,
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )
    np.testing.assert_allclose(yaw_corrected_velocities, 0.5 * wind_speed_N_TURBINES)


def test_rotor_velocity_tilt_correction():
    N_TURBINES = 4

    wind_speed = average_velocity(10.0 * np.ones((1, 1, 1, 3, 3)))
    wind_speed_N_TURBINES = average_velocity(10.0 * np.ones((1, 1, N_TURBINES, 3, 3)))

    turbine_data = SampleInputs().turbine
    turbine_floating_data = SampleInputs().turbine_floating
    turbine = Turbine.from_dict(turbine_data)
    turbine_floating = Turbine.from_dict(turbine_floating_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, None, :]

    # Test single non-floating turbine
    tilt_corrected_velocities = _rotor_velocity_tilt_correction(
        turbine_type_map=np.array([turbine_type_map[:, :, 0]]),
        tilt_angle=5.0*np.ones((1, 1, 1)),
        ref_tilt_cp_ct=np.array([turbine.ref_tilt_cp_ct]),
        pT=np.array([turbine.pT]),
        tilt_interp=np.array([(turbine.turbine_type, turbine.fTilt_interp)]),
        correct_cp_ct_for_tilt=np.array([[[False]]]),
        rotor_effective_velocities=wind_speed,
    )

    np.testing.assert_allclose(tilt_corrected_velocities, wind_speed)

    # Test multiple non-floating turbines
    tilt_corrected_velocities = _rotor_velocity_tilt_correction(
        turbine_type_map=turbine_type_map,
        tilt_angle=5.0*np.ones((1, 1, N_TURBINES)),
        ref_tilt_cp_ct=np.array([turbine.ref_tilt_cp_ct] * N_TURBINES),
        pT=np.array([turbine.pT] * N_TURBINES),
        tilt_interp=np.array([(turbine.turbine_type, turbine.fTilt_interp)] * N_TURBINES),
        correct_cp_ct_for_tilt=np.array([[[False] * N_TURBINES]]),
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )

    np.testing.assert_allclose(tilt_corrected_velocities, wind_speed_N_TURBINES)

    # Test single floating turbine
    tilt_corrected_velocities = _rotor_velocity_tilt_correction(
        turbine_type_map=np.array([turbine_type_map[:, :, 0]]),
        tilt_angle=5.0*np.ones((1, 1, 1)),
        ref_tilt_cp_ct=np.array([turbine_floating.ref_tilt_cp_ct]),
        pT=np.array([turbine_floating.pT]),
        tilt_interp=np.array([(turbine_floating.turbine_type, turbine_floating.fTilt_interp)]),
        correct_cp_ct_for_tilt=np.array([[[True]]]),
        rotor_effective_velocities=wind_speed,
    )

    np.testing.assert_allclose(tilt_corrected_velocities, wind_speed)

    # Test multiple floating turbines
    tilt_corrected_velocities = _rotor_velocity_tilt_correction(
        turbine_type_map,
        tilt_angle=5.0*np.ones((1, 1, N_TURBINES)),
        ref_tilt_cp_ct=np.array([turbine_floating.ref_tilt_cp_ct] * N_TURBINES),
        pT=np.array([turbine_floating.pT] * N_TURBINES),
        tilt_interp=np.array(
            [(turbine_floating.turbine_type, turbine_floating.fTilt_interp)] * N_TURBINES
        ),
        correct_cp_ct_for_tilt=np.array([[[True] * N_TURBINES]]),
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )

    np.testing.assert_allclose(tilt_corrected_velocities, wind_speed_N_TURBINES)


def test_compute_tilt_angles_for_floating_turbines():
    N_TURBINES = 4

    wind_speed = 25.0
    rotor_effective_velocities = average_velocity(wind_speed * np.ones((1, 1, 1, 3, 3)))
    rotor_effective_velocities_N_TURBINES = average_velocity(
        wind_speed * np.ones((1, 1, N_TURBINES, 3, 3))
    )

    turbine_floating_data = SampleInputs().turbine_floating
    turbine_floating = Turbine.from_dict(turbine_floating_data)
    turbine_type_map = np.array(N_TURBINES * [turbine_floating.turbine_type])
    turbine_type_map = turbine_type_map[None, None, :]

    # Single turbine
    tilt = compute_tilt_angles_for_floating_turbines(
        turbine_type_map=np.array([turbine_type_map[:, :, 0]]),
        tilt_angle=5.0*np.ones((1, 1, 1)),
        tilt_interp=np.array([(turbine_floating.turbine_type, turbine_floating.fTilt_interp)]),
        rotor_effective_velocities=rotor_effective_velocities,
    )

    # calculate tilt again
    truth_index = turbine_floating_data["floating_tilt_table"]["wind_speeds"].index(wind_speed)
    tilt_truth = turbine_floating_data["floating_tilt_table"]["tilt"][truth_index]
    np.testing.assert_allclose(tilt, tilt_truth)

    # Mulitple turbines
    tilt_N_turbines = compute_tilt_angles_for_floating_turbines(
        turbine_type_map=np.array(turbine_type_map),
        tilt_angle=5.0*np.ones((1, 1, N_TURBINES)),
        tilt_interp=np.array(
            [(turbine_floating.turbine_type, turbine_floating.fTilt_interp)] * N_TURBINES
        ),
        rotor_effective_velocities=rotor_effective_velocities_N_TURBINES,
    )

    # calculate tilt again
    truth_index = turbine_floating_data["floating_tilt_table"]["wind_speeds"].index(wind_speed)
    tilt_truth = turbine_floating_data["floating_tilt_table"]["tilt"][truth_index]
    np.testing.assert_allclose(tilt_N_turbines, [[[tilt_truth] * N_TURBINES]])


def test_asdict(sample_inputs_fixture: SampleInputs):

    turbine = Turbine.from_dict(sample_inputs_fixture.turbine)
    dict1 = turbine.as_dict()

    new_turb = Turbine.from_dict(dict1)
    dict2 = new_turb.as_dict()

    assert dict1 == dict2
