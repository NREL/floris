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

from tests.conftest import WIND_SPEEDS, SampleInputs
from floris.simulation import Ct, Turbine, power, axial_induction, average_velocity
from floris.simulation.turbine import PowerThrustTable, _filter_convert


# size 3 x 4 x 1 x 1
WIND_SPEEDS_BROADCAST = np.stack(
    (
        np.reshape(np.array(WIND_SPEEDS), (1, -1, 1, 1)),
        np.reshape(np.array(WIND_SPEEDS), (1, -1, 1, 1)),
        np.reshape(np.array(WIND_SPEEDS), (1, -1, 1, 1)),
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

    # Test that the values are immutable
    with pytest.raises(attr.exceptions.FrozenInstanceError):
        table.power = np.arange(len(table.power))

    with pytest.raises(attr.exceptions.FrozenInstanceError):
        table.thrust = np.arange(len(table.thrust))

    with pytest.raises(attr.exceptions.FrozenInstanceError):
        table.wind_speed = np.arange(len(table.wind_speed))

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
    np.testing.assert_array_equal(turbine.power_thrust_table.power, np.array(pt_data["power"]))
    np.testing.assert_array_equal(turbine.power_thrust_table.thrust, np.array(pt_data["thrust"]))
    np.testing.assert_array_equal(turbine.power_thrust_table.wind_speed, np.array(pt_data["wind_speed"]))

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
    assert average_velocity(velocities) == 1

    # Constructs an array of shape 1 x 1 x 2 x 3 x 3 with finrst turbie all 1, second turbine all 2
    velocities = np.stack(
        (
            np.ones((1, 1, 3, 3)),  # The first dimension here is the wind direction and the second
            2 * np.ones((1, 1, 3, 3)),  # is the wind speed since we are stacking on axis=2
        ),
        axis=2,
    )

    # Pull out the first wind speed for the test
    np.testing.assert_array_equal(average_velocity(velocities)[0, 0], np.array([1, 2]))

    # Test boolean filter
    ix_filter = [True, False, True, False]
    velocities = np.stack(  # 4 turbines with 3 x 3 velocity array; shape (1,1,4,3,3)
        ( i * np.ones((1, 1, 3, 3)) for i in range(1,5) ),
        # (
        #     np.ones(
        #         (1, 1, 3, 3)
        #     ),  # The first dimension here is the wind direction and second is the wind speed since we are stacking on axis=2
        #     2 * np.ones((1, 1, 3, 3)),
        #     3 * np.ones((1, 1, 3, 3)),
        #     4 * np.ones((1, 1, 3, 3)),
        # ),
        axis=2,
    )
    avg = average_velocity(velocities, ix_filter)
    assert avg.shape == (1, 1, 2)  # 1 wind direction, 1 wind speed, 2 turbines filtered

    # Pull out the first wind direction and wind speed for the comparison
    assert np.allclose(avg[0, 0], np.array([1.0, 3.0]))
    # This fails in GitHub Actions due to a difference in precision:
    # E           assert 3.0000000000000004 == 3.0
    # np.testing.assert_array_equal(avg[0], np.array([1.0, 3.0]))

    # Test integer array filter
    # np.arange(1, 5).reshape((-1,1,1)) * np.ones((1, 1, 3, 3))
    velocities = np.stack(  # 4 turbines with 3 x 3 velocity array; shape (1,1,4,3,3)
        ( i * np.ones((1, 1, 3, 3)) for i in range(1,5) ),
        axis=2,
    )
    avg = average_velocity(velocities, INDEX_FILTER)
    assert avg.shape == (1, 1, 2)  # 1 wind direction, 1 wind speed, 2 turbines filtered

    # Pull out the first wind direction and wind speed for the comparison
    assert np.allclose(avg[0, 0], np.array([1.0, 3.0]))


def test_ct():

    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)

    # Single turbine
    # yaw angle / fCt are (n wind direction, n wind speed, n turbine)
    wind_speed = 10.0
    thrust = Ct(
        velocities=wind_speed * np.ones((1, 1, 1, 3, 3)),
        yaw_angle=np.zeros((1, 1, 1)),
        fCt=turbine.fCt_interp,
    )

    truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    assert thrust == turbine_data["power_thrust_table"]["thrust"][truth_index]

    # Multiple turbines with index filter
    # 4 turbines with 3 x 3 grid arrays
    thrusts = Ct(
        velocities=np.ones((3, 3)) * WIND_SPEEDS_BROADCAST,  # 3 x 4 x 4 x 3 x 3
        yaw_angle=np.zeros((1, 1, 4)),
        fCt=turbine.fCt_interp,
        ix_filter=INDEX_FILTER,
    )
    assert len(thrusts[0, 0]) == len(INDEX_FILTER)

    for i, index in enumerate(INDEX_FILTER):
        truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(WIND_SPEEDS[index])
        assert thrusts[0, 0, i] == turbine_data["power_thrust_table"]["thrust"][truth_index]


def test_power():

    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)

    # Single turbine
    wind_speed = 10.0
    p = power(
        air_density=1.0 * np.ones((1, 1, 1)),
        velocities=wind_speed * np.ones((1, 1, 1, 3, 3)),
        yaw_angle=np.zeros((1, 1, 1)),
        pP=turbine.pP,
        power_interp=turbine.fCp,
    )

    truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    assert p == turbine_data["power_thrust_table"]["power"][truth_index]

    # Multiple turbines with ix filter
    # Why are we using air density of 1.0 here? If not 1, the test fails.
    p = power(
        air_density=1.0 * np.ones((1, 1, 4)),
        velocities=np.ones((3, 3)) * WIND_SPEEDS_BROADCAST,  # 3 x 4 x 4 x 3 x 3
        yaw_angle=np.zeros((1, 1, 4)),
        pP=turbine.pP,
        power_interp=turbine.fCp,
        ix_filter=INDEX_FILTER,
    )
    assert len(p[0, 0]) == len(INDEX_FILTER)

    for i, index in enumerate(INDEX_FILTER):
        truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(WIND_SPEEDS[index])
        assert p[0, 0, i] == turbine_data["power_thrust_table"]["power"][truth_index]


def test_axial_induction():

    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)

    baseline_ai = 0.25116283939089806

    # Single turbine
    wind_speed = 10.0
    ai = axial_induction(
        velocities=wind_speed * np.ones((1, 1, 1, 3, 3)),
        yaw_angle=np.zeros((1, 1, 1)),
        fCt=turbine.fCt,
    )
    assert ai == baseline_ai

    # Multiple turbines with ix filter
    ai = axial_induction(
        velocities=np.ones((3, 3)) * WIND_SPEEDS_BROADCAST,  # 3 x 4 x 4 x 3 x 3
        yaw_angle=np.zeros((1, 1, 4)),
        fCt=turbine.fCt,
        ix_filter=INDEX_FILTER,
    )

    assert len(ai[0, 0]) == len(INDEX_FILTER)
    for calc_ai, truth in zip(ai[0, 0], [0.2565471298176996, 0.2565471298176996]):
        pytest.approx(calc_ai, truth)
