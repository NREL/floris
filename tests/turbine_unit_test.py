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

from src.turbine import (
    Ct,
    Turbine,
    PowerThrustTable,
    power,
    _filter_convert,
    axial_induction,
    average_velocity,
)
from tests.conftest import SampleInputs


WIND_SPEEDS = [8.0, 9.0, 8.0, 11.0]
WIND_SPEEDS_BROADCAST = np.array(WIND_SPEEDS).reshape(-1, 1, 1)  # size 4 x 1 x 1
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
    np.testing.assert_array_equal(
        turbine.power_thrust_table.power, np.array(pt_data["power"])
    )
    np.testing.assert_array_equal(
        turbine.power_thrust_table.thrust, np.array(pt_data["thrust"])
    )
    np.testing.assert_array_equal(
        turbine.power_thrust_table.wind_speed, np.array(pt_data["wind_speed"])
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
    print(turbine.rotor_diameter, turbine.rotor_radius)
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
    # Test None is returned
    ix_filter = None
    sample_arg = 1
    assert _filter_convert(ix_filter, sample_arg) is None

    # Test a boolean truth array is returned
    N = 10
    ix_filter = None
    sample_arg = np.arange(N)
    ix_filter = _filter_convert(ix_filter, sample_arg)
    assert ix_filter.sum() == N
    assert ix_filter.shape == (N,)

    # Test that a numpy array is returned
    ix_filter = [1, 2]
    sample_arg = np.stack(np.arange(4).reshape(-1, 1, 1) * np.ones((3, 3)))
    ix_filter = _filter_convert(ix_filter, sample_arg)
    np.testing.assert_array_equal(ix_filter, np.array([1, 2]))

    # Test that a 1-D boolean truth array is returned
    N = 4
    ix_filter = None
    sample_arg = np.stack(np.arange(N).reshape(-1, 1, 1) * np.ones((3, 3)))
    ix_filter = _filter_convert(ix_filter, sample_arg)
    assert ix_filter.sum() == N
    assert ix_filter.shape == (N,)


def test_average_velocity():
    # TODO: why do we use cube root - mean - cube (like rms) instead of a simple average (np.mean)?
    velocities = np.ones((5, 5))
    assert average_velocity(velocities) == 1

    velocities = np.stack(
        (np.ones((3, 3)), 2 * np.ones((3, 3)))
    )  # 2 x 3 x 3 array with first turbine all 1, second turbine all 2
    np.testing.assert_array_equal(average_velocity(velocities), 1 + np.arange(2))

    # Test boolean filter
    rng = np.arange(4).reshape(-1, 1, 1)
    ix_filter = [True, False, True, False]
    velocities = np.stack(rng * np.ones((3, 3)))
    avg = average_velocity(velocities, ix_filter)
    assert avg.shape == (2,)
    np.testing.assert_array_equal(avg, np.array([0, 2]))

    # Test integer array filter
    rng = np.arange(4).reshape(-1, 1, 1)
    ix_filter = [0, 2]
    velocities = np.stack(
        rng * np.ones((3, 3))
    )  # 4 turbines with 3 x 3 velocity array; shape (4,3,3)
    avg = average_velocity(velocities, ix_filter)
    assert avg.shape == (2,)
    np.testing.assert_array_equal(avg, np.array([0, 2]))


def test_ct():

    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)

    # Single turbine
    wind_speed = 10.0
    thrust = Ct(velocities=wind_speed * np.ones((5, 5)), yaw_angle=0.0, fCt=turbine.fCt)

    truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    assert thrust == turbine_data["power_thrust_table"]["thrust"][truth_index]

    # Multiple turbines with index filter
    thrusts = Ct(
        velocities=np.ones((3, 3))
        * WIND_SPEEDS_BROADCAST,  # 4 turbines with 3 x 3 velocity array; shape (4,3,3)
        yaw_angle=np.zeros(4),
        fCt=4 * [turbine.fCt],
        ix_filter=INDEX_FILTER,
    )

    assert len(thrusts) == len(INDEX_FILTER)
    assert isinstance(thrusts, np.ndarray)

    for i, index in enumerate(INDEX_FILTER):
        truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(
            WIND_SPEEDS[index]
        )
        assert thrusts[i] == turbine_data["power_thrust_table"]["thrust"][truth_index]


def test_power():

    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)

    # Single turbine
    wind_speed = 10.0
    p = power(
        air_density=1.0,
        velocities=wind_speed * np.ones((5, 5)),
        yaw_angle=0.0,
        pP=turbine.pP,
        power_interp=turbine.fCp,
    )

    truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    assert p == turbine_data["power_thrust_table"]["power"][truth_index]

    # Multiple turbines with ix filter
    ix_filter = [0, 2]

    p = power(
        air_density=1.0 * np.ones(4),
        velocities=np.ones((3, 3))
        * WIND_SPEEDS_BROADCAST,  # 4 turbines with 3 x 3 velocity array; shape (4,3,3)
        yaw_angle=np.zeros(4),
        pP=4 * [turbine.pP],
        power_interp=4 * [turbine.fCp],
        ix_filter=ix_filter,
    )

    assert len(p) == len(ix_filter)
    assert isinstance(p, np.ndarray)

    for i, index in enumerate(ix_filter):
        truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(
            WIND_SPEEDS[index]
        )
        assert p[i] == turbine_data["power_thrust_table"]["power"][truth_index]


def test_axial_induction():

    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)

    # Single turbine
    wind_speed = 10.0
    ai = axial_induction(
        velocities=wind_speed * np.ones((5, 5)), yaw_angle=0.0, fCt=turbine.fCt
    )
    assert ai == 0.25116283939089806

    # Multiple turbines with ix filter
    # rng = np.array(WIND_SPEEDS).reshape(-1, 1, 1)  # size 4 x 1 x 1
    ix_filter = [0, 2]

    ai = axial_induction(
        velocities=np.ones((3, 3))
        * WIND_SPEEDS_BROADCAST,  # 4 turbines with 3 x 3 velocity array; shape (4,3,3)
        yaw_angle=np.zeros(4),
        fCt=4 * [turbine.fCt],
        ix_filter=ix_filter,
    )

    assert len(ai) == len(ix_filter)
    for calc_ai, truth in zip(ai, [0.2565471298176996, 0.2565471298176996]):
        pytest.approx(calc_ai, truth)
