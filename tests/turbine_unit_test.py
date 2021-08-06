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
from numpy.lib.arraysetops import isin
from numpy.lib.index_tricks import ix_

from src.turbine import (
    Ct,
    Turbine,
    PowerThrustTable,
    power,
    _filter_convert,
    axial_induction,
    average_velocity,
)
from tests.conftest import SampleInputs, sample_inputs_fixture


@pytest.fixture
def turbine_fixture(sample_inputs_fixture) -> Turbine:
    return sample_inputs_fixture.turbine


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
        table.thrust = np.arange(len(table.thrust))
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


# def test_turbine_rotor_radius(turbine_fixture, sample_inputs_fixture):
#     # Test that the radius is set correctly from the input file
#     assert turbine_fixture.rotor_radius == sample_inputs_fixture.turbine["rotor_diameter"] / 2.0

#     # Test the radius setter method since it actually sets the diameter
#     turbine_fixture.rotor_radius = 200.0
#     assert turbine_fixture.rotor_diameter == 400.0

#     # Test the getter-method again
#     assert turbine_fixture.rotor_radius == 200.0


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

    velocities = np.stack((np.ones((3, 3)), 2 * np.ones((3, 3))))
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
    velocities = np.stack(rng * np.ones((3, 3)))
    avg = average_velocity(velocities, ix_filter)
    assert avg.shape == (2,)
    np.testing.assert_array_equal(avg, np.array([0, 2]))
