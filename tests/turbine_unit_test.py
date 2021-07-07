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


# import numpy as np
from tests.conftest import sample_inputs_fixture
import pytest
import numpy as np
from src.turbine import Turbine, power, Ct, axial_induction, average_velocity

@pytest.fixture
def turbine_fixture(sample_inputs_fixture) -> Turbine:
    return Turbine(sample_inputs_fixture.turbine)


def test_turbine_rotor_radius(turbine_fixture, sample_inputs_fixture):
    # Test that the radius is set correctly from the input file
    assert turbine_fixture.rotor_radius == sample_inputs_fixture.turbine["rotor_diameter"] / 2.0

    # Test the radius setter method since it actually sets the diameter
    turbine_fixture.rotor_radius = 200.0
    assert turbine_fixture.rotor_diameter == 400.0

    # Test the getter-method again
    assert turbine_fixture.rotor_radius == 200.0


def test_average_velocity():
    # TODO: why do we use cube root - mean - cube (like rms) instead of a simple average (np.mean)?
    velocities = np.array([
        [
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
        ],
        [
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
        ],
        [
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
        ],
    ])
    print(average_velocity(velocities))
    assert 1 == 2