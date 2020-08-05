# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import copy

import numpy as np
import pytest

from floris.simulation import Turbine
from floris.utilities import Vec3

@pytest.fixture
def turbine_fixture(sample_inputs_fixture):
    turbine = Turbine(sample_inputs_fixture.turbine)
    return turbine


def test_calculate_swept_area_velocities(turbine_fixture):
    local_wind_speed = np.array(
        [
            [
                [6.92380037, 7.59695563, 8.000000, 8.29335086, 8.52597113],
                [6.92380037, 7.59695563, 8.000000, 8.29335086, 8.52597113],
                [6.92380037, 7.59695563, 8.000000, 8.29335086, 8.52597113],
                [6.92380037, 7.59695563, 8.000000, 8.29335086, 8.52597113],
                [6.92380037, 7.59695563, 8.000000, 8.29335086, 8.52597113]
            ]
        ]
    )

    coord = Vec3(0, 0, 90)

    x = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]
        ]
    )

    y = np.array(
        [
            [
                [-63.0, -63.0, -63.0, -63.0, -63.0],
                [-31.5, -31.5, -31.5, -31.5, -31.5],
                [  0.0,   0.0,   0.0,   0.0,   0.0],
                [ 31.5,  31.5,  31.5,  31.5,  31.5],
                [ 63.0,  63.0,  63.0,  63.0,  63.0]
            ]
        ]
    )

    z = np.array(
        [
            [
                [ 27.0,  58.5,  90.0,  121.5, 153.0],
                [ 27.0,  58.5,  90.0,  121.5, 153.0],
                [ 27.0,  58.5,  90.0,  121.5, 153.0],
                [ 27.0,  58.5,  90.0,  121.5, 153.0],
                [ 27.0,  58.5,  90.0,  121.5, 153.0]
            ]
        ]
    )

    ws1, ws2 = turbine_fixture.calculate_swept_area_velocities(local_wind_speed, coord, x, y, z, local_wind_speed)

    baseline = np.array([7.59695563, 7.59695563, 7.59695563, 8.0, 8.0, 8.0, 8.29335086, 8.29335086, 8.29335086])

    assert pytest.approx(ws1) == baseline
    assert pytest.approx(ws1) == ws2
