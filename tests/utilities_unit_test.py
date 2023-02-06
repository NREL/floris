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

from floris.utilities import (
    cosd,
    rotate_coordinates_rel_west,
    sind,
    tand,
    wind_delta,
    wrap_180,
    wrap_360,
)
from tests.conftest import (
    X_COORDS,
    Y_COORDS,
    Z_COORDS,
)


def test_cosd():
    assert pytest.approx(cosd(0.0)) == 1.0
    assert pytest.approx(cosd(90.0)) == 0.0
    assert pytest.approx(cosd(180.0)) == -1.0
    assert pytest.approx(cosd(270.0)) == 0.0


def test_sind():
    assert pytest.approx(sind(0.0)) == 0.0
    assert pytest.approx(sind(90.0)) == 1.0
    assert pytest.approx(sind(180.0)) == 0.0
    assert pytest.approx(sind(270.0)) == -1.0


def test_tand():
    assert pytest.approx(tand(0.0)) == 0.0
    assert pytest.approx(tand(45.0)) == 1.0
    assert pytest.approx(tand(135.0)) == -1.0
    assert pytest.approx(tand(180.0)) == 0.0
    assert pytest.approx(tand(225.0)) == 1.0
    assert pytest.approx(tand(315.0)) == -1.0


def test_wrap_180():
    assert wrap_180(-180.0) == -180.0
    assert wrap_180(180.0) == -180.0
    assert wrap_180(-181.0) == 179.0
    assert wrap_180(-179.0) == -179.0
    assert wrap_180(179.0) == 179.0
    assert wrap_180(181.0) == -179.0


def test_wrap_360():
    assert wrap_360(0.0) == 0.0
    assert wrap_360(360.0) == 0.0
    assert wrap_360(-1.0) == 359.0
    assert wrap_360(1.0) == 1.0
    assert wrap_360(359.0) == 359.0
    assert wrap_360(361.0) == 1.0


def test_wind_deviation_from_west():
    assert wind_delta(270.0) == 0.0
    assert wind_delta(280.0) == 10.0
    assert wind_delta(360.0) == 90.0
    assert wind_delta(180.0) == 270.0


def test_rotate_coordinates_rel_west():

    coordinates = np.array([ [x,y,z] for x,y,z in zip(X_COORDS, Y_COORDS, Z_COORDS)])

    # For 270, the coordinates should not change.
    wind_directions = np.array([270.0])
    x_rotated, y_rotated, z_rotated = rotate_coordinates_rel_west(wind_directions, coordinates)

    np.testing.assert_array_equal( X_COORDS, x_rotated[0,0] )
    np.testing.assert_array_equal( Y_COORDS, y_rotated[0,0] )
    np.testing.assert_array_equal( Z_COORDS, z_rotated[0,0] )

    # For 360, the coordinates should be rotated 90 degrees counter clockwise
    # from looking fown at the wind farm from above. The series of turbines
    # in a line (same y, increasing x) should change to a series of turbines
    # in parallel (same x, increasing y).
    # Since the rotation in `rotate_coordinates_rel_west` happens about the
    # center of all the points, adjust the coordinates so that the results are
    # adjusted to the baseline values.
    # NOTE: These adjustments are not general and will fail if the coordinates in
    # conftest change.
    wind_directions = np.array([360.0])
    x_rotated, y_rotated, z_rotated = rotate_coordinates_rel_west(wind_directions, coordinates)
    np.testing.assert_almost_equal( Y_COORDS, x_rotated[0,0] - np.min(x_rotated[0,0]))
    np.testing.assert_almost_equal( X_COORDS, y_rotated[0,0] - np.min(y_rotated[0,0]))
    np.testing.assert_almost_equal(
        Z_COORDS + np.min(Z_COORDS),
        z_rotated[0,0] + np.min(z_rotated[0,0])
    )

    wind_directions = np.array([90.0])
    x_rotated, y_rotated, z_rotated = rotate_coordinates_rel_west(wind_directions, coordinates)
    np.testing.assert_almost_equal( X_COORDS[-1:-4:-1], x_rotated[0,0] )
    np.testing.assert_almost_equal( Y_COORDS, y_rotated[0,0] )
    np.testing.assert_almost_equal( Z_COORDS, z_rotated[0,0] )
