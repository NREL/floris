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


import pytest

import floris.utilities


def test_cosd():
    assert pytest.approx(floris.utilities.cosd(0.0)) == 1.0
    assert pytest.approx(floris.utilities.cosd(90.0)) == 0.0
    assert pytest.approx(floris.utilities.cosd(180.0)) == -1.0
    assert pytest.approx(floris.utilities.cosd(270.0)) == 0.0


def test_sind():
    assert pytest.approx(floris.utilities.sind(0.0)) == 0.0
    assert pytest.approx(floris.utilities.sind(90.0)) == 1.0
    assert pytest.approx(floris.utilities.sind(180.0)) == 0.0
    assert pytest.approx(floris.utilities.sind(270.0)) == -1.0


def test_tand():
    assert pytest.approx(floris.utilities.tand(0.0)) == 0.0
    assert pytest.approx(floris.utilities.tand(45.0)) == 1.0
    assert pytest.approx(floris.utilities.tand(135.0)) == -1.0
    assert pytest.approx(floris.utilities.tand(180.0)) == 0.0
    assert pytest.approx(floris.utilities.tand(225.0)) == 1.0
    assert pytest.approx(floris.utilities.tand(315.0)) == -1.0


def test_wrap_180():
    assert floris.utilities.wrap_180(-180.0) == 180.0
    assert floris.utilities.wrap_180(180.0) == 180.0
    assert floris.utilities.wrap_180(-181.0) == 179.0
    assert floris.utilities.wrap_180(-179.0) == -179.0
    assert floris.utilities.wrap_180(179.0) == 179.0
    assert floris.utilities.wrap_180(181.0) == -179.0


def test_wrap_360():
    assert floris.utilities.wrap_360(0.0) == 0.0
    assert floris.utilities.wrap_360(360.0) == 0.0
    assert floris.utilities.wrap_360(-1.0) == 359.0
    assert floris.utilities.wrap_360(1.0) == 1.0
    assert floris.utilities.wrap_360(359.0) == 359.0
    assert floris.utilities.wrap_360(361.0) == 1.0
