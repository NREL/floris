# Copyright 2024 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import numpy as np
import pytest

from floris.tools import TimeSeries, WindRose, WindDataBase

class ChildClassTest(WindDataBase):
    def __init__(self):
        pass

def test_bad_inheritance():
    test_class = ChildClassTest()
    with pytest.raises(NotImplementedError):
        test_class.unpack()

def test_time_series_instantiation():
    wind_directions = np.array([270, 280, 290])
    wind_speeds = np.array([5, 5, 5])
    time_series = TimeSeries(wind_directions, wind_speeds)
    time_series


def test_time_series_wrong_dimensions():
    wind_directions = np.array([270, 280, 290])
    wind_speeds = np.array([5, 5])
    with pytest.raises(ValueError):
        TimeSeries(wind_directions, wind_speeds)


def test_wind_rose_wrong_dimensions():
    wind_directions = np.array([270, 280, 290])
    wind_speeds = np.array([6, 7])

    # This should be ok:
    _ = WindRose(wind_directions, wind_speeds)

    # This should be ok
    _ = WindRose(wind_directions, wind_speeds, np.ones((3, 2)))

    # This should raise an error
    with pytest.raises(ValueError):
        WindRose(wind_directions, wind_speeds, np.ones((3, 3)))


def test_wind_rose_grid():
    wind_directions = np.array([270, 280, 290])
    wind_speeds = np.array([6, 7])

    wind_rose = WindRose(wind_directions, wind_speeds)

    # Wd grid has same dimensions as freq table
    assert wind_rose.wd_grid.shape == wind_rose.freq_table.shape

    # Flattening process occurs wd first
    np.testing.assert_allclose(wind_rose.wd_flat, [270, 270, 280, 280, 290, 290])


def test_wind_rose_unpack():
    wind_directions = np.array([270, 280, 290])
    wind_speeds = np.array([6, 7])
    freq_table = np.array([[1.0, 0.0], [0, 1.0], [0, 0]])

    # First test using default assumption only non-zero frequency cases computed
    wind_rose = WindRose(wind_directions, wind_speeds, freq_table)

    (
        wind_directions_unpack,
        wind_speeds_unpack,
        freq_table_unpack,
        ti_table_unpack,
        price_table_unpack,
    ) = wind_rose.unpack()

    # Given the above frequency table, would only expect the
    # (270 deg, 6 m/s) and (280 deg, 7 m/s) rows
    np.testing.assert_allclose(wind_directions_unpack, [270, 280])
    np.testing.assert_allclose(wind_speeds_unpack, [6, 7])
    np.testing.assert_allclose(freq_table_unpack, [0.5, 0.5])

    # In this case n_findex == 2
    assert wind_rose.n_findex == 2

    # Now test computing 0-freq cases too
    wind_rose = WindRose(wind_directions, wind_speeds, freq_table, compute_zero_freq_occurence=True)

    (
        wind_directions_unpack,
        wind_speeds_unpack,
        freq_table_unpack,
        ti_table_unpack,
        price_table_unpack,
    ) = wind_rose.unpack()

    # Expect now to compute all combinations
    np.testing.assert_allclose(wind_directions_unpack, [270, 270, 280, 280, 290, 290])

    # In this case n_findex == 6
    assert wind_rose.n_findex == 6


def test_unpack_for_reinitialize():
    wind_directions = np.array([270, 280, 290])
    wind_speeds = np.array([6, 7])
    freq_table = np.array([[1.0, 0.0], [0, 1.0], [0, 0]])

    # First test using default assumption only non-zero frequency cases computed
    wind_rose = WindRose(wind_directions, wind_speeds, freq_table)

    (
        wind_directions_unpack,
        wind_speeds_unpack,
        ti_table_unpack,
    ) = wind_rose.unpack_for_reinitialize()

    # Given the above frequency table, would only expect the
    # (270 deg, 6 m/s) and (280 deg, 7 m/s) rows
    np.testing.assert_allclose(wind_directions_unpack, [270, 280])
    np.testing.assert_allclose(wind_speeds_unpack, [6, 7])


def test_wind_rose_resample():
    wind_directions = np.array([0, 2, 4, 6, 8, 10])
    wind_speeds = np.array([8])
    freq_table = np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])

    wind_rose = WindRose(wind_directions, wind_speeds, freq_table)

    # Test that resampling with a new step size returns the same
    wind_rose_resample = wind_rose.resample_wind_rose()

    np.testing.assert_allclose(wind_rose.wind_directions, wind_rose_resample.wind_directions)
    np.testing.assert_allclose(wind_rose.wind_speeds, wind_rose_resample.wind_speeds)
    np.testing.assert_allclose(wind_rose.freq_table_flat, wind_rose_resample.freq_table_flat)

    # Now test resampling the wind direction to 5 deg bins
    wind_rose_resample = wind_rose.resample_wind_rose(wd_step=5.0)
    np.testing.assert_allclose(wind_rose_resample.wind_directions, [0, 5, 10])
    np.testing.assert_allclose(wind_rose_resample.freq_table_flat, [2 / 6, 2 / 6, 2 / 6])


def test_wrap_wind_directions_near_360():
    wd_step = 5.0
    wd_values = np.array([0, 180, 357, 357.5, 358])
    time_series = TimeSeries(np.array([0]), np.array([0]))

    wd_wrapped = time_series._wrap_wind_directions_near_360(wd_values, wd_step)

    expected_result = np.array([0, 180, 357, -wd_step / 2.0, -2.0])
    assert np.allclose(wd_wrapped, expected_result)


def test_time_series_to_wind_rose():
    # Test just 1 wind speed
    wind_directions = np.array([259.8, 260.2, 264.3])
    wind_speeds = np.array([5.0, 5.0, 5.1])
    time_series = TimeSeries(wind_directions, wind_speeds)
    wind_rose = time_series.to_wind_rose(wd_step=2.0, ws_step=1.0)

    # The wind directions should be 260, 262 and 264
    assert np.allclose(wind_rose.wind_directions, [260, 262, 264])

    # Freq table should have dimension of 3 wd x 1 ws
    freq_table = wind_rose.freq_table
    assert freq_table.shape[0] == 3
    assert freq_table.shape[1] == 1

    # The frequencies should [2/3, 0, 1/3]
    assert np.allclose(freq_table.squeeze(), [2 / 3, 0, 1 / 3])

    # Test just 2 wind speeds
    wind_directions = np.array([259.8, 260.2, 264.3])
    wind_speeds = np.array([5.0, 5.0, 6.1])
    time_series = TimeSeries(wind_directions, wind_speeds)
    wind_rose = time_series.to_wind_rose(wd_step=2.0, ws_step=1.0)

    # The wind directions should be 260, 262 and 264
    assert np.allclose(wind_rose.wind_directions, [260, 262, 264])

    # The wind speeds should be 5 and 6
    assert np.allclose(wind_rose.wind_speeds, [5, 6])

    # Freq table should have dimension of 3 wd x 2 ws
    freq_table = wind_rose.freq_table
    assert freq_table.shape[0] == 3
    assert freq_table.shape[1] == 2

    # The frequencies should [2/3, 0, 1/3]
    assert freq_table[0, 0] == 2 / 3
    assert freq_table[2, 1] == 1 / 3


def test_time_series_to_wind_rose_wrapping():
    wind_directions = np.arange(0.0, 360.0, 0.25)
    wind_speeds = 8.0 * np.ones_like(wind_directions)
    time_series = TimeSeries(wind_directions, wind_speeds)
    wind_rose = time_series.to_wind_rose(wd_step=2.0, ws_step=1.0)

    # Expert for the first bin in this case to be 0, and the final to be 358
    # and both to have equal numbers of points
    np.testing.assert_almost_equal(wind_rose.wind_directions[0], 0)
    np.testing.assert_almost_equal(wind_rose.wind_directions[-1], 358)
    np.testing.assert_almost_equal(wind_rose.freq_table[0, 0], wind_rose.freq_table[-1, 0])


def test_time_series_to_wind_rose_with_ti():
    wind_directions = np.array([259.8, 260.2, 260.3, 260.1])
    wind_speeds = np.array([5.0, 5.0, 5.1, 7.2])
    turbulence_intensity = np.array([0.5, 1.0, 1.5, 2.0])
    time_series = TimeSeries(
        wind_directions, wind_speeds, turbulence_intensity=turbulence_intensity
    )
    wind_rose = time_series.to_wind_rose(wd_step=2.0, ws_step=1.0)

    # Turbulence intensity should average to 1 in the 5 m/s bin and 2 in the 7 m/s bin
    ti_table = wind_rose.ti_table
    np.testing.assert_almost_equal(ti_table[0, 0], 1)
    np.testing.assert_almost_equal(ti_table[0, 2], 2)

    # The 6 m/s bin should be empty
    freq_table = wind_rose.freq_table
    np.testing.assert_almost_equal(freq_table[0, 1], 0)
