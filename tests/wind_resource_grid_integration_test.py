import copy
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import curve_fit

from floris.wind_resource_grid import WindResourceGrid


WRG_FILE_FILE = (
    Path(__file__).resolve().parent / "../examples/examples_wind_resource_grid/wrg_example.wrg"
)


def test_load_wrg():
    WindResourceGrid(WRG_FILE_FILE)


def test_read_header():
    """Test reading the header of a WRG file.

    The header of a WRG file is the first line of the file and contains the
    number of grid points in x and y, the minimum x and y coordinates, and the
    grid size.
    """

    wrg = WindResourceGrid(WRG_FILE_FILE)

    assert wrg.nx == 2
    assert wrg.ny == 3
    assert wrg.xmin == 0.0
    assert wrg.ymin == 0.0
    assert wrg.grid_size == 1000.0

    # Test x and y arrays
    assert np.allclose(wrg.x_array, np.array([0.0, 1000.0]))
    assert np.allclose(wrg.y_array, np.array([0.0, 1000.0, 2000.0]))

    # Test number of grid points
    assert wrg.n_gid == 6

    # Test number of sectors
    assert wrg.n_sectors == 12


def test_read_data():
    """Test reading the data of a WRG file.

    The data of a WRG file is the information about each grid point, including
    the x, y, z, and h coordinates, the frequency of each sector, and the Weibull
    parameters for each sector.
    """

    wrg = WindResourceGrid(WRG_FILE_FILE)

    # Test z and h values
    assert wrg.z == 0.0
    assert wrg.h == 90.0

    # Test the first and last gid for sector_freq, weibull_A, and weibull_k
    assert wrg.sector_freq[0, 0, 0] == 116 / 1000.0
    assert wrg.sector_freq[-1, -1, -1] == 34 / 1000.0

    assert wrg.weibull_A[0, 0, 0] == 106 / 10.0
    assert wrg.weibull_A[-1, -1, -1] == 81 / 10.0

    assert wrg.weibull_k[0, 0, 0] == 273 / 100.0
    assert wrg.weibull_k[-1, -1, -1] == 199 / 100.0


def test_build_interpolant_function_list():
    wrg = WindResourceGrid(WRG_FILE_FILE)

    # Initialize the values
    x = np.array([0.0, 1000.0])
    y = np.array([0.0, 500.0, 1000.0])
    n_sectors = 3
    data = np.ones((2, 3, 3))

    # For the first sector, the point at (1000, 0) is 2.0
    data[1, 0, 0] = 2.0

    # For the second sector, the point at (1000, 500) is 3.0
    data[1, 1, 1] = 3.0

    function_list = wrg._build_interpolant_function_list(x, y, n_sectors, data)

    # Length of the function list should be n_sectors
    assert len(function_list) == n_sectors

    # Test the interpolation in the first sector
    test_value = function_list[0]((500, 0))
    assert test_value == 1.5

    # Test the interpolation in the second sector
    test_value = function_list[1]((1000, 250))
    assert test_value == 2.0

    # Test using linear method
    test_value = function_list[0]((500, 0), method="linear")
    assert test_value == 1.5

    # Test using nearest method
    test_value = function_list[0]((1001, 0), method="nearest")
    assert test_value == 2.0


def test_interpolate_data():
    wrg = WindResourceGrid(WRG_FILE_FILE)

    sector_freq = wrg.sector_freq

    # Test that interpolating onto the point 0,0 returns the 1st row of the sector_freq
    test_value = wrg._interpolate_data(0, 0, wrg.interpolant_sector_freq)
    assert np.allclose(test_value, sector_freq[0, 0, :])

    # Test the interpolating just out of bounds of 0,0 also yields the 1st row of the sector_freq
    test_value = wrg._interpolate_data(-1, -1, wrg.interpolant_sector_freq)
    assert np.allclose(test_value, sector_freq[0, 0, :])

    # Test that value at x=500, y0, this is the average of the rows at [0,0] and [1,0]
    test_value = wrg._interpolate_data(500, 0, wrg.interpolant_sector_freq)
    assert np.allclose(test_value, (sector_freq[0, 0, :] + sector_freq[1, 0, :]) / 2)

    # Test that value at x=0, y=500, this is the average of the rows at [0,0] and [0,1]
    test_value = wrg._interpolate_data(0, 500, wrg.interpolant_sector_freq)
    assert np.allclose(test_value, (sector_freq[0, 0, :] + sector_freq[0, 1, :]) / 2)


def test_generate_wind_speed_frequencies_from_weibull():
    wrg = WindResourceGrid(WRG_FILE_FILE)

    wind_speeds_in = np.array([0.0, 5.0, 10.0, 15.0])
    wind_speeds, freq = wrg._generate_wind_speed_frequencies_from_weibull(
        10.0, 2.0, wind_speeds=wind_speeds_in
    )

    # Test that the wind speeds are the same
    assert np.allclose(wind_speeds, wind_speeds_in)

    # Test that freq is the same length as wind_speeds
    assert len(freq) == len(wind_speeds)

    # Test that the frequencies sum to 1.0
    assert np.allclose(np.sum(freq), 1.0)

    # Test the correctness of the frequencies by reversing the process
    wind_speeds = np.arange(0.0, 100.0, 1.0)
    A_test = 9.0
    k_test = 1.8
    wind_speeds, freq = wrg._generate_wind_speed_frequencies_from_weibull(
        A_test, k_test, wind_speeds=wind_speeds
    )

    # Test that the mean value matches theory
    mean_value = np.sum(wind_speeds * freq)
    assert np.allclose(mean_value, A_test * np.math.gamma(1 + 1 / k_test), rtol=0.1)


def test_get_wind_rose_at_point():
    wrg = WindResourceGrid(WRG_FILE_FILE)

    wind_speeds = np.arange(0.0, 25.0, 1.0)
    n_wind_speeds = len(wind_speeds)

    wind_rose = wrg.get_wind_rose_at_point(0, 0)

    # Test that there are 12 wind directions at n_wind_speeds wind speeds
    assert wind_rose.freq_table.shape == (12, n_wind_speeds)
    assert len(wind_rose.wind_speeds) == n_wind_speeds
    assert len(wind_rose.wind_directions) == 12

    # Test that freq table generated at (0, 0) is the same at that of (-1 , -1)
    wind_rose2 = wrg.get_wind_rose_at_point(-1, -1)
    assert np.allclose(wind_rose.freq_table, wind_rose2.freq_table)
