import copy
from pathlib import Path

import numpy as np
import pytest

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

    assert wrg.nx == 3
    assert wrg.ny == 3
    assert wrg.xmin == 0.0
    assert wrg.ymin == 0.0
    assert wrg.grid_size == 1000.0

    # Test x and y arrays
    assert np.allclose(wrg.x_array, np.array([0.0, 1000.0, 2000.0]))
    assert np.allclose(wrg.y_array, np.array([0.0, 1000.0, 2000.0]))

    # Test number of grid points
    assert wrg.n_gid == 9

    # Test number of sectors
    assert wrg.n_sectors == 12


def test_read_data():
    """Test reading the data of a WRG file.

    The data of a WRG file is the information about each grid point, including
    the x, y, z, and h coordinates, the frequency of each sector, and the Weibull
    parameters for each sector.
    """

    wrg = WindResourceGrid(WRG_FILE_FILE)

    # Test x, y, z, and h arrays
    assert np.allclose(
        wrg.x, np.array([0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0, 2000.0, 2000.0, 2000.0])
    )
    assert np.allclose(
        wrg.y, np.array([0.0, 1000.0, 2000.0, 0.0, 1000.0, 2000.0, 0.0, 1000.0, 2000.0])
    )
    assert np.allclose(wrg.z, np.zeros(9))
    assert np.allclose(wrg.h, 90.0 * np.ones(9))

    # Test the first and last gid for sector_freq, weibull_A, and weibull_k
    assert wrg.sector_freq[0, 0] == 116 / 1000.0
    assert wrg.sector_freq[-1, -1] == 31 / 1000.0

    assert wrg.weibull_A[0, 0] == 106 / 10.0
    assert wrg.weibull_A[-1, -1] == 90 / 10.0

    assert wrg.weibull_k[0, 0] == 273 / 100.0
    assert wrg.weibull_k[-1, -1] == 231 / 100.0
