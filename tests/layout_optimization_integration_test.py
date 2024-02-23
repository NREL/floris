from pathlib import Path

import numpy as np
import pytest

from floris.tools import (
    TimeSeries,
    WindRose,
)
from floris.tools.floris_interface import FlorisInterface
from floris.tools.optimization.layout_optimization.layout_optimization_base import (
    LayoutOptimization,
)
from floris.tools.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy,
)
from floris.tools.wind_data import WindDataBase


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"


def test_base_class():
    # Get a test fi
    fi = FlorisInterface(configuration=YAML_INPUT)

    # Set up a sample boundary
    boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]

    # Declare an instance of the base class with only fi and boundaries
    LayoutOptimization(fi=fi, boundaries=boundaries)

    # Now initiate layout optimization with a frequency matrix passed in the 4th position
    # (this should fail)
    freq = np.ones((5, 5))
    freq = freq / freq.sum()
    with pytest.raises(ValueError):
        LayoutOptimization(fi, boundaries, 5, freq)

    # Passing as a keyword freq to wind_data should also fail
    with pytest.raises(ValueError):
        LayoutOptimization(fi=fi, boundaries=boundaries, min_dist=5, wind_data=freq)

    time_series = TimeSeries(
        wind_directions=fi.floris.flow_field.wind_directions,
        wind_speeds=fi.floris.flow_field.wind_speeds,
        turbulence_intensities=fi.floris.flow_field.turbulence_intensities,
    )
    wind_rose = time_series.to_wind_rose()

    # Passing wind_data objects in the 4th position should not fail
    LayoutOptimization(fi, boundaries, 5, time_series)
    LayoutOptimization(fi, boundaries, 5, wind_rose)

    # Passing wind_data objects by keyword should not fail
    LayoutOptimization(fi=fi, boundaries=boundaries, min_dist=5, wind_data=time_series)
    LayoutOptimization(fi=fi, boundaries=boundaries, min_dist=5, wind_data=wind_rose)
