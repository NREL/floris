from pathlib import Path

import numpy as np
import pytest

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
)
from floris.optimization.layout_optimization.layout_optimization_base import (
    LayoutOptimization,
)
from floris.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy,
)
from floris.wind_data import WindDataBase


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"


def test_base_class():
    # Get a test fi
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Set up a sample boundary
    boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]

    # Now initiate layout optimization with a frequency matrix passed in the 3rd position
    # (this should fail)
    freq = np.ones((5, 5))
    freq = freq / freq.sum()
    with pytest.raises(ValueError):
        LayoutOptimization(fmodel, boundaries, freq, 5)

    # Passing as a keyword freq to wind_data should also fail
    with pytest.raises(ValueError):
        LayoutOptimization(fmodel=fmodel, boundaries=boundaries, wind_data=freq, min_dist=5,)

    time_series = TimeSeries(
        wind_directions=fmodel.core.flow_field.wind_directions,
        wind_speeds=fmodel.core.flow_field.wind_speeds,
        turbulence_intensities=fmodel.core.flow_field.turbulence_intensities,
    )
    wind_rose = time_series.to_wind_rose()

    # Passing wind_data objects in the 3rd position should not fail
    LayoutOptimization(fmodel, boundaries, time_series, 5)
    LayoutOptimization(fmodel, boundaries, wind_rose, 5)

    # Passing wind_data objects by keyword should not fail
    LayoutOptimization(fmodel=fmodel, boundaries=boundaries, wind_data=time_series, min_dist=5)
    LayoutOptimization(fmodel=fmodel, boundaries=boundaries, wind_data=wind_rose, min_dist=5)
