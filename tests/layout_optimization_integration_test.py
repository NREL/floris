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

    # Check that ValueError raised if fmodel does not contain wind_data
    with pytest.raises(ValueError):
        LayoutOptimization(fmodel, boundaries, 5)
    with pytest.raises(ValueError):
        LayoutOptimization(fmodel=fmodel, boundaries=boundaries, min_dist=5,)

    time_series = TimeSeries(
        wind_directions=fmodel.core.flow_field.wind_directions,
        wind_speeds=fmodel.core.flow_field.wind_speeds,
        turbulence_intensities=fmodel.core.flow_field.turbulence_intensities,
    )
    fmodel.set(wind_data=time_series)

    # Passing without keyword arguments should work, or with keyword arguments
    LayoutOptimization(fmodel, boundaries, 5)
    LayoutOptimization(fmodel=fmodel, boundaries=boundaries, min_dist=5)

    # Check with WindRose on fmodel
    fmodel.set(wind_data=time_series.to_wind_rose())
    LayoutOptimization(fmodel, boundaries, 5)
    LayoutOptimization(fmodel=fmodel, boundaries=boundaries, min_dist=5)
