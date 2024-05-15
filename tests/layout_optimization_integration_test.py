import logging
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
from floris.optimization.layout_optimization.layout_optimization_random_search import (
    LayoutOptimizationRandomSearch,
)
from floris.wind_data import WindDataBase


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"


def test_base_class(caplog):
    # Get a test fi
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Set up a sample boundary
    boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]

    # Now initiate layout optimization with a frequency matrix passed in the 3rd position
    # (this should fail)
    freq = np.ones((5, 5))
    freq = freq / freq.sum()

    # Check that warning is raised if fmodel does not contain wind_data
    with caplog.at_level(logging.WARNING):
        LayoutOptimization(fmodel, boundaries, 5)
    assert caplog.text != "" # Checking not empty

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        LayoutOptimization(fmodel=fmodel, boundaries=boundaries, min_dist=5,)
    assert caplog.text != "" # Checking not empty

    time_series = TimeSeries(
        wind_directions=fmodel.core.flow_field.wind_directions,
        wind_speeds=fmodel.core.flow_field.wind_speeds,
        turbulence_intensities=fmodel.core.flow_field.turbulence_intensities,
    )
    fmodel.set(wind_data=time_series)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        LayoutOptimization(fmodel, boundaries, 5)
    assert caplog.text != "" # Not empty, because get_farm_AEP called on TimeSeries

    # Passing without keyword arguments should work, or with keyword arguments
    LayoutOptimization(fmodel, boundaries, 5)
    LayoutOptimization(fmodel=fmodel, boundaries=boundaries, min_dist=5)

    # Check with WindRose on fmodel
    fmodel.set(wind_data=time_series.to_WindRose())

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        LayoutOptimization(fmodel, boundaries, 5)
    assert caplog.text == "" # Empty

    LayoutOptimization(fmodel, boundaries, 5)
    LayoutOptimization(fmodel=fmodel, boundaries=boundaries, min_dist=5)

def test_LayoutOptimizationRandomSearch():
    fmodel = FlorisModel(configuration=YAML_INPUT)
    fmodel.set(layout_x=[0, 500], layout_y = [0, 0])

    # Set up a sample boundary
    boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]

    layout_opt = LayoutOptimizationRandomSearch(
        fmodel=fmodel,
        boundaries=boundaries,
        min_dist_D=5,
        seconds_per_iteration=1,
        total_optimization_seconds=1,
        use_dist_based_init=False,
    )

    # Check that the optimization runs
    layout_opt.optimize()
