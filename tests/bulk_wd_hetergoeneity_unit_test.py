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
from floris.wind_data import WindDataBase


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"

def test_base_class(caplog):
    # Get a test fi (single turbine at 0,0)
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Directly downstream at 270 degrees
    sample_x = [500.0]
    sample_y = [0.0]
    sample_z = [90.0]

    # Sweep across wind directions
    wd_array = np.arange(180, 360, 1)
    ws_array = 8.0 * np.ones_like(wd_array)
    ti_array = 0.06 * np.ones_like(wd_array)
    fmodel.set(wind_directions=wd_array, wind_speeds=ws_array, turbulence_intensities=ti_array)

    # Standard case; expect minimum to be at 270 degrees
    u_at_points = fmodel.sample_flow_at_points(sample_x, sample_y, sample_z)
    assert (wd_array[np.argmin(u_at_points)] == 270)

    # Now, apply bulk wind direction heterogeneity to the flow
    fmodel.set(
        heterogeneous_inflow_config={
            "bulk_wd_change": [0.0, 10.0], # 10 degree change, CW
            "bulk_wd_x": [0, 500.0]
        }
    )

    u_at_points = fmodel.sample_flow_at_points(sample_x, sample_y, sample_z)
    assert (wd_array[np.argmin(u_at_points)] != 270)


