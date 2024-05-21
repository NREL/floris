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

def test_bulk_wd_heterogeneity_turbine():
    fmodel = FlorisModel(configuration=YAML_INPUT)
    fmodel.set(layout_x=[0, 600], layout_y=[0, 0])

    # Run straight as a benchmark
    fmodel.set(wind_directions=[270.0], wind_speeds=[8.0], turbulence_intensities=[0.06])
    fmodel.run()
    powers_straight = fmodel.get_turbine_powers()[0,:]


    # Single wind direction, with two wind direction shifts as well as straight
    fmodel.set(
        wind_directions=[270.0, 270.0, 270.0],
        wind_speeds=[8.0, 8.0, 8.0],
        turbulence_intensities=[0.06, 0.06, 0.06],
        heterogeneous_inflow_config={
            "bulk_wd_change": [[0.0, 0.0], [0.0, -10.0], [0.0, 10.0]], # 10 degree changes
            "bulk_wd_x": [[0, 1000], [0, 1000], [0, 1000]] # Past downstream turbine
        }
    )

    fmodel.run()
    powers = fmodel.get_turbine_powers()

    assert (powers_straight == powers[0,:]).all() # Verify straight case

    assert (powers[:,0] == powers[0,0]).all() # Upstream turbine not affected
    assert powers[0,1] < powers[1,1] # wake shifted away from downstream turbine
    assert np.allclose(powers[1,1], powers[2,1]) # Shifted wake is symmetric

    # Check works for other directions too
    fmodel.set(
        wind_directions=[225.0, 225.0, 225.0],
        layout_x=[0, 600/np.sqrt(2)],
        layout_y=[0, 600/np.sqrt(2)],
    )
    fmodel.run()
    powers_2 = fmodel.get_turbine_powers()

    assert np.allclose(powers_2, powers)

def test_bulk_wd_heterogeneity_flow_field():
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
            "bulk_wd_change": [[0.0, -10.0]]*180, # -10 degree change, CW
            "bulk_wd_x": [[0, 500.0]]*180
        }
    ) # TODO: Build something that checks the dimensions of the inputs here

    u_at_points = fmodel.sample_flow_at_points(sample_x, sample_y, sample_z)
    assert (wd_array[np.argmin(u_at_points)] == 280)
