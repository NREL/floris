from pathlib import Path

import numpy as np
import pytest

from floris import FlorisModel
from floris.heterogeneous_map import HeterogeneousMap


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"

layout_x = [0.0, 500.0]
layout_y = [0.0, 0.0]

heterogeneous_wd_inflow_config_2D = {
    "x": [-100, -100, 1000, 1000],
    "y": [-100, 100, -100, 100],
    "u": [[8.0, 8.0, 8.0, 8.0]],
    "v": [[0.0, 0.0, 8.0, 8.0]],
    "speed_multipliers": [[1.0, 1.0, 1.0, 1.0]] # Currently a necessary input
}

def test_power():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # First run without heterogeneity
    fmodel.set(layout_x=layout_x, layout_y=layout_y)
    fmodel.run()
    P_without_het = fmodel.get_turbine_powers()[0,:]

    # Add wind direction heterogeneity and rerun
    fmodel.set(heterogeneous_inflow_config=heterogeneous_wd_inflow_config_2D)
    fmodel.run()
    P_with_het = fmodel.get_turbine_powers()[0,:]

    # Confirm upstream power the same; downstream power increased
    assert np.allclose(P_without_het[0], P_with_het[0])
    assert P_with_het[1] > P_without_het[1]
    
def test_symmetry():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Set layout and heterogeneity
    fmodel.set(
        layout_x=layout_x,
        layout_y=layout_y,
        heterogeneous_inflow_config=heterogeneous_wd_inflow_config_2D
    )

    # Run in this direction
    fmodel.run()
    P_positive_v = fmodel.get_turbine_powers()[0,:]

    # Switch the sign of the inflow v component and rurun
    fmodel.set(
        heterogeneous_inflow_config={
            "x": heterogeneous_wd_inflow_config_2D["x"],
            "y": heterogeneous_wd_inflow_config_2D["y"],
            "u": heterogeneous_wd_inflow_config_2D["u"],
            "v": -np.array(heterogeneous_wd_inflow_config_2D["v"]),
            "speed_multipliers": heterogeneous_wd_inflow_config_2D["speed_multipliers"]
        }
    )
    fmodel.run()
    P_negative_v = fmodel.get_turbine_powers()[0,:]

    # Confirm symmetry
    assert np.allclose(P_positive_v, P_negative_v)

def test_multiple_findices():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Set layout and heterogeneity
    fmodel.set(
        layout_x=layout_x,
        layout_y=layout_y,
        heterogeneous_inflow_config=heterogeneous_wd_inflow_config_2D
    )
    fmodel.run()
    P_1findex = fmodel.get_turbine_powers()[0,:]

    # Duplicate the inflow conditons
    fmodel.set(
        heterogeneous_inflow_config={
            "x": heterogeneous_wd_inflow_config_2D["x"],
            "y": heterogeneous_wd_inflow_config_2D["y"],
            "u": np.repeat(heterogeneous_wd_inflow_config_2D["u"], 2, axis=0),
            "v": np.repeat(heterogeneous_wd_inflow_config_2D["v"], 2, axis=0),
            "speed_multipliers": np.repeat(
                heterogeneous_wd_inflow_config_2D["speed_multipliers"], 2, axis=0
            )
        },
        wind_directions = [270.0, 270.0],
        wind_speeds=[8.0, 8.0],
        turbulence_intensities=[0.06, 0.06]
    )
    fmodel.run()
    P_2findex = fmodel.get_turbine_powers()

    # Confirm the same results
    assert np.allclose(P_1findex, P_2findex)




def test_rotated_wind_direction(): # Fails because turbines are not correctly inside the domain (possibly)
    fmodel = FlorisModel(configuration=YAML_INPUT)
    fmodel.set(
        layout_x=layout_x,
        layout_y=layout_y,
        heterogeneous_inflow_config=heterogeneous_wd_inflow_config_2D
    )
    fmodel.run()
    P_270deg = fmodel.get_turbine_powers()[0,:]

    # Rotate the wind direction by 90 degrees and rerun
    fmodel.set(
        wind_directions=[0],
        heterogeneous_inflow_config=heterogeneous_wd_inflow_config_2D
    )
    fmodel.run()
    P_0deg = fmodel.get_turbine_powers()[0,:]

    print(P_270deg)
    print(P_0deg)


