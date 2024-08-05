from pathlib import Path

import numpy as np
import pytest

from floris import FlorisModel
from floris.heterogeneous_map import HeterogeneousMap


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"

layout_x = np.array([0.0, 500.0])
layout_y = np.array([0.0, 0.0])

heterogeneous_wd_inflow_config_2D = {
    "x": [-1000, -1000, 1000, 1000],
    "y": [-1000, 1000, -1000, 1000],
    "u": [[8.0, 8.0, 8.0, 8.0]],
    "v": [[0.0, 0.0, 4.0, 4.0]],
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

def test_input_types():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # First run without heterogeneity
    fmodel.set(
        layout_x=layout_x,
        layout_y=layout_y,
        heterogeneous_inflow_config=heterogeneous_wd_inflow_config_2D
    )
    fmodel.run()
    P_lists = fmodel.get_turbine_powers()

    # Convert all elements of dictionary to arrays
    for k in heterogeneous_wd_inflow_config_2D:
        heterogeneous_wd_inflow_config_2D[k] = np.array(heterogeneous_wd_inflow_config_2D[k])
    assert isinstance(heterogeneous_wd_inflow_config_2D["u"], np.ndarray)

    fmodel.set(heterogeneous_inflow_config=heterogeneous_wd_inflow_config_2D)
    fmodel.run()
    P_arrays = fmodel.get_turbine_powers()

    # Confirm upstream power the same
    assert np.allclose(P_lists, P_arrays)

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

def test_flipped_direction():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Set up a case with 2 findices, central turbines, aligned in
    fmodel.set(
        wind_directions=[270, 90],
        wind_speeds=[8.0, 8.0],
        turbulence_intensities=[0.06, 0.06],
        layout_x=np.array([-250.0, 250.0]),
        layout_y=np.array([0.0, 0.0]),
        heterogeneous_inflow_config={
            "x": heterogeneous_wd_inflow_config_2D["x"],
            "y": heterogeneous_wd_inflow_config_2D["y"],
            "u": np.concatenate(
                (
                    heterogeneous_wd_inflow_config_2D["u"],
                    np.flip(heterogeneous_wd_inflow_config_2D["u"], axis=1)
                ),
                axis=0
            ),
            "v": np.concatenate(
                (
                    heterogeneous_wd_inflow_config_2D["v"],
                    np.flip(heterogeneous_wd_inflow_config_2D["v"], axis=1)
                ),
                axis=0
            ),
            "speed_multipliers": np.repeat(
                heterogeneous_wd_inflow_config_2D["speed_multipliers"], 2, axis=0
            )
        }
    )
    fmodel.run()
    powers = fmodel.get_turbine_powers()
    assert np.allclose(powers[0,:], np.flip(powers[:,1]))

def test_rotational_symmetry():
    layout_x_symmetry = np.array([-250.0, 250.0])
    layout_y_symmetry = np.array([0.0, 0.0])

    fmodel = FlorisModel(configuration=YAML_INPUT)
    fmodel.set(
        layout_x=layout_x_symmetry,
        layout_y=layout_y_symmetry,
        heterogeneous_inflow_config=heterogeneous_wd_inflow_config_2D
    )
    fmodel.run()
    P_base = fmodel.get_turbine_powers()

    # Rotate to a series of new positions, and rotate layout also
    wind_directions_test = np.linspace(0, 360, 10)

    for wd in wind_directions_test:
        het_wd_inflow_test = heterogeneous_wd_inflow_config_2D.copy()
        het_wd_inflow_test["x"] = \
            np.cos(np.deg2rad(270-wd))*np.array(heterogeneous_wd_inflow_config_2D["x"]) \
            - np.sin(np.deg2rad(270-wd))*np.array(heterogeneous_wd_inflow_config_2D["y"])
        het_wd_inflow_test["y"] = \
            np.sin(np.deg2rad(270-wd))*np.array(heterogeneous_wd_inflow_config_2D["x"]) \
            + np.cos(np.deg2rad(270-wd))*np.array(heterogeneous_wd_inflow_config_2D["y"])

        layout_x_test = (
            np.cos(np.deg2rad(270-wd))*layout_x_symmetry
            - np.sin(np.deg2rad(270-wd))*layout_y_symmetry
        )
        layout_y_test = (
            np.sin(np.deg2rad(270-wd))*layout_x_symmetry
            + np.cos(np.deg2rad(270-wd))*layout_y_symmetry
        )

        fmodel.set(
            wind_directions=[wd],
            layout_x=layout_x_test,
            layout_y=layout_y_test,
            heterogeneous_inflow_config=het_wd_inflow_test
        )
        fmodel.run()
        P_new = fmodel.get_turbine_powers()

        assert np.allclose(P_base, P_new, rtol=1e-4)
