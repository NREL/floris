
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import floris.layout_visualization as layoutviz
from floris import FlorisModel


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"


def test_get_wake_direction():
    # Turbine 0 wakes Turbine 1 at 270 degrees
    assert np.isclose(layoutviz.get_wake_direction(0, 0, 1, 0), 270.0)

    # Turbine 0 wakes Turbine 1 at 0 degrees
    assert np.isclose(layoutviz.get_wake_direction(0, 1, 0, 0), 0.0)

    # Winds from the south
    assert np.isclose(layoutviz.get_wake_direction(0, -1, 0, 0), 180.0)

def test_plotting_functions():

    fmodel = FlorisModel(configuration=YAML_INPUT)

    ax = layoutviz.plot_turbine_points(fmodel=fmodel)
    assert isinstance(ax, plt.Axes)

    ax = layoutviz.plot_turbine_labels(fmodel=fmodel)
    assert isinstance(ax, plt.Axes)

    ax = layoutviz.plot_turbine_rotors(fmodel=fmodel)
    assert isinstance(ax, plt.Axes)

    ax = layoutviz.plot_waking_directions(fmodel=fmodel)
    assert isinstance(ax, plt.Axes)

    # Add additional turbines to test plot farm terrain
    fmodel.set(
        layout_x=[0, 1000, 0, 1000, 3000],
        layout_y=[0, 0, 2000, 2000, 3000],
    )
    ax = layoutviz.plot_farm_terrain(fmodel=fmodel)
    assert isinstance(ax, plt.Axes)
