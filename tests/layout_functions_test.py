
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import floris.tools.layout_functions as lf
from floris.tools.floris_interface import FlorisInterface


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"


def test_get_wake_direction():
    # Turbine 0 wakes Turbine 1 at 270 degrees
    assert np.isclose(lf.get_wake_direction(0, 0, 1, 0), 270.0)

    # Turbine 0 wakes Turbine 1 at 0 degrees
    assert np.isclose(lf.get_wake_direction(0, 1, 0, 0), 0.0)

    # Winds from the south
    assert np.isclose(lf.get_wake_direction(0, -1, 0, 0), 180.0)

def test_plotting_functions():

    fi = FlorisInterface(configuration=YAML_INPUT)
    ax = lf.plot_turbine_points(fi=fi)
    assert isinstance(ax, plt.Axes)
    ax = lf.plot_turbine_labels(fi=fi)
    assert isinstance(ax, plt.Axes)
    ax = lf.plot_waking_directions(fi=fi)
    assert isinstance(ax, plt.Axes)
