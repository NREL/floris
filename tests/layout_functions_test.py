import pytest
import numpy as np
import floris.tools.layout_functions as lf


def test_get_wake_direction():
    # Turbine 0 wakes Turbine 1 at 270 degrees
    assert np.isclose(lf.get_wake_direction(0, 0, 1, 0), 270.0)

    # Turbine 0 wakes Turbine 1 at 0 degrees
    assert np.isclose(lf.get_wake_direction(0, 1, 0, 0), 0.0)

    # Winds from the south
    assert np.isclose(lf.get_wake_direction(0, -1, 0, 0), 180.0)
