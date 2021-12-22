from copy import deepcopy

import attr
import numpy as np
import pytest
import xarray as xr

from tests.conftest import SampleInputs
from floris.utilities import Vec3
from floris.simulation import Farm

from tests.conftest import (
    X_COORDS,
    Y_COORDS,
    Z_COORDS,
    N_TURBINES,
    WIND_SPEEDS,
    N_WIND_SPEEDS,
    TURBINE_GRID_RESOLUTION,
    WIND_DIRECTIONS,
    N_WIND_DIRECTIONS,
)

def test_farm_init_homogenous_turbines():
    farm_data = SampleInputs().farm
    turbine_data = SampleInputs().turbine

    layout_x = farm_data["layout_x"]
    layout_y = farm_data["layout_y"]

    coordinates = [Vec3([x, y, turbine_data["hub_height"]]) for x, y in zip(layout_x, layout_y)]

    farm = Farm(
        n_wind_directions=N_WIND_DIRECTIONS,
        n_wind_speeds=N_WIND_SPEEDS,
        layout_x=layout_x,
        layout_y=layout_y,
        turbine=turbine_data
    )

    # Check initial values
    assert farm.coordinates == coordinates
    assert isinstance(farm.layout_x, np.ndarray)
    assert isinstance(farm.layout_y, np.ndarray)

    # Check generated values
    assert np.all(farm.rotor_diameter == turbine_data["rotor_diameter"])
    assert np.all(farm.hub_height == turbine_data["hub_height"])
    assert np.all(farm.pP == turbine_data["pP"])
    assert np.all(farm.pT == turbine_data["pT"])
    assert np.all(farm.generator_efficiency == turbine_data["generator_efficiency"])

