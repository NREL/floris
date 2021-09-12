from copy import deepcopy

import attr
import numpy as np
import pytest
import xarray as xr

from src.farm import FarmGenerator
from src.utilities import Vec3
from tests.conftest import SampleInputs


def test_farm_init_homogenous_turbines():
    turbine_data = SampleInputs().turbine
    farm_data = SampleInputs().farm

    layout_x = farm_data["layout_x"]
    layout_y = farm_data["layout_y"]
    wtg_id = [f"WTG_{str(i).zfill(3)}" for i in range(len(layout_x))]
    turbine_id = ["t1"] * 3
    turbine_map = dict(t1=turbine_data)

    coordinates = [Vec3([x, y, 90.0]) for x, y in zip(layout_x, layout_y)]

    farm = FarmGenerator(turbine_id, turbine_map, layout_x, layout_y, wtg_id)

    # Check initial values
    assert farm.coordinates == coordinates
    assert isinstance(farm.layout_x, np.ndarray)
    assert isinstance(farm.layout_y, np.ndarray)
    assert farm.wtg_id == wtg_id

    # Check generated values
    assert np.all(farm.rotor_diameter == turbine_data["rotor_diameter"])
    assert np.all(farm.hub_height == turbine_data["hub_height"])
    assert np.all(farm.pP == turbine_data["pP"])
    assert np.all(farm.pT == turbine_data["pT"])
    assert np.all(farm.generator_efficiency == turbine_data["generator_efficiency"])

    # Shape should be N turbines x 10 turbine attributes
    assert isinstance(farm.data_array, xr.DataArray)
    assert farm.data_array.shape == (len(layout_x), 10)

    # Check the shape of one of the attribute arrays
    assert farm.rotor_diameter.shape == (len(layout_x),)

    # Check generated WTG IDs is the default
    farm = FarmGenerator(turbine_id, turbine_map, layout_x, layout_y)
    assert farm.wtg_id == ["t001", "t002", "t003"]

    # Check the layout_x validator
    layout_x_fail_too_few = layout_x[:-1]
    with pytest.raises(ValueError):
        FarmGenerator(turbine_id, turbine_map, layout_x_fail_too_few, layout_y)

    layout_x_fail_too_many = deepcopy(layout_x)
    layout_x_fail_too_many.append(3)
    with pytest.raises(ValueError):
        FarmGenerator(turbine_id, turbine_map, layout_x_fail_too_many, layout_y)

    # Check the layout_y validator
    layout_y_fail_too_few = layout_y[:-1]
    with pytest.raises(ValueError):
        FarmGenerator(turbine_id, turbine_map, layout_x, layout_y_fail_too_few)

    layout_y_fail_too_many = deepcopy(layout_y)
    layout_y_fail_too_many.append(3)
    with pytest.raises(ValueError):
        FarmGenerator(turbine_id, turbine_map, layout_x, layout_y_fail_too_many)
