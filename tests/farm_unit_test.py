from copy import deepcopy

import attr
import numpy as np
import pytest
import xarray as xr

from src.farm import Farm
from src.utilities import Vec3
from tests.conftest import SampleInputs


def test_farm_init_homogenous_turbines():
    turbine_data = SampleInputs().turbine
    farm_data = SampleInputs().farm

    layout_x = farm_data["layout_x"]
    layout_y = farm_data["layout_y"]
    wtg_id = [f"WTG_{str(i).zfill(3)}" for i in range(len(layout_x))]
    turbine_id = ["t1"] * len(layout_x)
    turbine_map = dict(t1=turbine_data)

    coordinates = [Vec3([x, y, 90.0]) for x, y in zip(layout_x, layout_y)]

    farm = Farm(turbine_id, turbine_map, layout_x, layout_y, wtg_id)

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
    farm = Farm(turbine_id, turbine_map, layout_x, layout_y)
    assert farm.wtg_id == ["t0001", "t0002", "t0003"]

    # Check the layout_x validator
    layout_x_fail_too_few = layout_x[:-1]
    with pytest.raises(ValueError):
        Farm(turbine_id, turbine_map, layout_x_fail_too_few, layout_y)

    layout_x_fail_too_many = deepcopy(layout_x)
    layout_x_fail_too_many.append(3)
    with pytest.raises(ValueError):
        Farm(turbine_id, turbine_map, layout_x_fail_too_many, layout_y)

    # Check the layout_y validator
    layout_y_fail_too_few = layout_y[:-1]
    with pytest.raises(ValueError):
        Farm(turbine_id, turbine_map, layout_x, layout_y_fail_too_few)

    layout_y_fail_too_many = deepcopy(layout_y)
    layout_y_fail_too_many.append(3)
    with pytest.raises(ValueError):
        Farm(turbine_id, turbine_map, layout_x, layout_y_fail_too_many)


def test_sort_turbines():
    turbine_data = SampleInputs().turbine
    farm_data = SampleInputs().farm

    layout_x = [farm_data["layout_x"]]
    layout_y = [630, 0, 1260]
    turbine_id = ["t1"] * len(layout_x)
    turbine_map = dict(t1=turbine_data)

    farm = Farm(turbine_id, turbine_map, layout_x, layout_y)

    x_sort = farm.sort_turbines(by="x")
    assert np.all(x_sort == np.array([0, 1, 2]))

    y_sort = farm.sort_turbines(by="y")
    assert np.all(y_sort == np.array([1, 0, 2]))


def test_make_fail_to_demonstrate_xarray_properties():
    turbine_data = SampleInputs().turbine
    farm_data = SampleInputs().farm

    layout_x = farm_data["layout_x"]
    layout_y = farm_data["layout_y"]
    wtg_id = [f"WTG_{str(i).zfill(3)}" for i in range(len(layout_x))]
    turbine_id = ["t1"] * len(layout_x)
    turbine_map = dict(t1=turbine_data)

    farm = Farm(turbine_id, turbine_map, layout_x, layout_y, wtg_id)
    print("The array:\n", farm.data_array)
    print("\n----------\n")
    print("Columns:", farm.data_array.turbine_attributes)
    print(
        "\nx coordinates:", farm.data_array.layout_x
    )  # what an attribute look like (there are layout_x and layout_y)
    print(
        "\nWTG IDs:", farm.data_array.wtg_id
    )  # what a coordinate looks like (this one is dim 0)
    assert 1 == 2
