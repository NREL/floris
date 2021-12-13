from copy import deepcopy

import attr
import numpy as np
import pytest
import xarray as xr

from tests.conftest import SampleInputs
from floris.utilities import Vec3
from floris.simulation import Farm


def test_farm_init_homogenous_turbines():
    farm_data = SampleInputs().farm
    flow_field_data = SampleInputs().flow_field
    turbine_data = SampleInputs().turbine

    wind_directions = flow_field_data["wind_directions"]
    wind_speeds = flow_field_data["wind_speeds"]
    layout_x = farm_data["layout_x"]
    layout_y = farm_data["layout_y"]
    turbine_id = ["test_turb"] * len(layout_x)
    turbine_map = deepcopy(turbine_data)

    coordinates = [Vec3([x, y, 90.0]) for x, y in zip(layout_x, layout_y)]

    farm = Farm(
        turbine_id=turbine_id,
        turbine_map=turbine_map,
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    # Check initial values
    assert farm.coordinates == coordinates
    assert isinstance(farm.layout_x, np.ndarray)
    assert isinstance(farm.layout_y, np.ndarray)

    # Check generated values
    assert np.all(farm.rotor_diameter == turbine_data["test_turb"]["rotor_diameter"])
    assert np.all(farm.hub_height == turbine_data["test_turb"]["hub_height"])
    assert np.all(farm.pP == turbine_data["test_turb"]["pP"])
    assert np.all(farm.pT == turbine_data["test_turb"]["pT"])
    assert np.all(farm.generator_efficiency == turbine_data["test_turb"]["generator_efficiency"])

    # Shape should be N turbines x 10 turbine attributes
    assert isinstance(farm.array_data, xr.DataArray)
    assert farm.array_data.shape == (len(wind_directions), len(wind_speeds), len(layout_x), 10)

    # Check the shape of one of the attribute arrays
    assert farm.rotor_diameter.shape == (
        len(wind_directions),
        len(wind_speeds),
        len(layout_x),
    )

    # Check the layout_x validator
    layout_x_fail_too_few = layout_x[:-1]
    with pytest.raises(ValueError):
        Farm(
            turbine_id=turbine_id,
            turbine_map=turbine_map,
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            layout_x=layout_x_fail_too_few,
            layout_y=layout_y,
        )

    layout_x_fail_too_many = deepcopy(layout_x)
    layout_x_fail_too_many.append(3)
    with pytest.raises(ValueError):
        Farm(
            turbine_id=turbine_id,
            turbine_map=turbine_map,
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            layout_x=layout_x_fail_too_many,
            layout_y=layout_y,
        )

    # Check the layout_y validator
    layout_y_fail_too_few = layout_y[:-1]
    with pytest.raises(ValueError):
        Farm(
            turbine_id=turbine_id,
            turbine_map=turbine_map,
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            layout_x=layout_x,
            layout_y=layout_y_fail_too_few,
        )

    layout_y_fail_too_many = deepcopy(layout_y)
    layout_y_fail_too_many.append(3)
    with pytest.raises(ValueError):
        Farm(
            turbine_id=turbine_id,
            turbine_map=turbine_map,
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            layout_x=layout_x,
            layout_y=layout_y_fail_too_many,
        )


def test_sort_turbines():
    farm_data = SampleInputs().farm
    flow_field_data = SampleInputs().flow_field
    turbine_data = SampleInputs().turbine

    wind_directions = flow_field_data["wind_directions"]
    wind_speeds = flow_field_data["wind_speeds"]
    layout_x = farm_data["layout_x"]
    layout_y = [630, 0, 1260]
    turbine_id = ["test_turb"] * len(layout_x)
    turbine_map = deepcopy(turbine_data)

    farm = Farm(
        turbine_id=turbine_id,
        turbine_map=turbine_map,
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    x_sort = farm.sort_turbines(by="x")
    assert np.all(x_sort == np.array([0, 1, 2]))

    y_sort = farm.sort_turbines(by="y")
    assert np.all(y_sort == np.array([1, 0, 2]))


def test_make_fail_to_demonstrate_xarray_properties():
    farm_data = SampleInputs().farm
    flow_field_data = SampleInputs().flow_field
    turbine_data = SampleInputs().turbine

    wind_directions = flow_field_data["wind_directions"]
    wind_speeds = flow_field_data["wind_speeds"]
    layout_x = farm_data["layout_x"]
    layout_y = farm_data["layout_y"]
    turbine_id = ["test_turb"] * len(layout_x)
    turbine_map = deepcopy(turbine_data)

    farm = Farm(
        turbine_id=turbine_id,
        turbine_map=turbine_map,
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        layout_x=layout_x,
        layout_y=layout_y,
    )
    print("The array:\n", farm.array_data)
    print("\n----------\n")
    print("Columns:", farm.array_data.turbine_attributes)
    print("\nx coordinates:", farm.array_data.layout_x)  # what an attribute look like (there are layout_x and layout_y)
    assert True


def test_turbine_array_rotor_diameter():
    """
    This tests the rotor diameter column of the turbine array
    but it also verifies the other columns of type float.
    """
    farm_data = SampleInputs().farm
    flow_field_data = SampleInputs().flow_field
    turbine_data = SampleInputs().turbine

    wind_directions = flow_field_data["wind_directions"]
    wind_speeds = flow_field_data["wind_speeds"]
    layout_x = farm_data["layout_x"]
    layout_y = farm_data["layout_y"]
    turbine_id = ["test_turb"] * len(layout_x)
    turbine_map = deepcopy(turbine_data)

    n_wind_directions = len(flow_field_data["wind_directions"])
    n_wind_speeds = len(flow_field_data["wind_speeds"])
    n_turbines = len(farm_data["layout_x"])

    farm = Farm(
        turbine_id=turbine_id,
        turbine_map=turbine_map,
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        layout_x=layout_x,
        layout_y=layout_y,
    )
    assert np.array_equal(
        farm.rotor_diameter,
        np.array(
            n_wind_directions * n_wind_speeds * n_turbines * [turbine_data["test_turb"]["rotor_diameter"]]
        ).reshape((n_wind_directions, n_wind_speeds, n_turbines)),
    )
