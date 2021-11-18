from copy import deepcopy

import attr
import numpy as np
import pytest
import xarray as xr

from src.utilities import Vec3
from src.simulation import Farm
from tests.conftest import SampleInputs


def test_farm_init_homogenous_turbines():
    turbine_data = SampleInputs().turbine
    farm_data = SampleInputs().farm

    wind_directions = farm_data["wind_directions"]
    wind_speeds = farm_data["wind_speeds"]
    layout_x = farm_data["layout_x"]
    layout_y = farm_data["layout_y"]
    wtg_id = [f"WTG_{str(i).zfill(3)}" for i in range(len(layout_x))]
    turbine_id = ["t1"] * len(layout_x)
    turbine_map = dict(t1=turbine_data)
    wind_shear = farm_data["wind_shear"]
    wind_veer = farm_data["wind_veer"]
    reference_wind_height = farm_data["reference_wind_height"]
    reference_turbine_diameter = farm_data["reference_turbine_diameter"]
    air_density = farm_data["air_density"]

    coordinates = [Vec3([x, y, 90.0]) for x, y in zip(layout_x, layout_y)]

    farm = Farm(
        turbine_id=turbine_id,
        turbine_map=turbine_map,
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        layout_x=layout_x,
        layout_y=layout_y,
        wtg_id=wtg_id,
        wind_shear=wind_shear,
        wind_veer=wind_veer,
        air_density=air_density,
        reference_wind_height=reference_wind_height,
        reference_turbine_diameter=reference_turbine_diameter,
    )

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
    assert isinstance(farm.array_data, xr.DataArray)
    assert farm.array_data.shape == (len(wind_directions), len(wind_speeds), len(layout_x), 10)

    # Check the shape of one of the attribute arrays
    assert farm.rotor_diameter.shape == (
        len(wind_directions),
        len(wind_speeds),
        len(layout_x),
    )

    # Check generated WTG IDs is the default
    farm = Farm(
        turbine_id=turbine_id,
        turbine_map=turbine_map,
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        layout_x=layout_x,
        layout_y=layout_y,
        wind_shear=wind_shear,
        wind_veer=wind_veer,
        air_density=air_density,
        reference_wind_height=reference_wind_height,
        reference_turbine_diameter=reference_turbine_diameter,
    )
    assert farm.wtg_id == ["WTG_0001", "WTG_0002", "WTG_0003"]

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
            wtg_id=wtg_id,
            wind_shear=wind_shear,
            wind_veer=wind_veer,
            air_density=air_density,
            reference_wind_height=reference_wind_height,
            reference_turbine_diameter=reference_turbine_diameter,
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
            wind_shear=wind_shear,
            wind_veer=wind_veer,
            air_density=air_density,
            reference_wind_height=reference_wind_height,
            reference_turbine_diameter=reference_turbine_diameter,
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
            wind_shear=wind_shear,
            wind_veer=wind_veer,
            air_density=air_density,
            reference_wind_height=reference_wind_height,
            reference_turbine_diameter=reference_turbine_diameter,
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
            wind_shear=wind_shear,
            wind_veer=wind_veer,
            air_density=air_density,
            reference_wind_height=reference_wind_height,
            reference_turbine_diameter=reference_turbine_diameter,
        )


def test_sort_turbines():
    turbine_data = SampleInputs().turbine
    farm_data = SampleInputs().farm

    wind_directions = farm_data["wind_directions"]
    wind_speeds = farm_data["wind_speeds"]
    layout_x = farm_data["layout_x"]
    layout_y = [630, 0, 1260]
    turbine_id = ["t1"] * len(layout_x)
    turbine_map = dict(t1=turbine_data)
    wind_shear = farm_data["wind_shear"]
    wind_veer = farm_data["wind_veer"]
    reference_wind_height = farm_data["reference_wind_height"]
    reference_turbine_diameter = farm_data["reference_turbine_diameter"]
    air_density = farm_data["air_density"]

    farm = Farm(
        turbine_id=turbine_id,
        turbine_map=turbine_map,
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        layout_x=layout_x,
        layout_y=layout_y,
        wind_shear=wind_shear,
        wind_veer=wind_veer,
        air_density=air_density,
        reference_wind_height=reference_wind_height,
        reference_turbine_diameter=reference_turbine_diameter,
    )

    x_sort = farm.sort_turbines(by="x")
    assert np.all(x_sort == np.array([0, 1, 2]))

    y_sort = farm.sort_turbines(by="y")
    assert np.all(y_sort == np.array([1, 0, 2]))


def test_make_fail_to_demonstrate_xarray_properties():
    turbine_data = SampleInputs().turbine
    farm_data = SampleInputs().farm

    wind_directions = farm_data["wind_directions"]
    wind_speeds = farm_data["wind_speeds"]
    layout_x = farm_data["layout_x"]
    layout_y = farm_data["layout_y"]
    wtg_id = [f"WTG_{str(i).zfill(3)}" for i in range(len(layout_x))]
    turbine_id = ["t1"] * len(layout_x)
    turbine_map = dict(t1=turbine_data)
    wind_shear = farm_data["wind_shear"]
    wind_veer = farm_data["wind_veer"]
    reference_wind_height = farm_data["reference_wind_height"]
    reference_turbine_diameter = farm_data["reference_turbine_diameter"]
    air_density = farm_data["air_density"]

    farm = Farm(
        turbine_id=turbine_id,
        turbine_map=turbine_map,
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        layout_x=layout_x,
        layout_y=layout_y,
        wind_shear=wind_shear,
        wind_veer=wind_veer,
        air_density=air_density,
        reference_wind_height=reference_wind_height,
        reference_turbine_diameter=reference_turbine_diameter,
        wtg_id=wtg_id,
    )
    print("The array:\n", farm.array_data)
    print("\n----------\n")
    print("Columns:", farm.array_data.turbine_attributes)
    print("\nx coordinates:", farm.array_data.layout_x)  # what an attribute look like (there are layout_x and layout_y)
    print("\nWTG IDs:", farm.array_data.wtg_id)  # what a coordinate looks like (this one is dim 0)
    assert True


def test_turbine_array_rotor_diameter():
    """
    This tests the rotor diameter column of the turbine array
    but it also verifies the other columns of type float.
    """
    turbine_data = SampleInputs().turbine
    farm_data = SampleInputs().farm

    wind_directions = farm_data["wind_directions"]
    wind_speeds = farm_data["wind_speeds"]
    layout_x = farm_data["layout_x"]
    layout_y = farm_data["layout_y"]
    wtg_id = [f"WTG_{str(i).zfill(3)}" for i in range(len(layout_x))]
    turbine_id = ["t1"] * len(layout_x)
    turbine_map = dict(t1=turbine_data)
    wind_shear = farm_data["wind_shear"]
    wind_veer = farm_data["wind_veer"]
    reference_wind_height = farm_data["reference_wind_height"]
    reference_turbine_diameter = farm_data["reference_turbine_diameter"]
    air_density = farm_data["air_density"]

    n_wind_directions = len(farm_data["wind_directions"])
    n_wind_speeds = len(farm_data["wind_speeds"])
    n_turbines = len(farm_data["layout_x"])

    farm = Farm(
        turbine_id=turbine_id,
        turbine_map=turbine_map,
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        layout_x=layout_x,
        layout_y=layout_y,
        wind_shear=wind_shear,
        wind_veer=wind_veer,
        air_density=air_density,
        reference_wind_height=reference_wind_height,
        reference_turbine_diameter=reference_turbine_diameter,
        wtg_id=wtg_id,
    )
    assert np.array_equal(
        farm.rotor_diameter,
        np.array(n_wind_directions * n_wind_speeds * n_turbines * [turbine_data["rotor_diameter"]]).reshape(
            (n_wind_directions, n_wind_speeds, n_turbines)
        ),
    )
