
import numpy as np
import pytest

from floris.core import FlowField, TurbineGrid
from tests.conftest import N_FINDEX, N_TURBINES


def test_n_findex(flow_field_fixture):
    assert flow_field_fixture.n_findex == N_FINDEX


def test_initialize_velocity_field(flow_field_fixture, turbine_grid_fixture: TurbineGrid):
    flow_field_fixture.wind_shear = 1.0
    flow_field_fixture.initialize_velocity_field(turbine_grid_fixture)

    # Check the shape of the velocity arrays: u_initial, v_initial, w_initial  and u, v, w
    # Dimensions are (# findex, # turbines, N grid points, M grid points)
    assert np.shape(flow_field_fixture.u_sorted)[0] == flow_field_fixture.n_findex
    assert np.shape(flow_field_fixture.u_sorted)[1] == N_TURBINES
    assert np.shape(flow_field_fixture.u_sorted)[2] == turbine_grid_fixture.grid_resolution
    assert np.shape(flow_field_fixture.u_sorted)[3] == turbine_grid_fixture.grid_resolution

    # Check that the wind speed profile was created correctly. By setting the shear
    # exponent to 1.0 above, the shear profile is a linear function of height and
    # the points on the turbine rotor are equally spaced about the rotor.
    # This means that their average should equal the wind speed at the center
    # which is the input wind speed.
    shape = np.shape(flow_field_fixture.u_sorted[0, 0, :, :])
    n_elements = shape[0] * shape[1]
    average = (
        np.sum(flow_field_fixture.u_sorted[:, 0, :, :], axis=(-2, -1))
        / np.array([n_elements])
    )
    assert np.array_equal(average, flow_field_fixture.wind_speeds)


def test_asdict(flow_field_fixture: FlowField, turbine_grid_fixture: TurbineGrid):

    flow_field_fixture.initialize_velocity_field(turbine_grid_fixture)
    dict1 = flow_field_fixture.as_dict()

    new_ff = FlowField.from_dict(dict1)
    new_ff.initialize_velocity_field(turbine_grid_fixture)
    dict2 = new_ff.as_dict()

    assert dict1 == dict2

def test_len_ws_equals_len_wd(flow_field_fixture: FlowField, turbine_grid_fixture: TurbineGrid):

    flow_field_fixture.initialize_velocity_field(turbine_grid_fixture)
    dict1 = flow_field_fixture.as_dict()

    # Test that having the 3 equal in lenght raises no error
    dict1['wind_directions'] = np.array([180, 180])
    dict1['wind_speeds'] = np.array([5., 6.])
    dict1['turbulence_intensities'] = np.array([175., 175.])

    FlowField.from_dict(dict1)

    # Set the wind speeds as a different length of wind directions and turbulence_intensities
    # And confirm error raised
    dict1['wind_directions'] = np.array([180, 180])
    dict1['wind_speeds'] = np.array([5., 6., 7.])
    dict1['turbulence_intensities'] = np.array([175., 175.])

    with pytest.raises(ValueError):
        FlowField.from_dict(dict1)

    # Set the wind directions as a different length of wind speeds and turbulence_intensities
    dict1['wind_directions'] = np.array([180, 180, 180.])
    # And confirm error raised
    dict1['wind_speeds'] = np.array([5., 6.])
    dict1['turbulence_intensities'] = np.array([175., 175.])

    with pytest.raises(ValueError):
        FlowField.from_dict(dict1)

def test_dim_ws_wd_ti(flow_field_fixture: FlowField, turbine_grid_fixture: TurbineGrid):

    flow_field_fixture.initialize_velocity_field(turbine_grid_fixture)
    dict1 = flow_field_fixture.as_dict()

    # Test that having an extra dimension in wind_directions raises an error
    with pytest.raises(ValueError):
        dict1['wind_directions'] = np.array([[180, 180]])
        dict1['wind_speeds'] = np.array([5., 6.])
        dict1['turbulence_intensities'] = np.array([175., 175.])
        FlowField.from_dict(dict1)

    # Test that having an extra dimension in wind_speeds raises an error
    with pytest.raises(ValueError):
        dict1['wind_directions'] = np.array([180, 180])
        dict1['wind_speeds'] = np.array([[5., 6.]])
        dict1['turbulence_intensities'] = np.array([175., 175.])
        FlowField.from_dict(dict1)

    # Test that having an extra dimension in turbulence_intensities raises an error
    with pytest.raises(ValueError):
        dict1['wind_directions'] = np.array([180, 180])
        dict1['wind_speeds'] = np.array([5., 6.])
        dict1['turbulence_intensities'] = np.array([[175., 175.]])
        FlowField.from_dict(dict1)


def test_turbulence_intensities_to_n_findex(flow_field_fixture, turbine_grid_fixture):
    # Assert tubulence intensity has same length as n_findex
    assert len(flow_field_fixture.turbulence_intensities) == flow_field_fixture.n_findex

    # Assert turbulence_intensity_field is the correct shape
    flow_field_fixture.initialize_velocity_field(turbine_grid_fixture)
    assert flow_field_fixture.turbulence_intensity_field.shape == (N_FINDEX, N_TURBINES, 1, 1)

    # Assert that turbulence_intensity_field has values matched to turbulence_intensity
    for findex in range(N_FINDEX):
        for t in range(N_TURBINES):
            assert (
                flow_field_fixture.turbulence_intensities[findex]
                == flow_field_fixture.turbulence_intensity_field[findex, t, 0, 0]
            )
