import numpy as np

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
)
from floris.optimization.load_optimization.load_optimization import (
    compute_farm_revenue,
    compute_farm_voc,
    compute_load_ti,
    compute_net_revenue,
    compute_turbine_voc,
    find_A_to_satisfy_rev_voc_ratio,
    get_max_powers,
    get_rotor_diameters,
    optimize_power_setpoints,
)


def test_get_max_powers():
    fmodel = FlorisModel(configuration="defaults")
    fmodel.set(layout_x=[0, 500.0], layout_y=[0.0, 0.0])
    rated_powers = get_max_powers(fmodel)
    np.testing.assert_allclose(rated_powers, [5e6, 5e6], atol=1e-4)


def test_get_rotor_diameters():
    fmodel = FlorisModel(configuration="defaults")
    fmodel.set(layout_x=[0, 500.0], layout_y=[0.0, 0.0])
    rotor_diameters = get_rotor_diameters(fmodel)
    np.testing.assert_allclose(rotor_diameters, [125.88, 125.88], atol=1e-4)


def test_compute_load_ti_no_wake():
    # If we pass in a two-turbine, two-findex case where the turbines are
    # not aligned in flow, would expect to get back np.ones((2,2)) * ambient
    fmodel = FlorisModel(configuration="defaults")
    fmodel.set(layout_x=[0, 500.0], layout_y=[0.0, 0.0])
    fmodel.set(
        wind_speeds=[8.0, 8.0], wind_directions=[0.0, 0.0], turbulence_intensities=[0.06, 0.06]
    )
    fmodel.run()
    load_ambient_tis = [0.12, 0.1]
    load_ti = compute_load_ti(fmodel, load_ambient_tis)
    assert load_ti.shape == (2, 2)
    assert load_ti[0, 0] == load_ambient_tis[0]
    assert load_ti[0, 1] == load_ambient_tis[0]
    assert load_ti[1, 0] == load_ambient_tis[1]
    assert load_ti[1, 1] == load_ambient_tis[1]


def test_compute_load_ti_wake():
    # Test two turbines in a wake, n_findex = 1
    fmodel = FlorisModel(configuration="defaults")
    fmodel.set(layout_x=[0, 500.0], layout_y=[0.0, 0.0])
    fmodel.set(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])
    fmodel.run()
    load_ambient_tis = [0.1]
    load_ti = compute_load_ti(fmodel, load_ambient_tis)
    assert load_ti.shape == (1, 2)
    np.testing.assert_almost_equal(load_ti[0, 0], load_ambient_tis)
    assert load_ti[0, 1] > load_ti[0, 0]


def test_compute_turbine_voc_no_wake():
    # If we pass in a two-turbine, two-findex case where the turbines are
    # not aligned in flow, would expect to get back a 2x2 numpy array where
    # all values are the same
    fmodel = FlorisModel(configuration="defaults")
    fmodel.set(layout_x=[0, 500.0], layout_y=[0.0, 0.0])
    fmodel.set(
        wind_speeds=[8.0, 8.0], wind_directions=[0.0, 0.0], turbulence_intensities=[0.06, 0.06]
    )
    fmodel.run()
    load_ambient_tis = [0.12, 0.12]
    voc = compute_turbine_voc(fmodel, 0.01, load_ambient_tis)
    assert voc.shape == (2, 2)
    assert voc[0, 0] == voc[0, 1]
    assert voc[0, 0] == voc[1, 0]
    assert voc[0, 0] == voc[1, 1]


def test_compute_turbien_voc_wake():
    # Test two turbines in a wake, n_findex = 1
    fmodel = FlorisModel(configuration="defaults")
    fmodel.set(layout_x=[0, 500.0], layout_y=[0.0, 0.0])
    fmodel.set(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])
    fmodel.run()
    load_ambient_tis = [0.1]
    voc = compute_turbine_voc(fmodel, 0.01, load_ambient_tis)
    assert voc.shape == (1, 2)
    assert voc[0, 1] > voc[0, 0]


def test_compute_net_revenue_no_wake():
    # Assuming two turbines, two findex, no wake, and uniform value
    # net_revenue should be a 2-element array with the same value
    fmodel = FlorisModel(configuration="defaults")
    fmodel.set(layout_x=[0, 500.0], layout_y=[0.0, 0.0])
    time_series = TimeSeries(
        wind_directions=np.array([0, 0]),
        wind_speeds=8.0,
        turbulence_intensities=0.06,
        values=np.array([0.5, 0.5]),
    )

    fmodel.set(wind_data=time_series)
    fmodel.run()
    load_ambient_tis = [0.12, 0.12]
    net_revenue = compute_net_revenue(fmodel, 1, load_ambient_tis)
    assert net_revenue.shape == (2,)
    assert net_revenue[0] == net_revenue[1]

def test_find_A_to_satisfy_rev_voc_ratio():
    # Test the function that finds the A value that satisfies the revenue to voc ratio
    target_rev_voc_ratio = 10.0

    fmodel = FlorisModel(configuration="defaults")
    fmodel.set(layout_x=[0, 500.0], layout_y=[0.0, 0.0])
    N = 100
    time_series = TimeSeries(
        wind_directions=np.ones(N) * 270.0,
        wind_speeds=8.0,
        turbulence_intensities=0.06,
        values=np.ones(N)
    )
    fmodel.set(wind_data=time_series)
    fmodel.run()

    load_ambient_tis = np.ones(N) * 0.1

    A = find_A_to_satisfy_rev_voc_ratio(fmodel, target_rev_voc_ratio, load_ambient_tis)

    farm_revenue = compute_farm_revenue(fmodel)

    farm_voc = compute_farm_voc(fmodel, A, load_ambient_tis)

    assert np.allclose(farm_revenue.sum() / farm_voc.sum(), target_rev_voc_ratio, atol=1e-4)
