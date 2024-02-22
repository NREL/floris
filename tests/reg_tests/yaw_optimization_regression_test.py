
import numpy as np
import pandas as pd

from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_geometric import (
    YawOptimizationGeometric,
)
from floris.tools.optimization.yaw_optimization.yaw_optimizer_scipy import YawOptimizationScipy
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

baseline_serial_refine = pd.DataFrame(
        {
        "wind_direction": [0.0, 90.0, 180.0, 270.0],
        "wind_speed": [8.0] * 4,
        "turbulence_intensity": [0.1] * 4,
        "yaw_angles_opt": [
            [0.0, 0.0, 0.0],
            [0.0, 25.0, 15.625],
            [0.0, 0.0, 0.0],
            [15.625, 25.0, 0.0],
        ],
        "farm_power_opt": [5.261863e+06, 3.262218e+06, 5.261863e+06, 3.262218e+06],
        "farm_power_baseline": [5.261863e+06, 3.206038e+06, 5.261863e+06, 3.206038e+06],
    }
)

baseline_geometric_yaw = pd.DataFrame(
        {
        "wind_direction": [0.0, 90.0, 180.0, 270.0],
        "wind_speed": [8.0] * 4,
        "turbulence_intensity": [0.1] * 4,
        "yaw_angles_opt": [
            [0.0, 0.0, 0.0],
            [0.0, 19.9952335557674, 19.9952335557674],
            [0.0, 0.0, 0.0],
            [19.9952335557674, 19.9952335557674, 0.0],
        ],
        "farm_power_opt": [5.261863e+06, 3.252509e+06, 5.261863e+06, 3.252509e+06],
        "farm_power_baseline": [5.261863e+06, 3.206038e+06, 5.261863e+06, 3.206038e+06],
    }
)

baseline_scipy = pd.DataFrame(
        {
        "wind_direction": [0.0, 90.0, 180.0, 270.0],
        "wind_speed": [8.0] * 4,
        "turbulence_intensity": [0.1] * 4,
        "yaw_angles_opt": [
            [0.0, 0.0, 0.0],
            [0.0, 24.999999999999982, 12.165643400939755],
            [0.0, 0.0, 0.0],
            [12.165643399558299, 25.0, 0.0],
        ],
        "farm_power_opt": [5.261863e+06, 3.264975e+06, 5.261863e+06, 3.264975e+06],
        "farm_power_baseline": [5.261863e+06, 3.206038e+06, 5.261863e+06, 3.206038e+06],
    }
)


def test_serial_refine(sample_inputs_fixture):
    """
    The Serial Refine (SR) method optimizes yaw angles based on a sequential, iterative yaw
    optimization scheme. This test compares the optimization results from the SR method for
    a simple farm with a simple wind rose to stored baseline results.
    """
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fi = FlorisInterface(sample_inputs_fixture.floris)
    wd_array = np.arange(0.0, 360.0, 90.0)
    ws_array = 8.0 * np.ones_like(wd_array)
    D = 126.0 # Rotor diameter for the NREL 5 MW
    fi.reinitialize(
        layout_x=[0.0, 5 * D, 10 * D],
        layout_y=[0.0, 0.0, 0.0],
        wind_directions=wd_array,
        wind_speeds=ws_array,
    )

    yaw_opt = YawOptimizationSR(fi)
    df_opt = yaw_opt.optimize()

    if DEBUG:
        print(baseline_serial_refine.to_string())
        print(df_opt.to_string())

    pd.testing.assert_frame_equal(df_opt, baseline_serial_refine)


def test_geometric_yaw(sample_inputs_fixture):
    """
    The Geometric Yaw optimization method optimizes yaw angles using geometric data and derived
    optimal yaw relationships. This test compares the optimization results from the Geometric Yaw
    optimizaiton for a simple farm with a simple wind rose to stored baseline results.
    """
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fi = FlorisInterface(sample_inputs_fixture.floris)
    wd_array = np.arange(0.0, 360.0, 90.0)
    ws_array = 8.0 * np.ones_like(wd_array)
    D = 126.0 # Rotor diameter for the NREL 5 MW
    fi.reinitialize(
        layout_x=[0.0, 5 * D, 10 * D],
        layout_y=[0.0, 0.0, 0.0],
        wind_directions=wd_array,
        wind_speeds=ws_array,
    )
    fi.calculate_wake()
    baseline_farm_power = fi.get_farm_power().squeeze()

    yaw_opt = YawOptimizationGeometric(fi)
    df_opt = yaw_opt.optimize()

    yaw_angles_opt_geo = np.vstack(yaw_opt.yaw_angles_opt)
    fi.calculate_wake(yaw_angles=yaw_angles_opt_geo)
    geo_farm_power = fi.get_farm_power().squeeze()

    df_opt['farm_power_baseline'] = baseline_farm_power
    df_opt['farm_power_opt'] = geo_farm_power

    if DEBUG:
        print(baseline_geometric_yaw.to_string())
        print(df_opt.to_string())

    pd.testing.assert_frame_equal(df_opt, baseline_geometric_yaw)


def test_scipy_yaw_opt(sample_inputs_fixture):
    """
    The SciPy optimization method optimizes yaw angles using SciPy's minimize method. This test
    compares the optimization results from the SciPy yaw optimizaiton for a simple farm with a
    simple wind rose to stored baseline results.
    """
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    opt_options = {
        "maxiter": 5,
        "disp": True,
        "iprint": 2,
        "ftol": 1e-12,
        "eps": 0.5,
    }

    fi = FlorisInterface(sample_inputs_fixture.floris)
    wd_array = np.arange(0.0, 360.0, 90.0)
    ws_array = 8.0 * np.ones_like(wd_array)
    D = 126.0 # Rotor diameter for the NREL 5 MW
    fi.reinitialize(
        layout_x=[0.0, 5 * D, 10 * D],
        layout_y=[0.0, 0.0, 0.0],
        wind_directions=wd_array,
        wind_speeds=ws_array,
    )

    yaw_opt = YawOptimizationScipy(fi, opt_options=opt_options)
    df_opt = yaw_opt.optimize()

    if DEBUG:
        print(baseline_scipy.to_string())
        print(df_opt.to_string())

    pd.testing.assert_frame_equal(df_opt, baseline_scipy)
