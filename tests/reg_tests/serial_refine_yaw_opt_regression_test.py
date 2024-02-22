
import numpy as np
import pandas as pd

from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

baseline = pd.DataFrame(
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


def test_serial_refine(sample_inputs_fixture):
    """
    The calculate_no_wake function calculates the power production of a wind farm
    assuming no wake losses. It does this by initializing and finalizing the
    floris simulation while skipping the wake calculation. The power for all wind
    turbines should be the same for a uniform wind condition. The chosen wake model
    is not important since it will not actually be used. However, it is left enabled
    instead of using "None" so that additional tests can be constructed here such
    as one with yaw activated.
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
        print(baseline.to_string())
        print(df_opt.to_string())

    pd.testing.assert_frame_equal(df_opt, baseline)
