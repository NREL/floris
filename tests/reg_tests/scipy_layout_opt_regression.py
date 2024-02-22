
import numpy as np
import pandas as pd

from floris.tools import FlorisInterface
from floris.tools.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy,
)
from tests.conftest import (
    assert_results_arrays,
)


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

baseline = np.array(
    [
        [0.00000000e+00, 4.96470529e+02, 1.00000000e+03],
        [4.58108861e-15, 1.09603647e+01, 2.47721427e+01],
    ]
)


def test_scipy_layout_opt(sample_inputs_fixture):
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

    opt_options = {
        "maxiter": 5,
        "disp": True,
        "iprint": 2,
        "ftol": 1e-12,
        "eps": 0.01,
    }

    boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]

    fi = FlorisInterface(sample_inputs_fixture.floris)
    wd_array = np.arange(0, 360.0, 5.0)
    ws_array = 8.0 * np.ones_like(wd_array)
    D = 126.0 # Rotor diameter for the NREL 5 MW
    fi.reinitialize(
        layout_x=[0.0, 5 * D, 10 * D],
        layout_y=[0.0, 0.0, 0.0],
        wind_directions=wd_array,
        wind_speeds=ws_array,
    )

    layout_opt = LayoutOptimizationScipy(fi, boundaries, optOptions=opt_options)
    sol = layout_opt.optimize()
    locations_opt = np.array([sol[0], sol[1]])

    if DEBUG:
        print(baseline)
        print(locations_opt)

    assert_results_arrays(locations_opt, baseline)
