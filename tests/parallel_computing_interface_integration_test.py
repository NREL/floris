
import copy

import numpy as np

from floris.tools import FlorisInterface, ParallelComputingInterface
from tests.conftest import (
    assert_results_arrays,
)


DEBUG = True
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"


def test_parallel_turbine_powers(sample_inputs_fixture):
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

    fi_serial = FlorisInterface(sample_inputs_fixture.floris)
    fi_parallel_input = copy.deepcopy(fi_serial)
    fi_serial.calculate_wake()

    serial_turbine_powers = fi_serial.get_turbine_powers()

    fi_parallel = ParallelComputingInterface(
        fi=fi_parallel_input,
        max_workers=2,
        n_wind_condition_splits=2,
        interface="concurrent",
        print_timings=False,
    )

    parallel_turbine_powers = fi_parallel.get_turbine_powers()

    if DEBUG:
        print(serial_turbine_powers)
        print(parallel_turbine_powers)

    assert_results_arrays(parallel_turbine_powers, serial_turbine_powers)
