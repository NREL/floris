
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
    The parallel computing interface behaves like the floris interface, but distributes
    calculations among available cores to speep up the necessary computations. This test compares
    the individual turbine powers computed with the parallel interface to those computed with
    the serial floris interface. The expected result is that the turbine powers should be
    exactly the same.
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
