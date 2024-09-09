
import copy
import logging

import numpy as np
import pytest

from floris import (
    FlorisModel,
    UncertainFlorisModel,
)
from floris.parallel_floris_model_2 import ParallelFlorisModel
from tests.conftest import (
    assert_results_arrays,
)


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

def test_None_interface(sample_inputs_fixture):
    """
    With interface=None, the ParallelFlorisModel should behave exactly like the FlorisModel.
    (ParallelFlorisModel.run() simply calls the parent FlorisModel.run()).
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel = ParallelFlorisModel(
        sample_inputs_fixture.core,
        interface=None,
        n_wind_condition_splits=2 # Not used when interface=None
    )

    fmodel.run()
    pfmodel.run()

    f_turb_powers = fmodel.get_turbine_powers()
    pf_turb_powers = pfmodel.get_turbine_powers()

    assert np.allclose(f_turb_powers, pf_turb_powers)

def test_multiprocessing_interface(sample_inputs_fixture):
    """
    With interface="multiprocessing", the ParallelFlorisModel should return the same powers
    as the FlorisModel.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel = ParallelFlorisModel(
        sample_inputs_fixture.core,
        interface="multiprocessing",
        n_wind_condition_splits=2
    )

    fmodel.run()
    pfmodel.run()

    f_turb_powers = fmodel.get_turbine_powers()
    pf_turb_powers = pfmodel.get_turbine_powers()

    assert np.allclose(f_turb_powers, pf_turb_powers)

def test_return_turbine_powers_only(sample_inputs_fixture):
    """
    With return_turbine_powers_only=True, the ParallelFlorisModel should return only the
    turbine powers, not the full results.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel = ParallelFlorisModel(
        sample_inputs_fixture.core,
        interface="multiprocessing",
        n_wind_condition_splits=2,
        return_turbine_powers_only=True
    )

    fmodel.run()
    pfmodel.run()

    f_turb_powers = fmodel.get_turbine_powers()
    pf_turb_powers = pfmodel.get_turbine_powers()

    assert np.allclose(f_turb_powers, pf_turb_powers)

def test_run_error(sample_inputs_fixture, caplog):
    """
    Check that an error is raised if an output is requested before calling run().
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    pfmodel = ParallelFlorisModel(
        sample_inputs_fixture.core,
        interface="multiprocessing",
        n_wind_condition_splits=2
    )

    # In future versions, error will be raised
    # with pytest.raises(RuntimeError):
    #     pfmodel.get_turbine_powers()
    # with pytest.raises(RuntimeError):
    #     pfmodel.get_farm_AEP()

    # For now, only a warning is raised for backwards compatibility
    with caplog.at_level(logging.WARNING):
        pfmodel.get_turbine_powers()
