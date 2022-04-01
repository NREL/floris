# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

from floris.simulation import Floris
from tests.conftest import SampleInputs
from tests.conftest import X_COORDS, Y_COORDS, ROTOR_DIAMETER, assert_results_arrays
import numpy as np
import pytest

DEFLECTION_MODEL = "none"
TURBULENCE_INTENSITY_I = 0.06
CT_I = 0.2

def test_wake_deflection_none(sample_inputs_fixture: SampleInputs):
    EFFECTIVE_YAW_I = 0.0

    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    floris = Floris.from_dict(sample_inputs_fixture.floris)
    floris.initialize_domain()

    deflection_model_args = floris.wake.deflection_model.prepare_function(floris.grid, floris.flow_field)

    deflection_field = floris.wake.deflection_model.function(
        X_COORDS,
        Y_COORDS,
        EFFECTIVE_YAW_I,
        TURBULENCE_INTENSITY_I,
        CT_I,
        ROTOR_DIAMETER,
        **deflection_model_args
    )

    assert_results_arrays(deflection_field, np.zeros_like(deflection_model_args['freestream_velocity']))

def test_wake_deflection_none_with_yaw(sample_inputs_fixture: SampleInputs):
    EFFECTIVE_YAW_I = 1.0

    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    floris = Floris.from_dict(sample_inputs_fixture.floris)
    floris.initialize_domain()

    deflection_model_args = floris.wake.deflection_model.prepare_function(floris.grid, floris.flow_field)

    with pytest.raises(ValueError):
        deflection_field = floris.wake.deflection_model.function(
            X_COORDS,
            Y_COORDS,
            EFFECTIVE_YAW_I,
            TURBULENCE_INTENSITY_I,
            CT_I,
            ROTOR_DIAMETER,
            **deflection_model_args
        )
