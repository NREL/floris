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
from tests.conftest import X_COORDS, Y_COORDS, Z_COORDS, ROTOR_DIAMETER, assert_results_arrays
import numpy as np

VELOCITY_MODEL = "none"
TURBULENCE_INTENSITY_I = 0.06
AXIAL_INDUCTION_I = 0.3
DEFLECTION_FIELD_I = 0.0
YAW_ANGLE_I = 0.0
CT_I = 0.2

def test_wake_deflection_none(sample_inputs_fixture: SampleInputs):
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL

    floris = Floris.from_dict(sample_inputs_fixture.floris)
    floris.initialize_domain()

    deficit_model_args = floris.wake.velocity_model.prepare_function(floris.grid, floris.flow_field)

    velocity_deficit = floris.wake.velocity_model.function(
        X_COORDS,
        Y_COORDS,
        Z_COORDS,
        AXIAL_INDUCTION_I,
        DEFLECTION_FIELD_I,
        YAW_ANGLE_I,
        TURBULENCE_INTENSITY_I,
        CT_I,
        ROTOR_DIAMETER / 2,
        ROTOR_DIAMETER,
        **deficit_model_args
    )

    assert_results_arrays(velocity_deficit, np.zeros_like(deficit_model_args['u_initial']))
