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

from floris.simulation import WakeModelManager
from tests.conftest import SampleInputs

def test_asdict(sample_inputs_fixture: SampleInputs):

    wake_model_manager = WakeModelManager.from_dict(sample_inputs_fixture.wake)
    dict1 = wake_model_manager.as_dict()

    new_wake = WakeModelManager.from_dict(dict1)
    dict2 = new_wake.as_dict()

    assert dict1 == dict2
