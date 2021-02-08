# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import pytest

from floris.simulation import Floris


def test_gauss_to_curl_to_gauss(sample_inputs_fixture):
    """
    Start with the Gauss wake model
    Then, switch to Curl
    Then, switch back to Gauss
    """

    # Store the results from the first mode, Gauss
    floris = Floris(input_dict=sample_inputs_fixture.floris)
    floris.farm.set_wake_model("gauss_legacy")
    floris.farm.flow_field.calculate_wake()
    baseline = []
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        baseline.append([turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity])

    # Change the model to Curl, rerun calculate_wake, and compare to Curl
    floris.farm.set_wake_model("curl")
    floris.farm.flow_field.calculate_wake()

    # Change back to Gauss, rerun calculate_wake, and compare to gauss
    floris.farm.set_wake_model("gauss_legacy")
    floris.farm.flow_field.calculate_wake()
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        assert pytest.approx(turbine.Cp) == baseline[i][0]
        assert pytest.approx(turbine.Ct) == baseline[i][1]
        assert pytest.approx(turbine.power) == baseline[i][2]
        assert pytest.approx(turbine.aI) == baseline[i][3]
        assert pytest.approx(turbine.average_velocity) == baseline[i][4]
