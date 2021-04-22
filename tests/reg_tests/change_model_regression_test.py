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


from tests.conftest import assert_results, print_test_values, turbines_to_array
from tests.reg_tests.gauss_regression_test import baseline as gauss_baseline
from tests.reg_tests.curl_regression_test import baseline as curl_baseline
from src import Floris

DEBUG = False


def test_gauss_to_curl_to_gauss(sample_inputs_fixture):
    """
    Start with the Gauss wake model
    Then, switch to Curl
    Then, switch back to Gauss
    """
    # Establish that the Gauss test passes
    sample_inputs_fixture.floris["wake"]["properties"]["velocity_model"] = "gauss_legacy"
    sample_inputs_fixture.floris["wake"]["properties"]["deflection_model"] = "gauss"
    floris = Floris(input_dict=sample_inputs_fixture.floris)
    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    assert_results(test_results, gauss_baseline)

    # Change the model to Curl, rerun calculate_wake, and compare to Curl
    floris.farm.set_wake_model("curl")
    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    assert_results(test_results, curl_baseline)

    # Change back to Gauss, rerun calculate_wake, and compare to gauss
    floris.farm.set_wake_model("gauss_legacy")
    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    assert_results(test_results, gauss_baseline)
