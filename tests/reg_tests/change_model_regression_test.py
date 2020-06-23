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


import numpy as np
import pytest

from floris.simulation import Floris


try:
    from .gauss_regression_test import GaussRegressionTest
except ImportError:
    from gauss_regression_test import GaussRegressionTest
try:
    from .curl_regression_test import CurlRegressionTest
except ImportError:
    from curl_regression_test import CurlRegressionTest


class ChangeModelTest:
    """
    This test checks the ability to change the wake models programmatically.
    These tests use the baselines from other regression tests.

    Currently, it tests the parameters on the Turbines.
    TODO:
    - Timing test
    - Memory test
    """

    def __init__(self):
        self.debug = False


def test_gauss_to_curl_to_gauss():
    """
    Start with the Gauss wake model
    Then, switch to Curl
    Then, switch back to Gauss
    """
    test_class = ChangeModelTest()

    # Establish that the Gauss test passes
    gauss_test_class = GaussRegressionTest()
    floris = Floris(input_dict=gauss_test_class.input_dict)
    floris.farm.flow_field.calculate_wake()
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        if test_class.debug:
            print(
                "({:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.7f})".format(
                    turbine.Cp,
                    turbine.Ct,
                    turbine.power,
                    turbine.aI,
                    turbine.average_velocity,
                )
            )
        baseline = gauss_test_class.baseline(i)
        assert pytest.approx(turbine.Cp) == baseline[0]
        assert pytest.approx(turbine.Ct) == baseline[1]
        assert pytest.approx(turbine.power) == baseline[2]
        assert pytest.approx(turbine.aI) == baseline[3]
        assert pytest.approx(turbine.average_velocity) == baseline[4]

    # Change the model to Curl, rerun calculate_wake, and compare to Curl
    floris.farm.set_wake_model("curl")
    floris.farm.flow_field.calculate_wake()

    curl_test_class = CurlRegressionTest()
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        if test_class.debug:
            print(
                "({:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.7f})".format(
                    turbine.Cp,
                    turbine.Ct,
                    turbine.power,
                    turbine.aI,
                    turbine.average_velocity,
                )
            )
        baseline = curl_test_class.baseline(i)
        assert pytest.approx(turbine.Cp) == baseline[0]
        assert pytest.approx(turbine.Ct) == baseline[1]
        assert pytest.approx(turbine.power) == baseline[2]
        assert pytest.approx(turbine.aI) == baseline[3]
        assert pytest.approx(turbine.average_velocity) == baseline[4]

    # Change back to Gauss, rerun calculate_wake, and compare to gauss
    floris.farm.set_wake_model("gauss")
    floris.farm.flow_field.calculate_wake()
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        if test_class.debug:
            print(
                "({:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.7f})".format(
                    turbine.Cp,
                    turbine.Ct,
                    turbine.power,
                    turbine.aI,
                    turbine.average_velocity,
                )
            )
        baseline = gauss_test_class.baseline(i)
        assert pytest.approx(turbine.Cp) == baseline[0]
        assert pytest.approx(turbine.Ct) == baseline[1]
        assert pytest.approx(turbine.power) == baseline[2]
        assert pytest.approx(turbine.aI) == baseline[3]
        assert pytest.approx(turbine.average_velocity) == baseline[4]
