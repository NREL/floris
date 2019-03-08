"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import pytest
from floris import Floris
from .sample_inputs import SampleInputs


class RegressionTest():
    """
    These regression tests use a two turbine wind farm. One turbine is waked while
    the other is not.

    The varying parameters are:
    - Velocity deficity model
    - Wake deflection model

    Baseline solutions are [Cp, Ct, power, aI, average velocity].
    """
    def __init__(self):
        self.sample_inputs = SampleInputs()
        self.input_dict = self.build_input_dict()

    def build_input_dict(self):
        return self.sample_inputs.floris

    def jensen_jimenez_baseline(self, turbine_index):
        baseline = [
            (0.4632705, 0.7655827, 1793661.6494182, 0.2579167, 7.9736329),
            (0.4632705, 0.7655827, 1793661.6494182, 0.2579167, 7.9736329)
        ]
        return baseline[turbine_index]

    def floris_jimenez_baseline(self, turbine_index):
        baseline = [
            (0.4632705, 0.7655827, 1793661.6494182, 0.2579167, 7.9736329),
            (0.4632705, 0.7655827, 1793661.6494182, 0.2579167, 7.9736329)
        ]
        return baseline[turbine_index]

    def gauss_baseline(self, turbine_index):
        baseline = [
            (0.4632705, 0.7655827, 1793661.6494182, 0.2579167, 7.9736329),
            (0.4629500, 0.7741939, 1489994.2682055, 0.2624047, 7.4972929)
        ]
        return baseline[turbine_index]

    def curl_baseline(self, turbine_index):
        baseline = [
            (0.4632707, 0.7655868, 1793046.5944260, 0.2579188, 7.9727207),
            (0.4632722, 0.7656358, 1785632.3906150, 0.2579441, 7.9617077)
        ]
        return baseline[turbine_index]


def test_regression_jensen_jimenez():
    """
    Velocity defecit model: jensen
    Wake deflection model: jimenez
    """
    test_class = RegressionTest()
    test_class.input_dict["wake"]["properties"]["velocity_model"] = "jensen"
    test_class.input_dict["wake"]["properties"]["deflection_model"] = "jimenez"
    floris = Floris(input_dict=test_class.input_dict)
    floris.farm.flow_field.calculate_wake()
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        # print("({:.17f}, {:.17f}, {:.17f}, {:.17f}, {:.17f})".format(turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity))
        local = (turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity)
        assert pytest.approx(local) == test_class.jensen_jimenez_baseline(i)


def test_regression_floris_jimenez():
    """
    Velocity defecit model: floris
    Wake deflection model: jimenez
    """
    test_class = RegressionTest()
    test_class.input_dict["wake"]["properties"]["velocity_model"] = "floris"
    test_class.input_dict["wake"]["properties"]["deflection_model"] = "jimenez"
    floris = Floris(input_dict=test_class.input_dict)
    floris.farm.flow_field.calculate_wake()
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        # print("({:.17f}, {:.17f}, {:.17f}, {:.17f}, {:.17f})".format(turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity))
        local = (turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity)
        assert pytest.approx(local) == test_class.floris_jimenez_baseline(i)

def test_regression_gauss():
    """
    Velocity defecit model: gauss
    Wake deflection model: gauss
    """
    test_class = RegressionTest()
    test_class.input_dict["wake"]["properties"]["velocity_model"] = "gauss"
    test_class.input_dict["wake"]["properties"]["deflection_model"] = "gauss_deflection"
    floris = Floris(input_dict=test_class.input_dict)
    floris.farm.flow_field.calculate_wake()
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        # print("({:.17f}, {:.17f}, {:.17f}, {:.17f}, {:.17f})".format(turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity))
        local = (turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity)
        assert pytest.approx(local) == test_class.gauss_baseline(i)

def test_regression_curl():
    """
    Velocity defecit model: curl
    Wake deflection model: curl
    """
    test_class = RegressionTest()
    test_class.input_dict["wake"]["properties"]["velocity_model"] = "curl"
    test_class.input_dict["wake"]["properties"]["deflection_model"] = "curl"
    floris = Floris(input_dict=test_class.input_dict)
    floris.farm.flow_field.calculate_wake()
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        # print("({:.17f}, {:.17f}, {:.17f}, {:.17f}, {:.17f})".format(turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity))
        local = (turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity)
        assert pytest.approx(local) == test_class.curl_baseline(i)
