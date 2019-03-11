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
            (0.4632706, 0.7655828, 1793661.6494183, 0.2579167, 7.9736330),
            (0.4553179, 0.8273480,  807133.4600243, 0.2922430, 6.1456034)
        ]
        return baseline[turbine_index]

    def floris_jimenez_baseline(self, turbine_index):
        baseline = [
            (0.4632706, 0.7655828, 1793661.6494183, 0.2579167, 7.9736330),
            (0.4393791, 0.8699828,  510284.5989879, 0.3197105, 5.3375917)
        ]
        return baseline[turbine_index]

    def gauss_baseline(self, turbine_index):
        baseline = [
            (0.4632706, 0.7655828, 1793661.6494183, 0.2579167, 7.9736330),
            (0.4513828, 0.8425381,  679100.2574472, 0.3015926, 5.8185835)
        ]
        return baseline[turbine_index]

    def curl_baseline(self, turbine_index):
        baseline = [
            (0.4632707, 0.7655868, 1793046.5944261, 0.2579188, 7.9727208),
            (0.4577105, 0.8181123,  892703.0087349, 0.2867585, 6.3444362)
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
        # print("({:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.7f})".format(turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity))
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
        # print("({:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.7f})".format(turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity))
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
        # print("({:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.7f})".format(turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity))
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
        # print("({:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.7f})".format(turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity))
        local = (turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity)
        assert pytest.approx(local) == test_class.curl_baseline(i)

def test_regression_triangle_farm():
    """
    Velocity defecit model: gauss
    Wake deflection model: gauss
    Layout: triangle farm where wind direction of 270 and 315 should result in the same power
    """
    test_class = RegressionTest()
    test_class.input_dict["wake"]["properties"]["velocity_model"] = "gauss"
    test_class.input_dict["wake"]["properties"]["deflection_model"] = "gauss_deflection"
    distance = 5 * test_class.input_dict["turbine"]["properties"]["rotor_diameter"]
    test_class.input_dict["farm"]["properties"]["layout_x"] = [0.0, distance, 0.0]
    test_class.input_dict["farm"]["properties"]["layout_y"] = [distance, distance, 0.0]
    floris = Floris(input_dict=test_class.input_dict)
    
    ### unrotated
    floris.farm.flow_field.calculate_wake()
    
    # turbine 1 - unwaked
    turbine = floris.farm.turbine_map.turbines[0]
    local = (turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity)
    assert pytest.approx(local) == test_class.gauss_baseline(0)

    # turbine 2 - waked
    turbine = floris.farm.turbine_map.turbines[1]
    local = (turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity)
    assert pytest.approx(local) == test_class.gauss_baseline(1)

    # turbine 3 - unwaked
    turbine = floris.farm.turbine_map.turbines[2]
    local = (turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity)
    assert pytest.approx(local) == test_class.gauss_baseline(0)
    
    ### rotated
    floris.farm.flow_field.reinitialize_flow_field(wind_direction=360)
    floris.calculate_wake()

    # turbine 1 - unwaked
    turbine = floris.farm.turbine_map.turbines[0]
    local = (turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity)
    assert pytest.approx(local) == test_class.gauss_baseline(0)

    # turbine 2 - unwaked
    turbine = floris.farm.turbine_map.turbines[1]
    local = (turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity)
    assert pytest.approx(local) == test_class.gauss_baseline(0)

    # turbine 3 - waked
    turbine = floris.farm.turbine_map.turbines[2]
    local = (turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity)
    assert pytest.approx(local) == test_class.gauss_baseline(1)
