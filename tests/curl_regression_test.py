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
import numpy as np
from floris import Floris
from .sample_inputs import SampleInputs


class CurlRegressionTest():
    """
    """

    def __init__(self):
        sample_inputs = SampleInputs()
        sample_inputs.floris["wake"]["properties"]["velocity_model"] = "curl"
        sample_inputs.floris["wake"]["properties"]["deflection_model"] = "curl"
        self.input_dict = sample_inputs.floris

    def baseline(self, turbine_index):
        baseline = [
            (0.4632707, 0.7655868, 1793046.5944261, 0.2579188, 7.9727208),
            (0.4577105, 0.8181123,  892703.0087349, 0.2867585, 6.3444362)
        ]
        return baseline[turbine_index]

    def yawed_baseline(self, turbine_index):
        baseline = [
            (0.4632707, 0.7655868, 1780240.6778116, 0.2573987, 7.9727208),
            (0.4633075, 0.9900000, 1793661.6494183, 0.2579167, 7.9736330)
        ]
        return baseline[turbine_index]


def test_regression_tandem():
    """
    Tandem turbines
    """
    test_class = CurlRegressionTest()
    floris = Floris(input_dict=test_class.input_dict)
    floris.calculate_wake()
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        # print("({:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.7f})".format(turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity))
        baseline = test_class.baseline(i)
        assert pytest.approx(turbine.Cp) == baseline[0]
        assert pytest.approx(turbine.Ct) == baseline[1]
        assert pytest.approx(turbine.power) == baseline[2]
        assert pytest.approx(turbine.aI) == baseline[3]
        assert pytest.approx(turbine.average_velocity) == baseline[4]


def test_regression_triangle_farm():
    """
    Triangle farm where wind direction of 270 and 360 should result in the same power
    """
    test_class = CurlRegressionTest()
    distance = 5 * \
        test_class.input_dict["turbine"]["properties"]["rotor_diameter"]
    test_class.input_dict["farm"]["properties"]["layout_x"] = [
        0.0, distance, 0.0]
    test_class.input_dict["farm"]["properties"]["layout_y"] = [
        distance, distance, 0.0]
    floris = Floris(input_dict=test_class.input_dict)

    ### unrotated
    floris.farm.flow_field.calculate_wake()

    # turbine 1 - unwaked
    turbine = floris.farm.turbine_map.turbines[0]
    local = (turbine.Cp, turbine.Ct, turbine.power,
             turbine.aI, turbine.average_velocity)
    assert pytest.approx(local) == test_class.baseline(0)

    # turbine 2 - waked
    turbine = floris.farm.turbine_map.turbines[1]
    local = (turbine.Cp, turbine.Ct, turbine.power,
             turbine.aI, turbine.average_velocity)
    assert pytest.approx(local) == test_class.baseline(1)

    # turbine 3 - unwaked
    turbine = floris.farm.turbine_map.turbines[2]
    local = (turbine.Cp, turbine.Ct, turbine.power,
             turbine.aI, turbine.average_velocity)
    assert pytest.approx(local) == test_class.baseline(0)

    ### rotated
    floris.farm.flow_field.reinitialize_flow_field(wind_direction=360)
    floris.calculate_wake()

    # turbine 1 - unwaked
    turbine = floris.farm.turbine_map.turbines[0]
    local = (turbine.Cp, turbine.Ct, turbine.power,
             turbine.aI, turbine.average_velocity)
    assert pytest.approx(local) == test_class.baseline(0)

    # turbine 2 - unwaked
    turbine = floris.farm.turbine_map.turbines[1]
    local = (turbine.Cp, turbine.Ct, turbine.power,
             turbine.aI, turbine.average_velocity)
    assert pytest.approx(local) == test_class.baseline(0)

    # turbine 3 - waked
    turbine = floris.farm.turbine_map.turbines[2]
    local = (turbine.Cp, turbine.Ct, turbine.power,
             turbine.aI, turbine.average_velocity)
    assert pytest.approx(local) == test_class.baseline(1)


def test_regression_yaw():
    """
    Tandem turbines with the upstream turbine yawed
    """
    test_class = CurlRegressionTest()
    floris = Floris(input_dict=test_class.input_dict)

    # yaw the upstream turbine 5 degrees
    rotation_angle = 5.0
    floris.farm.set_yaw_angles([np.radians(rotation_angle), 0.0])
    floris.calculate_wake()
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        # print("({:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.7f})".format(turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity))
        baseline = test_class.yawed_baseline(i)
        assert pytest.approx(turbine.Cp) == baseline[0]
        assert pytest.approx(turbine.Ct) == baseline[1]
        assert pytest.approx(turbine.power) == baseline[2]
        assert pytest.approx(turbine.aI) == baseline[3]
        assert pytest.approx(turbine.average_velocity) == baseline[4]
