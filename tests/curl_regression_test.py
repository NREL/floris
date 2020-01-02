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
import copy
from floris.simulation import Floris
from floris.simulation import TurbineMap
from .sample_inputs import SampleInputs


class CurlRegressionTest():
    """
    """

    def __init__(self):
        sample_inputs = SampleInputs()
        sample_inputs.floris["wake"]["properties"]["velocity_model"] = "curl"
        sample_inputs.floris["wake"]["properties"]["deflection_model"] = "curl"
        self.input_dict = sample_inputs.floris
        self.debug = False

    def baseline(self, turbine_index):
        baseline = [
            (0.4632707, 0.7655868, 1793046.5944261, 0.2579188, 7.9727208),
            (0.4531780, 0.8356084, 735600.0517466, 0.2972739, 5.9677685)
        ]
        return baseline[turbine_index]

    def yawed_baseline(self, turbine_index):
        baseline = [
            (0.4632734, 0.7626735, 1780250.9240069, 0.2559082, 7.9727208),
            (0.4539774, 0.8325224, 761784.3584749, 0.2953798, 6.0342075)
        ]
        return baseline[turbine_index]


def test_regression_tandem():
    """
    Tandem turbines
    """
    test_class = CurlRegressionTest()
    floris = Floris(input_dict=test_class.input_dict)
    floris.farm.flow_field.calculate_wake()
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        if test_class.debug:
            print("({:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.7f})".format(turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity))
        baseline = test_class.baseline(i)
        assert pytest.approx(turbine.Cp) == baseline[0]
        assert pytest.approx(turbine.Ct) == baseline[1]
        assert pytest.approx(turbine.power) == baseline[2]
        assert pytest.approx(turbine.aI) == baseline[3]
        assert pytest.approx(turbine.average_velocity) == baseline[4]


def test_regression_rotation():
    """
    Turbines in tandem and rotated.
    The result from 270 degrees should match the results from 360 degrees.
    """
    test_class = CurlRegressionTest()
    floris = Floris(input_dict=test_class.input_dict)
    fresh_turbine = copy.deepcopy(floris.farm.turbine_map.turbines[0])

    ### unrotated
    floris.farm.flow_field.calculate_wake()
    turbine = floris.farm.turbine_map.turbines[0]
    unwaked_baseline = (turbine.Cp, turbine.Ct, turbine.power,
                        turbine.aI, turbine.average_velocity)
    turbine = floris.farm.turbine_map.turbines[1]
    waked_baseline = (turbine.Cp, turbine.Ct, turbine.power,
                      turbine.aI, turbine.average_velocity)
    wind_map = floris.farm.wind_map

    ### rotated
    wind_map.input_direction = [360]
    wind_map.calculate_wind_direction()
    new_map = TurbineMap(
        [0.0, 0.0],
        [5 * test_class.input_dict["turbine"]["properties"]["rotor_diameter"], 0.0],
        [copy.deepcopy(fresh_turbine), copy.deepcopy(fresh_turbine)]
    )
    floris.farm.flow_field.reinitialize_flow_field(
        wind_map = wind_map,
        turbine_map=new_map
    )
    floris.farm.flow_field.calculate_wake()

    turbine = floris.farm.turbine_map.turbines[0]
    assert pytest.approx(turbine.Cp) == unwaked_baseline[0]
    assert pytest.approx(turbine.Ct) == unwaked_baseline[1]
    assert pytest.approx(turbine.power) == unwaked_baseline[2]
    assert pytest.approx(turbine.aI) == unwaked_baseline[3]
    assert pytest.approx(turbine.average_velocity) == unwaked_baseline[4]

    turbine = floris.farm.turbine_map.turbines[1]
    assert pytest.approx(turbine.Cp) == waked_baseline[0]
    assert pytest.approx(turbine.Ct) == waked_baseline[1]
    assert pytest.approx(turbine.power) == waked_baseline[2]
    assert pytest.approx(turbine.aI) == waked_baseline[3]
    assert pytest.approx(turbine.average_velocity) == waked_baseline[4]


def test_regression_yaw():
    """
    Tandem turbines with the upstream turbine yawed
    """
    test_class = CurlRegressionTest()
    floris = Floris(input_dict=test_class.input_dict)

    # yaw the upstream turbine 5 degrees
    rotation_angle = 5.0
    floris.farm.set_yaw_angles([rotation_angle, 0.0])
    floris.farm.flow_field.calculate_wake()
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        if test_class.debug:
            print("({:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.7f})".format(turbine.Cp, turbine.Ct, turbine.power, turbine.aI, turbine.average_velocity))
        baseline = test_class.yawed_baseline(i)
        assert pytest.approx(turbine.Cp) == baseline[0]
        assert pytest.approx(turbine.Ct) == baseline[1]
        assert pytest.approx(turbine.power) == baseline[2]
        assert pytest.approx(turbine.aI) == baseline[3]
        assert pytest.approx(turbine.average_velocity) == baseline[4]
