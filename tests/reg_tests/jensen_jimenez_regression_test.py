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

import copy

import numpy as np
import pytest

from floris.simulation import Floris, TurbineMap


try:
    from .sample_inputs import SampleInputs
except ImportError:
    from sample_inputs import SampleInputs


class JensenJimenezRegressionTest:
    """
    """

    def __init__(self):
        sample_inputs = SampleInputs()
        sample_inputs.floris["wake"]["properties"]["velocity_model"] = "jensen"
        sample_inputs.floris["wake"]["properties"]["deflection_model"] = "jimenez"
        self.input_dict = sample_inputs.floris
        self.debug = False

    def baseline(self, turbine_index):
        baseline = [
            (0.4632706, 0.7655828, 1793661.6494183, 0.2579167, 7.9736330),
            (0.4553179, 0.8273480, 807133.4600243, 0.2922430, 6.1456034),
            (0.4494880, 0.8498521, 622820.0654135, 0.3062554, 5.6611213),
        ]
        return baseline[turbine_index]

    def yawed_baseline(self, turbine_index):
        baseline = [
            (0.4632733, 0.7626695, 1780861.5909742, 0.2559061, 7.9736330),
            (0.4554894, 0.8266861, 813067.4207660, 0.2918451, 6.1598540),
            (0.4495612, 0.8495696, 624931.5642086, 0.3060732, 5.6672040),
        ]
        return baseline[turbine_index]


def test_regression_tandem():
    """
    Tandem turbines
    """
    test_class = JensenJimenezRegressionTest()
    floris = Floris(input_dict=test_class.input_dict)
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
    test_class = JensenJimenezRegressionTest()
    floris = Floris(input_dict=test_class.input_dict)
    fresh_turbine = copy.deepcopy(floris.farm.turbine_map.turbines[0])
    wind_map = floris.farm.wind_map

    ### unrotated
    floris.farm.flow_field.calculate_wake()
    turbine = floris.farm.turbine_map.turbines[0]
    unwaked_baseline = (
        turbine.Cp,
        turbine.Ct,
        turbine.power,
        turbine.aI,
        turbine.average_velocity,
    )
    turbine = floris.farm.turbine_map.turbines[1]
    first_waked_baseline = (
        turbine.Cp,
        turbine.Ct,
        turbine.power,
        turbine.aI,
        turbine.average_velocity,
    )
    turbine = floris.farm.turbine_map.turbines[2]
    second_waked_baseline = (
        turbine.Cp,
        turbine.Ct,
        turbine.power,
        turbine.aI,
        turbine.average_velocity,
    )

    ### rotated
    wind_map.input_direction = [360]
    wind_map.calculate_wind_direction()
    new_map = TurbineMap(
        [0.0, 0.0, 0.0],
        [
            10 * test_class.input_dict["turbine"]["properties"]["rotor_diameter"],
            5 * test_class.input_dict["turbine"]["properties"]["rotor_diameter"],
            0.0,
        ],
        [
            copy.deepcopy(fresh_turbine),
            copy.deepcopy(fresh_turbine),
            copy.deepcopy(fresh_turbine),
        ],
    )
    floris.farm.flow_field.reinitialize_flow_field(
        turbine_map=new_map, wind_map=wind_map
    )
    floris.farm.flow_field.calculate_wake()

    turbine = floris.farm.turbine_map.turbines[0]
    assert pytest.approx(turbine.Cp) == unwaked_baseline[0]
    assert pytest.approx(turbine.Ct) == unwaked_baseline[1]
    assert pytest.approx(turbine.power) == unwaked_baseline[2]
    assert pytest.approx(turbine.aI) == unwaked_baseline[3]
    assert pytest.approx(turbine.average_velocity) == unwaked_baseline[4]

    turbine = floris.farm.turbine_map.turbines[1]
    assert pytest.approx(turbine.Cp) == first_waked_baseline[0]
    assert pytest.approx(turbine.Ct) == first_waked_baseline[1]
    assert pytest.approx(turbine.power) == first_waked_baseline[2]
    assert pytest.approx(turbine.aI) == first_waked_baseline[3]
    assert pytest.approx(turbine.average_velocity) == first_waked_baseline[4]

    turbine = floris.farm.turbine_map.turbines[2]
    assert pytest.approx(turbine.Cp) == second_waked_baseline[0]
    assert pytest.approx(turbine.Ct) == second_waked_baseline[1]
    assert pytest.approx(turbine.power) == second_waked_baseline[2]
    assert pytest.approx(turbine.aI) == second_waked_baseline[3]
    assert pytest.approx(turbine.average_velocity) == second_waked_baseline[4]


def test_regression_yaw():
    """
    Tandem turbines with the upstream turbine yawed
    """
    test_class = JensenJimenezRegressionTest()
    floris = Floris(input_dict=test_class.input_dict)

    # yaw the upstream turbine 5 degrees
    rotation_angle = 5.0
    floris.farm.set_yaw_angles([rotation_angle, 0.0])
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
        baseline = test_class.yawed_baseline(i)
        assert pytest.approx(turbine.Cp) == baseline[0]
        assert pytest.approx(turbine.Ct) == baseline[1]
        assert pytest.approx(turbine.power) == baseline[2]
        assert pytest.approx(turbine.aI) == baseline[3]
        assert pytest.approx(turbine.average_velocity) == baseline[4]
