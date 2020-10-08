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


class GaussRegressionTest:
    """
    """

    def __init__(self):
        sample_inputs = SampleInputs()
        sample_inputs.floris["wake"]["properties"]["velocity_model"] = "gauss"
        sample_inputs.floris["wake"]["properties"]["deflection_model"] = "gauss"
        self.input_dict = sample_inputs.floris
        self.debug = False

    def baseline(self, turbine_index):
        baseline = [
            (0.4632706, 0.7655828, 1793661.6494183, 0.2579167, 7.9736330),
            (0.4529770, 0.8363843, 729117.4390409, 0.2977528, 5.9510659),
            (0.4574196, 0.8192350, 881978.4823849, 0.2874177, 6.3202663),
        ]
        return baseline[turbine_index]

    def yawed_baseline(self, turbine_index):
        baseline = [
            (0.4632733, 0.7626695, 1780861.5909742, 0.2559061, 7.9736330),
            (0.4534739, 0.8344660, 745219.1570045, 0.2965706, 5.9923643),
            (0.4574952, 0.8189433, 884755.7595666, 0.2872462, 6.3265451),
        ]
        return baseline[turbine_index]

    def gch_baseline(self, turbine_index):
        baseline = [
            (0.4632733, 0.7626695, 1780861.5909742, 0.2559061, 7.9736330),
            (0.4535845, 0.8340392, 748834.7449840, 0.2963086, 6.0015520),
            (0.4579510, 0.8171837, 901641.6247690, 0.2862149, 6.3644267),
        ]
        return baseline[turbine_index]

    def yaw_added_recovery_baseline(self, turbine_index):
        baseline = [
            (0.4632733, 0.7626695, 1780861.5909742, 0.2559061, 7.9736330),
            (0.4535845, 0.8340392, 748834.7449840, 0.2963086, 6.0015520),
            (0.4572521, 0.8198817, 875841.5697827, 0.2877983, 6.3063432),
        ]
        return baseline[turbine_index]

    def secondary_steering_baseline(self, turbine_index):
        # TODO: why are these the same as yawed_baselines?
        #   does SS not work affect the current configuration?
        baseline = [
            (0.4632733, 0.7626695, 1780861.5909742, 0.2559061, 7.9736330),
            (0.4534739, 0.8344660, 745219.1570045, 0.2965706, 5.9923643),
            (0.4574952, 0.8189433, 884755.7595666, 0.2872462, 6.3265451),
        ]
        return baseline[turbine_index]


def test_regression_tandem():
    """
    Tandem turbines
    """
    test_class = GaussRegressionTest()
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
    test_class = GaussRegressionTest()
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
    test_class = GaussRegressionTest()
    floris = Floris(input_dict=test_class.input_dict)

    # yaw the upstream turbine 5 degrees
    rotation_angle = 5.0
    floris.farm.set_yaw_angles([rotation_angle, 0.0, 0.0])
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


def test_regression_gch():
    """
    Tandem turbines with the upstream turbine yawed and yaw added recovery
    correction enabled
    """
    test_class = GaussRegressionTest()
    floris = Floris(input_dict=test_class.input_dict)
    floris.farm.set_yaw_angles([5.0, 0.0, 0.0])

    # With secondary steering off, GCH should be same as Gauss
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

    # With secondary steering on, the results should change
    floris.farm.wake.velocity_model.use_yaw_added_recovery = True
    floris.farm.wake.deflection_model.use_secondary_steering = True
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
        baseline = test_class.gch_baseline(i)
        assert pytest.approx(turbine.Cp) == baseline[0]
        assert pytest.approx(turbine.Ct) == baseline[1]
        assert pytest.approx(turbine.power) == baseline[2]
        assert pytest.approx(turbine.aI) == baseline[3]
        assert pytest.approx(turbine.average_velocity) == baseline[4]


def test_regression_yaw_added_recovery():
    """
    Tandem turbines with the upstream turbine yawed and yaw added recovery
    correction enabled
    """
    test_class = GaussRegressionTest()
    floris = Floris(input_dict=test_class.input_dict)
    floris.farm.set_yaw_angles([5.0, 0.0, 0.0])

    # With yaw added recovery off, GCH should be same as Gauss
    floris.farm.wake.velocity_model.use_yaw_added_recovery = False
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

    # With yaw added recovery on, the results should change
    floris.farm.wake.velocity_model.use_yaw_added_recovery = True
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
        baseline = test_class.yaw_added_recovery_baseline(i)
        assert pytest.approx(turbine.Cp) == baseline[0]
        assert pytest.approx(turbine.Ct) == baseline[1]
        assert pytest.approx(turbine.power) == baseline[2]
        assert pytest.approx(turbine.aI) == baseline[3]
        assert pytest.approx(turbine.average_velocity) == baseline[4]


def test_regression_secondary_steering():
    """
    Tandem turbines with the upstream turbine yawed and yaw added recovery
    correction enabled
    """
    test_class = GaussRegressionTest()
    floris = Floris(input_dict=test_class.input_dict)
    floris.farm.set_yaw_angles([5.0, 0.0, 0.0])

    # With secondary steering off, GCH should be same as Gauss
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

    # With secondary steering on, the results should change
    floris.farm.wake.deflection_model.use_secondary_steering = True
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
        baseline = test_class.secondary_steering_baseline(i)
        assert pytest.approx(turbine.Cp) == baseline[0]
        assert pytest.approx(turbine.Ct) == baseline[1]
        assert pytest.approx(turbine.power) == baseline[2]
        assert pytest.approx(turbine.aI) == baseline[3]
        assert pytest.approx(turbine.average_velocity) == baseline[4]
