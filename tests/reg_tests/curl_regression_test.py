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

import copy
from pytest import approx

from tests.conftest import print_test_values, turbines_to_array
from floris.simulation import Floris, TurbineMap

DEBUG = False
VELOCITY_MODEL = "curl"
DEFLECTION_MODEL = "curl"

baseline = [
    (0.4632711, 0.7655987, 1808879.0573334, 0.2579249, 7.9700630),
    (0.4542347, 0.8315292, 790095.3264943, 0.2947740, 6.0555893),
    (0.4463487, 0.8568269, 585359.6178647, 0.3108089, 5.5408759),
]

yawed_baseline = [
    (0.4632738, 0.7626853, 1795186.5605035, 0.2559142, 7.9700630),
    (0.4547445, 0.8295616, 807715.7153082, 0.2935790, 6.0979495),
    (0.4479135, 0.8538732, 598723.1123378, 0.3088673, 5.5865154),
]

class CurlRegressionTest:
    """
    Class to contain test values to compare against.
    """

    def __init__(self):
        sample_inputs = SampleInputs()
        sample_inputs.floris["wake"]["properties"]["velocity_model"] = "curl"
        sample_inputs.floris["wake"]["properties"]["deflection_model"] = "curl"
        self.input_dict = sample_inputs.floris
        self.debug = True

    def baseline(self, turbine_index):
        """
        Three turbines spaced 5D apart in the streamwise direction,
        all aligned with the wind direction 270 degrees.
        """
        baseline = [
            (0.4365911, 0.7636967, 1689187.1164557, 0.2569448, 7.9700630),
            (0.4294209, 0.8326933, 731401.6677331, 0.2954843, 6.0592226),
            (0.4209117, 0.8589734, 547760.8043000, 0.3122325, 5.5397800),
        ]
        return baseline[turbine_index]

    def yawed_baseline(self, turbine_index):
        """
        Three turbines spaced 5D apart in the streamwise direction,
        the first turbine yawed 5 degrees while the others are aligned to the wind.
        """
        baseline = [
            (0.4366014, 0.7607906, 1677789.6107559, 0.2549496, 7.9700630),
            (0.4298470, 0.8307488, 748494.6942714, 0.2942993, 6.1014089),
            (0.4216931, 0.8566472, 563529.9139905, 0.3106902, 5.5852387),
        ]
        return baseline[turbine_index]

    def wd315_baseline(self, turbine_index):
        """
        Three turbines spaced 5D apart in the streamwise direction,
        all aligned with the wind direction of 315 degrees.
        """
        baseline = [
            (0.4365911, 0.7636967, 1689187.1164557, 0.2569448, 7.9700630),
            (0.4365911, 0.7636967, 1689187.1164557, 0.2569448, 7.9700630),
            (0.4365911, 0.7636967, 1689187.1164557, 0.2569448, 7.9700630),
        ]
        return baseline[turbine_index]


def test_regression_tandem():
    """
    Tandem turbines
    """
    sample_inputs_fixture.floris["wake"]["properties"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"]["deflection_model"] = DEFLECTION_MODEL
    floris = Floris(input_dict=sample_inputs_fixture.floris)
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


def test_regression_multiple_calc_wake():
    """
    Verify turbine values stay the same with repeated (3x) calculate_wake calls.
    """
    test_class = CurlRegressionTest()
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
    sample_inputs_fixture.floris["wake"]["properties"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"]["deflection_model"] = DEFLECTION_MODEL
    floris = Floris(input_dict=sample_inputs_fixture.floris)
    fresh_turbine = copy.deepcopy(floris.farm.turbine_map.turbines[0])
    wind_map = floris.farm.wind_map

    # unrotated
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

    # rotated
    wind_map.input_direction = [360]
    wind_map.calculate_wind_direction()
    new_map = TurbineMap(
        [0.0, 0.0, 0.0],
        [
            10 * sample_inputs_fixture.floris["turbine"]["properties"]["rotor_diameter"],
            5 * sample_inputs_fixture.floris["turbine"]["properties"]["rotor_diameter"],
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
    assert approx(turbine.Cp) == unwaked_baseline[0]
    assert approx(turbine.Ct) == unwaked_baseline[1]
    assert approx(turbine.power) == unwaked_baseline[2]
    assert approx(turbine.aI) == unwaked_baseline[3]
    assert approx(turbine.average_velocity) == unwaked_baseline[4]

    turbine = floris.farm.turbine_map.turbines[1]
    assert approx(turbine.Cp) == first_waked_baseline[0]
    assert approx(turbine.Ct) == first_waked_baseline[1]
    assert approx(turbine.power) == first_waked_baseline[2]
    assert approx(turbine.aI) == first_waked_baseline[3]
    assert approx(turbine.average_velocity) == first_waked_baseline[4]

    turbine = floris.farm.turbine_map.turbines[2]
    assert approx(turbine.Cp) == second_waked_baseline[0]
    assert approx(turbine.Ct) == second_waked_baseline[1]
    assert approx(turbine.power) == second_waked_baseline[2]
    assert approx(turbine.aI) == second_waked_baseline[3]
    assert approx(turbine.average_velocity) == second_waked_baseline[4]

def test_regression_yaw(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed 5 degrees.
    """
    sample_inputs_fixture.floris["wake"]["properties"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"]["deflection_model"] = DEFLECTION_MODEL
    floris = Floris(input_dict=sample_inputs_fixture.floris)

    # yaw the upstream turbine 5 degrees
    rotation_angle = 5.0
    floris.farm.set_yaw_angles([rotation_angle, 0.0, 0.0])
    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    for i in range(len(floris.farm.turbine_map.turbines)):
        baseline = yawed_baseline
        assert test_results[i][0] == approx(baseline[i][0])
        assert test_results[i][1] == approx(baseline[i][1])
        assert test_results[i][2] == approx(baseline[i][2])
        assert test_results[i][3] == approx(baseline[i][3])
        assert test_results[i][4] == approx(baseline[i][4])


def test_change_wind_direction():
    """
    Tandem turbines aligned to the wind direction, first calculated at 270 degrees wind
    direction, and then calculated at 315 degrees wind direction.
    """
    test_class = CurlRegressionTest()
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

    floris.farm.wind_map.input_direction = [315.0]
    floris.farm.wind_map.calculate_wind_direction()
    floris.farm.turbine_map.reinitialize_turbines()
    floris.farm.flow_field.reinitialize_flow_field()

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
        baseline = test_class.wd315_baseline(i)
        assert pytest.approx(turbine.Cp) == baseline[0]
        assert pytest.approx(turbine.Ct) == baseline[1]
        assert pytest.approx(turbine.power) == baseline[2]
        assert pytest.approx(turbine.aI) == baseline[3]
        assert pytest.approx(turbine.average_velocity) == baseline[4]
