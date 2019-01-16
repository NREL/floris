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
            (0.46328782548262326, 0.7661304442831962, 1712005.16797175561077893, 0.25819969204072352, 7.85065163365445962),
            (0.46184306919581714, 0.7972304459337220, 1127336.88354835403151810, 0.27485029754279156, 6.83713371659016733)
        ]
        return baseline[turbine_index]

    def floris_jimenez_baseline(self, turbine_index):
        baseline = [
            (0.46328782548262326, 0.7661304442831962, 1712005.16797175561077893, 0.25819969204072352, 7.85065163365445962),
            (0.46181073960039850, 0.7979032595900662, 1117748.16816022805869579, 0.27522414475196977, 6.81785286549156577)
        ]
        return baseline[turbine_index]

    def gauss_baseline(self, turbine_index):
        baseline = [
            (0.46328782548262326, 0.7661304442831962, 1712005.16797175561077893, 0.25819969204072352, 7.85065163365445962),
            (0.46216883282818080, 0.7904509543330357, 1227059.75092757423408329, 0.27111736322573293, 7.03141391117972780)
        ]
        return baseline[turbine_index]

    def curl_baseline(self, turbine_index):
        baseline = [
            (0.46328739380548106, 0.7661167268371584, 1714019.48714753659442067, 0.25819260083547813, 7.85373185183701050),
            (0.46320695168382275, 0.7688465842832697, 1584115.79411471495404840, 0.25960791625100760, 7.65053131178665691)
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
        local = (
            turbine.Cp,
            turbine.Ct,
            turbine.power,
            turbine.aI,
            turbine.get_average_velocity()
        )
        assert local == test_class.jensen_jimenez_baseline(i)


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
        local = (
            turbine.Cp,
            turbine.Ct,
            turbine.power,
            turbine.aI,
            turbine.get_average_velocity()
        )
        assert local == test_class.floris_jimenez_baseline(i)


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
        local = (
            turbine.Cp,
            turbine.Ct,
            turbine.power,
            turbine.aI,
            turbine.get_average_velocity()
        )
        assert local == test_class.gauss_baseline(i)


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
        local = (
            turbine.Cp,
            turbine.Ct,
            turbine.power,
            turbine.aI,
            turbine.get_average_velocity()
        )
        assert local == test_class.curl_baseline(i)
