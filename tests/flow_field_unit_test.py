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

import numpy as np
from floris.simulation import Floris, FlowField, Turbine, TurbineMap, Wake, WakeCombination
from floris.utilities import Vec3
from .sample_inputs import SampleInputs
import copy


class FlowFieldTest():
    def __init__(self):
        self.sample_inputs = SampleInputs()
        self.input_dict = self._build_input_dict()
        self.instance = self._build_instance()

    def _build_input_dict(self):
        wake = Wake(self.sample_inputs.wake)
        turbine = Turbine(self.sample_inputs.turbine)
        turbine_map = TurbineMap(
            [0.0, 100.0],
            [0.0, 0.0],
            [copy.deepcopy(turbine), copy.deepcopy(turbine)]
        )
        return {
            "wind_direction": 270.0,
            "wind_speed": 8.0,
            "wind_shear": 0.0,
            "wind_veer": 0.0,
            "turbulence_intensity": 1.0,
            "air_density": 1.225,
            "wake": wake,
            "turbine_map": turbine_map
        }

    def _build_instance(self):
        return FlowField(self.input_dict["wind_speed"],
                         self.input_dict["wind_direction"],
                         self.input_dict["wind_shear"],
                         self.input_dict["wind_veer"],
                         self.input_dict["turbulence_intensity"],
                         self.input_dict["air_density"],
                         self.input_dict["wake"],
                         self.input_dict["turbine_map"])


def test_instantiation():
    """
    The class should initialize with the standard inputs
    """
    test_class = FlowFieldTest()
    assert test_class.instance is not None


def test_discretize_domain():
    """
    The class should discretize the domain on initialization with three
    component-arrays each of type np.ndarray and size (100, 100, 50)
    """
    test_class = FlowFieldTest()
    x, y, z = test_class.instance._discretize_turbine_domain()
    assert np.shape(x) == (2, 5, 5) and type(x) is np.ndarray \
        and np.shape(y) == (2, 5, 5) and type(y) is np.ndarray \
        and np.shape(z) == (2, 5, 5) and type(z) is np.ndarray
