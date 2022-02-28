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

import attrs
from attrs import define, field

from floris.simulation import BaseClass, BaseModel
from floris.simulation.wake_deflection import (
    GaussVelocityDeflection,
    JimenezVelocityDeflection,
)
from floris.simulation.wake_combination import FLS, MAX, SOSFS
from floris.simulation.wake_turbulence.crespo_hernandez import CrespoHernandez
from floris.simulation.wake_velocity import (
    CumulativeGaussCurlVelocityDeficit,
    GaussVelocityDeficit,
    JensenVelocityDeficit
)


MODEL_MAP = {
    "combination_model": {
        "fls": FLS,
        "max": MAX,
        "sosfs": SOSFS
    },
    "deflection_model": {
        "jimenez": JimenezVelocityDeflection,
        "gauss": GaussVelocityDeflection
    },
    "turbulence_model": {
        "crespo_hernandez": CrespoHernandez
    },
    "velocity_model": {
        "cc": CumulativeGaussCurlVelocityDeficit,
        "gauss": GaussVelocityDeficit,
        "jensen": JensenVelocityDeficit
    },
}


@define
class WakeModelManager(BaseClass):
    """
    WakeModelManager is a container class for the wake velocity, deflection,
    turbulence, and combination models.

    Args:
        wake (:obj:`dict`): The wake's properties input dictionary
            - velocity_model (str): The name of the velocity model to be instantiated.
            - turbulence_model (str): The name of the turbulence model to be instantiated.
            - deflection_model (str): The name of the deflection model to be instantiated.
            - combination_model (str): The name of the combination model to be instantiated.
    """
    model_strings: dict = field(converter=dict)
    enable_secondary_steering: bool = field(converter=bool)
    enable_yaw_added_recovery: bool = field(converter=bool)
    enable_transverse_velocities: bool = field(converter=bool)

    wake_deflection_parameters: dict = field(converter=dict)
    wake_turbulence_parameters: dict = field(converter=dict)
    wake_velocity_parameters: dict = field(converter=dict, default={})

    combination_model: BaseModel = field(init=False)
    deflection_model: BaseModel = field(init=False)
    turbulence_model: BaseModel = field(init=False)
    velocity_model: BaseModel = field(init=False)

    def __attrs_post_init__(self) -> None:
        model: BaseModel = MODEL_MAP["velocity_model"][self.model_strings["velocity_model"]]
        model_parameters = self.wake_velocity_parameters[self.model_strings["velocity_model"]]
        if model_parameters is None:
            # Use model defaults
            self.velocity_model = model()
        else:
            self.velocity_model = model.from_dict(model_parameters)

        model: BaseModel = MODEL_MAP["deflection_model"][self.model_strings["deflection_model"]]
        model_parameters = self.wake_deflection_parameters[self.model_strings["deflection_model"]]
        if model_parameters is None:
            self.deflection_model = model()
        else:
            self.deflection_model = model.from_dict(model_parameters)

        model: BaseModel = MODEL_MAP["turbulence_model"][self.model_strings["turbulence_model"]]
        model_parameters = self.wake_turbulence_parameters[self.model_strings["turbulence_model"]]
        if model_parameters is None:
            self.turbulence_model = model()
        else:
            self.turbulence_model = model.from_dict(model_parameters)

        model: BaseModel = MODEL_MAP["combination_model"][self.model_strings["combination_model"]]
        self.combination_model = model()

    @model_strings.validator
    def validate_model_strings(self, instance: attrs.Attribute, value: dict) -> None:
        required_strings = [
            "velocity_model",
            "deflection_model",
            "combination_model",
            "turbulence_model"
        ]
        # Check that all required strings are given
        for s in required_strings:
            if s not in value.keys():
                raise KeyError(f"Wake: '{s}' not provided in the input but it is required.")

        # Check that no other strings are given
        for k in value.keys():
            if k not in required_strings:
                raise KeyError(f"Wake: '{k}' was given as input but it is not a valid option. Required inputs are: {', '.join(required_strings)}")

    @property
    def deflection_function(self):
        return self.deflection_model.function

    @property
    def velocity_function(self):
        return self.velocity_model.function

    @property
    def turbulence_function(self):
        return self.turbulence_model.function

    @property
    def combination_function(self):
        return self.combination_model.function
