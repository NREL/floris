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


from typing import Any

import attr

from floris.utilities import model_attrib, attr_serializer, attr_floris_filter
from floris.simulation import BaseClass, BaseModel
from floris.simulation.wake_deflection import (
    GaussVelocityDeflection,
    JimenezVelocityDeflection,
)
from floris.simulation.wake_velocity import (
    # CurlVelocityDeficit,
    GaussVelocityDeficit,
    JensenVelocityDeficit
)


MODEL_MAP = {
    # TODO: Need to uncomment these two model types once we have implementations
    # "combination_model": {},
    "deflection_model": {
        "jimenez": JimenezVelocityDeflection,
        "gauss": GaussVelocityDeflection
    },
    # "turbulence_model": {},
    "velocity_model": {
        # "curl": CurlVelocityDeficit,
        "gauss": GaussVelocityDeficit,
        "jensen": JensenVelocityDeficit
    },
}


@attr.s(auto_attribs=True)
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
    model_strings: dict = attr.ib(factory=dict)
    wake_combination_parameters: dict = attr.ib(factory=dict)
    wake_deflection_parameters: dict = attr.ib(factory=dict)
    wake_turbulence_parameters: dict = attr.ib(factory=dict)
    wake_velocity_parameters: dict = attr.ib(factory=dict)

    enable_secondary_steering: bool = attr.ib(factory=bool)
    enable_yaw_added_recovery: bool = attr.ib(factory=bool)
    enable_transverse_velocities: bool = attr.ib(factory=bool)

    combination_model: BaseModel = attr.ib(init=False)
    deflection_model: BaseModel = attr.ib(init=False)
    turbulence_model: BaseModel = attr.ib(init=False)
    velocity_model: BaseModel = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.model_generator()

    @model_strings.validator
    def validate_model_strings(self, instance: attr.Attribute, value: dict) -> None:
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

    def model_generator(self) -> Any:
        wake_models = {}
        for model_type in ("deflection", "velocity"): #, "combination", "turbulence")
            model_string = self.model_strings[f"{model_type}_model"]
            model = MODEL_MAP[f"{model_type}_model"][model_string]

            model_def = getattr(self, f"wake_{model_type}_parameters")[model_string]
            wake_models[model_type] = model.from_dict(model_def)

        # TODO: Uncomment the two models once implemented
        # self.combination_model = wake_models["combination"]
        self.deflection_model = wake_models["deflection"]
        # self.turbulence_model = wake_models["turbulence"]
        self.velocity_model = wake_models["velocity"]

    def _asdict(self) -> dict:
        """Creates a JSON and YAML friendly dictionary that can be save for future reloading.
        This dictionary will contain only `Python` types that can later be converted to their
        proper `Wake` formats.

        Returns:
            dict: All key, vaue pais required for class recreation.
        """

        def create_dict(wake_model, model_string):
            if wake_model is None:
                return {}
            output = attr.asdict(wake_model, filter=attr_floris_filter, value_serializer=attr_serializer)
            return {model_string: output}

        # TODO: Uncomment these lines once the models are implemented
        output = dict(
            model_strings=self.model_strings,
            # wake_combination_parameters=create_dict(self.combination_model),
            wake_deflection_parameters=create_dict(self.deflection_model, self.model_strings["deflection_model"]),
            # wake_turbulence_parameters=create_dict(self.turbulence_model),
            wake_velocity_parameters=create_dict(self.velocity_model, self.model_strings["velocity_model"]),
        )
        return output

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
