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

from src.utilities import model_attrib
from src.simulation.base_class import BaseClass
from src.simulation.wake_velocity.curl import CurlVelocityDeficit
from src.simulation.wake_velocity.jensen import JensenVelocityDeficit
from src.simulation.wake_deflection.jimenez import JimenezVelocityDeflection


# from .wake_combination.fls import FLS
# from .wake_combination.max import MAX
# from .wake_deflection.curl import Curl as CurlDeflection
# from .wake_deflection.gauss import Gauss as GaussDeflection
# from .wake_combination.sosfs import SOSFS
# from .wake_turbulence.direct import Direct as DirectTurbulence
# from .wake_velocity.multizone import MultiZone
# from .wake_velocity.turbopark import TurbOPark
# from .wake_turbulence.ishihara_qian import IshiharaQian as IshiharaQianTurbulence
# from .wake_turbulence.crespo_hernandez import (
#     CrespoHernandez as CrespoHernandezTurbulence,
# )
# from .wake_velocity.gaussianModels.gauss import Gauss as GaussDeficit
# from .wake_velocity.base_velocity_deficit import VelocityDeficit
# from .wake_turbulence.base_wake_turbulence import WakeTurbulence
# from .wake_velocity.gaussianModels.blondel import Blondel as BlondelDeficit
# from .wake_combination.base_wake_combination import WakeCombination
# from .wake_deflection.base_velocity_deflection import VelocityDeflection
# from .wake_velocity.gaussianModels.gauss_legacy import LegacyGauss as LegacyGaussDeficit
# from .wake_velocity.gaussianModels.ishihara_qian import (
#     IshiharaQian as IshiharaQianDeficit,
# )


MODEL_MAP = {
    "wake_combination": {},
    "wake_deflection": {"jimenez": JimenezVelocityDeflection},
    "wake_turbulence": {},
    "wake_velocity": {"curl": CurlVelocityDeficit, "jensen": JensenVelocityDeficit},
}


@attr.s(auto_attribs=True)
class Wake(BaseClass):
    """
    Wake is a container class for the wake velocity, deflection,
    turbulence, and combination models.

    Args:
        wake (:obj:`dict`): The wake's properties input dictionary
            - velocity_model (str): The name of the velocity model to be
                instantiated.
            - turbulence_model (str): The name of the turbulence model to be
                instantiated.
            - deflection_model (str): The name of the deflection model to be
                instantiated.
            - combination_model (str): The name of the combination model to
                be instantiated.
    """

    model_strings: dict
    wake_combination_parameters: dict = attr.ib(factory=dict)
    wake_deflection_parameters: dict = attr.ib(factory=dict)
    wake_turbulence_parameters: dict = attr.ib(factory=dict)
    wake_velocity_parameters: dict = attr.ib(factory=dict)

    combination_model: Any = attr.ib(init=False)
    deflection_model: Any = attr.ib(init=False)
    turbulence_model: Any = attr.ib(init=False)
    velocity_model: Any = attr.ib(init=False)
    model_string: str = model_attrib(default="wake")

    def __attrs_post_init__(self) -> None:
        self.model_generator()

    def model_generator(self) -> Any:
        models = ("combination", "deflection", "turbulence", "velocity")
        wake_models = {}
        for model_type in models:
            model_opts = MODEL_MAP[f"wake_{model_type}"]
            try:
                model_string = self.model_strings[f"{model_type}_model"]
            except KeyError:
                wake_models[model_type] = None
                continue  # TODO: We don't have to create all model types?

            print(model_string)
            if model_string is None:
                wake_models[model_type] = None
                continue  # TODO: We don't have to create all model types?
            elif model_string not in model_opts:
                raise ValueError(
                    f"Invalid wake {model_type} model: {model_string}. Valid options include: {', '.join(model_opts)}"
                )
            model = model_opts[model_string]
            model_def = getattr(self, f"wake_{model_type}_parameters")[model_string]
            wake_models[model_type] = model.from_dict(model_def)

        self.combination_model = wake_models["combination"]
        self.deflection_model = wake_models["deflection"]
        self.turbulence_model = wake_models["turbulence"]
        self.velocity_model = wake_models["velocity"]

    @property
    def deflection_function(self):
        """
        Function to calculate the wake deflection. This is dynamically
        gotten from the currently set model.

        Returns:
            :py:class:`~.base_velocity_deflection.VelocityDeflection`
        """
        return self.deflection_model.function

    @property
    def velocity_function(self):
        """
        Function to calculate the velocity deficit. This is dynamically
        gotten from the currently set model.

        Returns:
            :py:class:`~.base_velocity_deficit.VelocityDeficit`
        """
        return self.velocity_model.function

    @property
    def turbulence_function(self):
        """
        Function to calculate the turbulence impact. This is dynamically
        gotten from the currently set model.

        Returns:
            :py:class:`~.wake_turbulence.base_wake_turbulence.WakeTurbulence`
        """
        return self.turbulence_model.function

    @property
    def combination_function(self):
        """
        Function to apply the calculated wake to the freestream field.
        This is dynamically gotten from the currently set model.

        Returns:
            :py:class:`~.wake_combination.base_wake_combination.WakeCombination`
        """
        return self.combination_model.function
