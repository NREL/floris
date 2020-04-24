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
 

from .wake_velocity.base_velocity_deficit import VelocityDeficit
from .wake_velocity.curl import Curl as CurlDeficit
from .wake_velocity.gaussianModels.gauss_legacy \
    import LegacyGauss as LegacyGaussDeficit
from .wake_velocity.gaussianModels.gauss import Gauss as GaussDeficit
from .wake_velocity.jensen import Jensen
from .wake_velocity.multizone import MultiZone
from .wake_velocity.gaussianModels.ishihara_qian \
    import IshiharaQian as IshiharaQianDeficit
from .wake_velocity.gaussianModels.blondel import Blondel as BlondelDeficit

from .wake_deflection.base_velocity_deflection import VelocityDeflection
from .wake_deflection.jimenez import Jimenez
from .wake_deflection.gauss import Gauss as GaussDeflection
from .wake_deflection.curl import Curl as CurlDeflection

from .wake_turbulence.base_wake_turbulence import WakeTurbulence
from .wake_turbulence.crespo_hernandez \
    import CrespoHernandez as CrespoHernandezTurbulence
from .wake_turbulence.ishihara_qian \
    import IshiharaQian as IshiharaQianTurbulence
from .wake_turbulence.direct import Direct as DirectTurbulence

from .wake_combination.base_wake_combination import WakeCombination
from .wake_combination.fls import FLS
from .wake_combination.sosfs import SOSFS
from .wake_combination.max import MAX


class Wake():
    """
    Wake is a container class for the wake velocity, deflection,
    turbulence, and combination models.
    """
    def __init__(self, instance_dictionary):
        """
        Configures the mapping from model strings to their respective classes
        and unpacks the model parameters.

        Args:
            instance_dictionary (dict): Dictionary consisting of the following
                items:

                - velocity_model (str): The name of the velocity model to be
                    instantiated.
                - turbulence_model (str): The name of the turbulence model to be
                    instantiated.
                - deflection_model (str): The name of the deflection model to be
                    instantiated.
                - combination_model (str): The name of the combination model to
                    be instantiated.
                - parameters (dict): See specific model classes for parameters.
        """
        properties = instance_dictionary["properties"]
        if "parameters" not in properties.keys():
            self.parameters = {}
        else:
            self.parameters = properties["parameters"]
        # TODO: Add support for tuning wake combination parameters?
        # wake_comb_parameters = parameters["wake_combination_parameters"]

        self._velocity_models = {
            "jensen": Jensen,
            "multizone": MultiZone,
            "gauss": GaussDeficit,
            "gauss_legacy": LegacyGaussDeficit,
            "ishihara_qian": IshiharaQianDeficit,
            "curl": CurlDeficit,
            "blondel": BlondelDeficit
        }
        self.velocity_model = properties["velocity_model"]

        self._turbulence_models = {
            "crespo_hernandez": CrespoHernandezTurbulence,
            "ishihara_qian": IshiharaQianTurbulence,
            "direct": DirectTurbulence,
            "None": WakeTurbulence
        }
        self.turbulence_model = properties["turbulence_model"]

        self._deflection_models = {
            "jimenez": Jimenez,
            "gauss": GaussDeflection,
            "curl": CurlDeflection
        }
        self.deflection_model = properties["deflection_model"]

        self._combination_models = {
            "fls": FLS,
            "sosfs": SOSFS,
            "max": MAX
        }
        self.combination_model = properties["combination_model"]

    # Getters & Setters
    @property
    def velocity_model(self):
        """
        Velocity model.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (str, :py:class:`~.base_velocity_deficit.VelocityDeficit`):
                A string for the model to set or the model instance itself.

        Returns:
            :py:class:`~.base_velocity_deficit.VelocityDeficit`:
                Model currently set.

        Raises:
            ValueError: Invalid value.    
        """
        return self._velocity_model

    @velocity_model.setter
    def velocity_model(self, value):
        if type(value) is str:
            if "wake_velocity_parameters" not in self.parameters.keys():
                self._velocity_model = self._velocity_models[value]({})
            else:
                self._velocity_model = self._velocity_models[value](
                    self.parameters["wake_velocity_parameters"])
        elif isinstance(value, VelocityDeficit):
            self._velocity_model = value
        else:
            raise ValueError(
                "Invalid value given for VelocityDeficit: {}".format(value))

    @property
    def turbulence_model(self):
        """
        Turbulence model.

        **Note**: This is a virtual property used to "get" or "set" a value.

        Args:
            value (str, :py:class:`~.base_wake_turbulence.WakeTurbulence`):
                A string for the model to set or the model instance itself.

        Returns:
            :py:class:`~.base_wake_turbulence.WakeTurbulence`:
                Model currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._turbulence_model

    @turbulence_model.setter
    def turbulence_model(self, value):
        if type(value) is str:
            if "wake_turbulence_parameters" not in self.parameters.keys():
                self._turbulence_model = self._turbulence_models[value]({})
            else:
                self._turbulence_model = self._turbulence_models[value](
                    self.parameters["wake_turbulence_parameters"])
        elif isinstance(value, WakeTurbulence):
            self._turbulence_model = value
        else:
            raise ValueError(
                "Invalid value given for WakeTurbulence: {}".format(value))

    @property
    def deflection_model(self):
        """
        Deflection model.

        **Note**: This is a virtual property used to "get" or "set" a value.

        Args:
            value (str, :py:class:`~.base_velocity_deflection.VelocityDeflection`):
                A string for the model to set or the model instance itself.

        Returns:
            :py:class:`~.base_velocity_deflection.VelocityDeflection`:
                Model currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._deflection_model

    @deflection_model.setter
    def deflection_model(self, value):
        if type(value) is str:
            if "wake_deflection_parameters" not in self.parameters.keys():
                self._deflection_model = self._deflection_models[value]({})
            else:
                self._deflection_model = self._deflection_models[value](
                    self.parameters["wake_deflection_parameters"])
        elif isinstance(value, VelocityDeflection):
            self._deflection_model = value
        else:
            raise ValueError(
                "Invalid value given for VelocityDeflection: {}".format(value))

    @property
    def combination_model(self):
        """
        Combination model.

        **Note**: This is a virtual property used to "get" or "set" a value.

        Args:
            value (str, :py:class:`~.base_wake_combination.WakeCombination`):
                A string for the model to set or the model instance itself.

        Returns:
            :py:class:`~.base_wake_combination.WakeCombination`:
                Model currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._combination_model

    @combination_model.setter
    def combination_model(self, value):
        if type(value) is str:
            self._combination_model = self._combination_models[value]()
        elif isinstance(value, wake_combination.WakeCombination):
            self._combination_model = value
        else:
            raise ValueError(
                "Invalid value given for WakeCombination: {}".format(value))

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
