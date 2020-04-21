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
    Wake is a container class for the various wake model objects. In
    particular, Wake holds references to the velocity, deflection, turbulence,
    and combination models as well as their parameters.
    
    Raises:
        ValueError: Invalid value given for VelocityDeficit
        ValueError: Invalid value given for WakeTurbulence
        ValueError: Invalid value given for WakeCombination
        ValueError: Invalid value given for VelocityDeflection
    """
    

    def __init__(self, instance_dictionary):
        """
        Some info for the initialization of this class.
        # TODO

        Args:
            instance_dictionary (dict): Dictionary consisting of the following key-value pairs:

                - velocity_model
                - turbulence_model
                - deflection_model
                - combination_model
                - parameters - see model classes
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
        Get or assign the velocity model.

        **Note**: This is a virtual property used to "get" or "set" a value.

        Args:
            value (str, `VelocityDefecit`_): Velocity model to set.

        Returns:
            `VelocityDefecit`_: Velocity model currently set.

        Raises:
            ValueError: Invalid value given for `VelocityDefecit`_
        
        .. _VelocityDefecit:
            :py:class:`floris.simulation.wake_velocity.base_velocity_deficit.VelocityDeficit`
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
        Get or assign the turbulence model.

        **Note**: This is a virtual property used to "get" or "set" a value.

        Args:
            value (str, `WakeTurbulence`_): Turbulence model to set.

        Returns:
            `WakeTurbulence`_: Turbulence model currently set.

        Raises:
            ValueError: Invalid value given for `WakeTurbulence`_
        
        .. _WakeTurbulence:
            :py:class:`floris.simulation.wake_turbulence.base_wake_turbulence.WakeTurbulence
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
        Get or assign the deflection model.

        **Note**: This is a virtual property used to "get" or "set" a value.

        Args:
            value (str, `VelocityDeflection`_): Deflection model to set.

        Returns:
            `VelocityDeflection`_: Deflection model currently set.

        Raises:
            ValueError: Invalid value given for `VelocityDeflection`_
        
        .. _VelocityDeflection:
            :py:class:`floris.simulation.wake_deflection.base_wake_deflection.VelocityDeflection
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
        Get or assign the combination model.

        **Note**: This is a virtual property used to "get" or "set" a value.

        Args:
            value (str, `WakeCombination`_): Combination model to set.

        Returns:
            `WakeCombination`_: Combination model currently set.

        Raises:
            ValueError: Invalid value given for `WakeCombination`_
        
        .. _WakeCombination:
            :py:class:`floris.simulation.wake_combination.base_wake_combination.WakeCombination
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
        Get the function used as the deflection model.

        Returns:
            `VelocityDeflection`_

        .. _VelocityDeflection:
            :py:class:`floris.simulation.wake_deflection.base_wake_deflection.VelocityDeflection
        """
        return self.deflection_model.function

    @property
    def velocity_function(self):
        """
        Get the function used as the velocity deficit model.

        Returns:
            `VelocityDefecit`_

        .. _VelocityDefecit:
            :py:class:`floris.simulation.wake_velocity.base_velocity_deficit.VelocityDeficit`
        """
        return self.velocity_model.function

    @property
    def turbulence_function(self):
        """
        Get the function used as the turbulence model.
        
        Returns:
            `WakeTurbulence`_

        .. _WakeTurbulence:
            :py:class:`floris.simulation.wake_turbulence.base_wake_turbulence.WakeTurbulence
        """
        return self.turbulence_model.function

    @property
    def combination_function(self):
        """
        Get the function used as the combination model.
        
        Returns:
            `WakeCombination`_

        .. _WakeCombination:
            :py:class:`floris.simulation.wake_combination.base_wake_combination.WakeCombination
        """
        return self.combination_model.function
