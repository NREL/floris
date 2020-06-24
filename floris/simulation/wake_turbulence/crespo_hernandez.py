# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from .base_wake_turbulence import WakeTurbulence


class CrespoHernandez(WakeTurbulence):
    """
    CrespoHernandez is a wake-turbulence model that is used to compute
    additional variability introduced to the flow field by operation of a wind
    turbine. Implementation of the model follows the original formulation and
    limitations outlined in :cite:`cht-crespo1996turbulence`.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: cht-
    """

    default_parameters = {
        "initial": 0.1,
        "constant": 0.37,
        "ai": 0.8,
        "downstream": -0.275,
    }

    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                -   **initial** (*float*): The initial ambient turbulence
                    intensity, expressed as a decimal fraction.
                -   **constant** (*float*): The constant used to scale the
                    wake-added turbulence intensity.
                -   **ai** (*float*): The axial induction factor exponent used
                    in in the calculation of wake-added turbulence.
                -   **downstream** (*float*): The exponent applied to the
                    distance downstream of an upstream turbine normalized by
                    the rotor diameter used in the calculation of wake-added
                    turbulence.
        """
        super().__init__(parameter_dictionary)
        self.model_string = "crespo_hernandez"
        model_dictionary = self._get_model_dict(__class__.default_parameters)

        # turbulence parameters
        self.ti_initial = model_dictionary["initial"]
        self.ti_constant = model_dictionary["constant"]
        self.ti_ai = model_dictionary["ai"]
        self.ti_downstream = model_dictionary["downstream"]

    def function(self, ambient_TI, coord_ti, turbine_coord, turbine):
        """
        Calculates wake-added turbulence as a function of
        external conditions and wind turbine operation. This function is
        accessible through the :py:class:`~.wake.Wake` class as the
        :py:meth:`~.Wake.turbulence_function` method.

        Args:
            ambient_TI (float): TI of the background flow field.
            coord_ti (:py:class:`~.utilities.Vec3`): Coordinate where TI
                is to be calculated (e.g. downstream wind turbines).
            turbine_coord (:py:class:`~.utilities.Vec3`): Coordinate of
                the wind turbine adding turbulence to the flow.
            turbine (:py:class:`~.turbine.Turbine`): Wind turbine
                adding turbulence to the flow.

        Returns:
            float: Wake-added turbulence from the current
                wind turbine (**turbine**) at location specified
                by (**coord_ti**).
        """
        ti_initial = ambient_TI

        # turbulence intensity calculation based on Crespo et. al.
        ti_calculation = (
            self.ti_constant
            * turbine.aI ** self.ti_ai
            * ti_initial ** self.ti_initial
            * ((coord_ti.x1 - turbine_coord.x1) / turbine.rotor_diameter)
            ** self.ti_downstream
        )

        # Update turbulence intensity of downstream turbines
        return ti_calculation

    @property
    def ti_initial(self):
        """
        Parameter that is the initial ambient turbulence intensity, expressed as
        a decimal (e.g. 10% TI -> 0.10).

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            ti_initial (float): Initial ambient turbulence intensity.

        Returns:
            float: Initial ambient turbulence intensity.

        Raises:
            ValueError: Invalid value.
        """
        return self._ti_initial

    @ti_initial.setter
    def ti_initial(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for " + "initial: {}, expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._ti_initial = value
        if value != __class__.default_parameters["initial"]:
            self.logger.info(
                (
                    "Current value of initial, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["initial"])
            )

    @property
    def ti_constant(self):
        """
        Constant parameter used to scale the wake-added turbulence
        intensity.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            ti_constant (float): Scales the wake-added turbulence intensity.

        Returns:
            float: Scales the wake-added turbulence intensity.

        Raises:
            ValueError: Invalid value.
        """
        return self._ti_constant

    @ti_constant.setter
    def ti_constant(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for " + "constant: {}, expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._ti_constant = value
        if value != __class__.default_parameters["constant"]:
            self.logger.info(
                (
                    "Current value of constant, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["constant"])
            )

    @property
    def ti_ai(self):
        """
        Axial induction factor exponent used in in the calculation of
        wake-added turbulence.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            ti_ai (float): Axial induction factor exponent for wake-added
              turbulence.

        Returns:
            float: Axial induction factor exponent for wake-added turbulence.

        Raises:
            ValueError: Invalid value.
        """
        return self._ti_ai

    @ti_ai.setter
    def ti_ai(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for " + "ai: {}, expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._ti_ai = value
        if value != __class__.default_parameters["ai"]:
            self.logger.info(
                (
                    "Current value of ai, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["ai"])
            )

    @property
    def ti_downstream(self):
        """
        Exponent applied to the distance from an upstream turbine normalized by
        the rotor diameter. Used in the calculation of wake-added turbulence.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            ti_downstream (float): Downstream distance exponent for
              wake-added turbulence.

        Returns:
            float: Downstream distance exponent for wake-added turbulence.

        Raises:
            ValueError: Invalid value.
        """
        return self._ti_downstream

    @ti_downstream.setter
    def ti_downstream(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for " + "downstream: {}, expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._ti_downstream = value
        if value != __class__.default_parameters["downstream"]:
            self.logger.info(
                (
                    "Current value of downstream, {0}, is not equal to "
                    + "tuned value of {1}."
                ).format(value, __class__.default_parameters["downstream"])
            )
