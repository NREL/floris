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


class Direct(WakeTurbulence):
    """
    **In Development**

    Direct is a wake-turbulence model that will be used to prescribe
    turbine-local TI values observed from SCADA or other observations.
    """

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
        super().__init__()
        self.model_string = "direct"
        model_dictionary = parameter_dictionary[self.model_string]

        # wake model parameter
        self.local_TI_dict = model_dictionary["local_TI_dict"]

    def function(self, ambient_TI, coord_ti, turbine_coord, turbine):
        """
        Calculates wake-added turbulence as a function of
        external conditions and wind turbine operation. This function is
        accessible through the :py:class:`~.wake.Wake` class as the
        :py:meth:`~.Wake.turbulence_function` method.

        **NOTE:** Input arguments are not currently used, as no model is
        implemented. Arguments are retained currently for consistency of
        :py:meth:`~.wake.Wake.turbulence_function` call.

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
        # TODO develop and test function.
        turbine.current_turbulence_intensity = self.parameter_dictionary[
            "local_TI_dict"
        ][turbine]

        return self.parameter_dictionary["local_TI_dict"][turbine]
