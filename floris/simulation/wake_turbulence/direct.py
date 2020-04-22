# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from ...utilities import setup_logger
from .base_wake_turbulence import WakeTurbulence


class Direct(WakeTurbulence):
    """
    **In Development**

    Direct is a subclass of 
    :py:class:`floris.simulation.wake_velocity.WakeTurbulence` that will be
    used to prescribe turbine-local TI values observed from SCADA or other 
    observations.

    #TODO Add Raises fields.
    """

    def __init__(self, parameter_dictionary):
        """
        Initialize a Direct wake turbulence object with the parameter 
        values listed below. If no parameter values are provided, default 
        values are used.

        Args:
            parameter_dictionary (dict): A dictionary as 
            generated from the input_reader; it should have the following 
            key-value pairs:

                - **local_TI_dict**: A dictionary containing TI values for each 
                  turbine in the wind plant.

        Raises:
            ValueError: Invalid value type given for 
            current_turbulece_intensity.
        """
        super().__init__()
        self.logger = setup_logger(name=__name__)
        self.model_string = "direct"
        model_dictionary = parameter_dictionary[self.model_string]

        # wake model parameter
        self.local_TI_dict = model_dictionary["local_TI_dict"]

    def function(self, ambient_TI, coord_ti, turbine_coord, turbine):
        """
        Main function for calculating wake added turbulence as a function of 
        external conditions and wind turbine operation. This function is 
        accessible through the :py:class:`floris.simulation.Wake` class as the 
        :py:meth:`floris.simulation.Wake.turbulence_function` method.

        **NOTE:** Input arguments are not currently used, as no model is 
        implemented. Arguments are retained currently for consistency of 
        :py:meth:`floris.simulation.Wake.turbulence_function` call.

        Args:
            ambient_TI (float): TI of the background flowfield.
            coord_ti (:py:class:`floris.utilities.Vec3`): Coordinate where TI 
               is to be calculated (e.g. downstream wind turbines).
            turbine_coord (:py:class:`floris.utilities.Vec3`): Coordinate of 
               the wind turbine adding turbulence to the flow.
            turbine (:py:class:`floris.simulation.Turbine`): Wind turbine 
               adding turbulence to the flow.

        Returns:
            float: Wake-added turbulence reported for the current
            wind turbine (**turbine**).
        """
        
        #TODO develop and test function.
        turbine.current_turbulence_intensity = self.parameter_dictionary \
            ['local_TI_dict'][turbine]

        return self.parameter_dictionary['local_TI_dict'][turbine]