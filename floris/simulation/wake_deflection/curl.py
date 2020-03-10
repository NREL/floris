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
from .base_velocity_deflection import VelocityDeflection
import numpy as np


class Curl(VelocityDeflection):
    """
    Subclass of the
    :py:class:`floris.simulation.wake_deflection.VelocityDeflection`
    object. Parameters required for Curl wake model:

     - model_grid_resolution: #TODO What does this do?
    """

    def __init__(self, parameter_dictionary):
        """
        Instantiate Curl object and pass function paramter values.

        Args:
            parameter_dictionary (dict): input dictionary with the
                following key-value pair:
                    {
                        "model_grid_resolution": [
                                                    250,
                                                    100,
                                                    75
                                                ],
                    }
        """
        super().__init__(parameter_dictionary)
        self.logger = setup_logger(name=__name__)
        self.model_string = "curl"

    def function(self, x_locations, y_locations, z_locations, turbine, coord,
                 flow_field):
        """
        This function will return the wake centerline predicted with
        the curled wake model. #TODO Eventually. This is coded as
        defined in the Martinez-Tossas et al. paper.
        """
        return np.zeros(np.shape(x_locations))
