# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np

from .base_velocity_deflection import VelocityDeflection


class Curl(VelocityDeflection):
    """
    Stand-in class for the curled wake model. Wake deflection with the curl
    model is handled inherently in the wake velocity portion of the model.
    Passes zeros for deflection values. See
    :cite:`cdm-martinez2019aerodynamics` for additional info on the curled wake
    model.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: cdm-
    """

    def __init__(self, parameter_dictionary):
        """
        See super-class for initialization details. See
        :py:class:`floris.simulation.wake_velocity.curl` for details on
        `parameter_dictionary`.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
        """
        super().__init__(parameter_dictionary)
        self.model_string = "curl"

    def function(
        self, x_locations, y_locations, z_locations, turbine, coord, flow_field
    ):
        """
        Passes zeros for wake deflection as deflection is inherently handled in
        the wake velocity portion of the curled wake model.

        Args:
            x_locations (np.array): An array of floats that contains the
                streamwise direction grid coordinates of the flow field
                domain (m).
            y_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the direction normal to
                x and parallel to the ground (m).
            z_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the vertical
                direction (m).
            turbine (:py:obj:`floris.simulation.turbine`): Object that
                represents the turbine creating the wake.
            coord (:py:obj:`floris.utilities.Vec3`): Object containing
                the coordinate of the turbine creating the wake (m).
            flow_field (:py:class:`floris.simulation.flow_field`): Object
                containing the flow field information for the wind farm.

        Returns:
            np.array: Zeros the same size as the flow field grid points.
        """
        return np.zeros(np.shape(x_locations))
