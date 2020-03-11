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
    #TODO actually make this 'model'.
    # Direct WakeTurbulence model class simply assigns the local TI for each 
    # wind turbine according to observed or 'known' values. Local values of TI 
    # are supplied as an ordered dictionary with inputs for each turbine.

    Args:
        parameter_dictionary: A dictionary as generated from the
            input_reader; it should have the following key-value pairs:
            -   **direct**: A dictionary containing the following
                key-value pairs:

                -   **local_TI_dict**: an ordered dict 

    Returns:
        An instantiated Ishihara(WakeTurbulence) object.
    """

    def __init__(self, parameter_dictionary):
        super().__init__()
        self.logger = setup_logger(name=__name__)
        self.model_string = "direct"
        model_dictionary = parameter_dictionary[self.model_string]

        # wake model parameter
        self.local_TI_dict = model_dictionary["local_TI_dict"]

    def function(self, x_locations, y_locations, z_locations, turbine,
                 turbine_coord, flow_field, turb_u_wake, sorted_map):
        """
        This method ensures that the wake model sees local turbulence intensity 
        values for each constituent wind turbine.

        #TODO include all these inputs? Not really necessary for the model, but 
        # having them ensures that the function call is the same across all 
        # turbulence models.

        Args:
            turb_u_wake (np.array): not used for the current turbulence model,
                included for consistency of function form
            sorted_map (list): sorted turbine_map (coord, turbine)
            x_locations: An array of floats that contains the
                streamwise direction grid coordinates of the flow field
                domain (m).
            y_locations: An array of floats that contains the grid
                coordinates of the flow field domain in the direction
                normal to x and parallel to the ground (m).
            z_locations: An array of floats that contains the grid
                coordinates of the flow field domain in the vertical
                direction (m).
            turbine: A :py:obj:`floris.simulation.turbine` object that
                represents the turbine creating the wake (i.e. the 
                upstream turbine).
            turbine_coord: A :py:obj:`floris.utilities.Vec3` object
                containing the coordinate of the turbine creating the
                wake (m).
            deflection_field: An array of floats that contains the
                amount of wake deflection in meters in the y direction
                at each grid point of the flow field.
            flow_field: A :py:class:`floris.simulation.flow_field`
                object containing the flow field information for the
                wind farm.
        """
        #TODO write function