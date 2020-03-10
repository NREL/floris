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
from .base_velocity_deficit import VelocityDeficit
import numpy as np


class Jensen(VelocityDeficit):
    """
    Wake velocity deficit model based on the Jensen model.
    Jensen is a subclass of :py:class:`floris.simulation.wake_velocity.WakeVelocity` that is
    used to compute the wake velocity deficit based on the classic
    Jensen/Park model. See Jensen, N. O., "A note on wind generator
    interaction." Tech. Rep. Risø-M-2411, Risø, 1983.
    Args:
        parameter_dictionary: A dictionary as generated from the
            input_reader; it should have the following key-value pairs:
            -   **turbulence_intensity**: A dictionary containing the
                following key-value pairs used to calculate wake-added
                turbulence intensity from an upstream turbine, using
                the approach of Crespo, A. and Herna, J. "Turbulence
                characteristics in wind-turbine wakes." *J. Wind Eng
                Ind Aerodyn*. 1996.:
                -   **initial**: A float that is the initial ambient
                    turbulence intensity, expressed as a decimal
                    fraction.
                -   **constant**: A float that is the constant used to
                    scale the wake-added turbulence intensity.
                -   **ai**: A float that is the axial induction factor
                    exponent used in in the calculation of wake-added
                    turbulence.
                -   **downstream**: A float that is the exponent
                    applied to the distance downtream of an upstream
                    turbine normalized by the rotor diameter used in
                    the calculation of wake-added turbulence.
            -   **jensen**: A dictionary containing the following
                key-value pairs:
                -   **we**: A float that is the linear wake decay
                    constant that defines the cone boundary for the
                    wake as well as the velocity deficit. D/2 +/- we*x
                    is the cone boundary for the wake.
    Returns:
        An instantiated Jensen(WaveVelocity) object.
    """

    default_parameters = {
        "we": 0.05
    }

    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.logger = setup_logger(name=__name__)
        self.model_string = "jensen"
        model_dictionary = self._get_model_dict(__class__.default_parameters)
        self.we = float(model_dictionary["we"])

    def function(self, x_locations, y_locations, z_locations, turbine,
                 turbine_coord, deflection_field, flow_field):
        """
        Using the Jensen wake model, this method calculates and returns 
        the wake velocity deficits, caused by the specified turbine, 
        relative to the freestream velocities at the grid of points 
        comprising the wind farm flow field.

        Args:
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
                represents the turbine creating the wake.
            turbine_coord: A :py:obj:`floris.utilities.Vec3` object 
                containing the coordinate of the turbine creating the 
                wake (m).
            deflection_field: An array of floats that contains the 
                amount of wake deflection in meters in the y direction 
                at each grid point of the flow field.
            flow_field: A :py:class:`floris.simulation.flow_field` 
                object containing the flow field information for the 
                wind farm.

        Returns:
            Three arrays of floats that contain the wake velocity 
            deficit in m/s created by the turbine relative to the 
            freestream velocities for the u, v, and w components, 
            aligned with the x, y, and z directions, respectively. The 
            three arrays contain the velocity deficits at each grid 
            point in the flow field. 
        """

        # define the boundary of the wake model ... y = mx + b
        m = self.we
        x = x_locations - turbine_coord.x1
        b = turbine.rotor_radius

        boundary_line = m * x + b

        y_upper = boundary_line + turbine_coord.x2 + deflection_field
        y_lower = -1 * boundary_line + turbine_coord.x2 + deflection_field

        z_upper = boundary_line + turbine.hub_height
        z_lower = -1 * boundary_line + turbine.hub_height

        # calculate the wake velocity
        c = (turbine.rotor_diameter \
             / (2 * self.we * (x_locations - turbine_coord.x1) \
             + turbine.rotor_diameter))**2

        # filter points upstream and beyond the upper and 
        # lower bounds of the wake
        c[x_locations - turbine_coord.x1 < 0] = 0
        c[y_locations > y_upper] = 0
        c[y_locations < y_lower] = 0
        c[z_locations > z_upper] = 0
        c[z_locations < z_lower] = 0

        return 2 * turbine.aI * c * flow_field.u_initial, \
               np.zeros(np.shape(flow_field.u_initial)), \
               np.zeros(np.shape(flow_field.u_initial))

    @property
    def we(self):
        """
        A float that is the linear wake decay constant that defines the cone
            boundary for the wake as well as the velocity deficit. D/2 +/- we*x
            is the cone boundary for the wake.
        Args:
            we (float, int): The linear wake decay constant that defines the
                cone boundary for the wake as well as the velocity deficit.
        Returns:
            float: The linear wake decay constant that defines the cone
                boundary for the wake as well as the velocity deficit.
        """
        return self._we

    @we.setter
    def we(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for we: {}, ' + \
                       'expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._we = value
        if value != __class__.default_parameters['we']:
            self.logger.info(
                ('Current value of we, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['we'])
                )