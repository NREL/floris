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


class CrespoHernandez(WakeTurbulence):
    """
    CrespoHernandez is a wake turbulence subclass that contains objects related
    to the wake turbulence model.

    CrespoHernandez is a subclass of
    :py:class:`floris.simulation.wake_velocity.WakeTurbulence` that is
    used to compute the... #TODO finish updating docstring; add reference

    Args:
        parameter_dictionary: A dictionary as generated from the
            input_reader; it should have the following key-value pairs:

            -   **gauss**: A dictionary containing the following
                key-value pairs:

                -   **initial**: A float that is the initial ambient
                    turbulence intensity, expressed as a decimal
                    fraction.
                -   **constant**: A float that is the constant used to
                    scale the wake-added turbulence intensity.
                -   **ai**: A float that is the axial induction factor
                    exponent used in in the calculation of wake-added
                    turbulence.
                -   **downstream**: A float that is the exponent
                    applied to the distance downstream of an upstream
                    turbine normalized by the rotor diameter used in
                    the calculation of wake-added turbulence.


    Returns:
        An instantiated CrespoHernandez(WakeTurbulence) object.
    """

    default_parameters = {
        "initial": 0.5,
        "constant": 0.9,
        "ai": 0.75,
        "downstream": -0.325
    }



    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.logger = setup_logger(name=__name__)
        self.model_string = "crespo_hernandez"
        model_dictionary = self._get_model_dict(__class__.default_parameters)

        # turbulence parameters
        self.ti_initial = model_dictionary["initial"]
        self.ti_constant = model_dictionary["constant"]
        self.ti_ai = model_dictionary["ai"]
        self.ti_downstream = model_dictionary["downstream"]

    def function(self, ambient_TI, coord_ti, turbine_coord, turbine):
        """
        #TODO update docstring

        Args:
            turb_u_wake (np.array): u-component of turbine wake field
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
                represents the turbine creating the wake.
            turbine_coord: A :py:obj:`floris.utilities.Vec3` object
                containing the coordinate of the turbine creating the
                wake (m).
            flow_field: A :py:class:`floris.simulation.flow_field`
                object containing the flow field information for the
                wind farm.
        """

        ti_initial = ambient_TI

        # turbulence intensity calculation based on Crespo et. al.
        ti_calculation = self.ti_constant \
            * turbine.aI**self.ti_ai \
            * ti_initial**self.ti_initial \
            * ((coord_ti.x1 - turbine_coord.x1) / turbine.rotor_diameter)**self.ti_downstream

        # Update turbulence intensity of downstream turbines
        return ti_calculation

    @property
    def ti_initial(self):
        """
        Parameter that is the initial ambient turbulence intensity, expressed as
            a decimal fraction.

        Args:
            ti_initial (float): Initial ambient turbulence intensity.

        Returns:
            float: Initial ambient turbulence intensity.
        """
        return self._ti_initial

    @ti_initial.setter
    def ti_initial(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for ' + \
                       'initial: {}, expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._ti_initial = value
        if value != __class__.default_parameters['initial']:
            self.logger.info(
                ('Current value of initial, {0}, is not equal to tuned ' + \
                'value of {1}.').format(
                    value, __class__.default_parameters['initial']
                )
            )


    @property
    def ti_constant(self):
        """
        Parameter that is the constant used to scale the wake-added turbulence
            intensity.

        Args:
            ti_constant (float): Scales the wake-added turbulence intensity.

        Returns:
            float: Scales the wake-added turbulence intensity.
        """
        return self._ti_constant

    @ti_constant.setter
    def ti_constant(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for ' + \
                       'constant: {}, expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._ti_constant = value
        if value != __class__.default_parameters['constant']:
            self.logger.info(
                ('Current value of constant, {0}, is not equal to tuned ' + \
                'value of {1}.').format(
                    value, __class__.default_parameters['constant']
                )
            )
    
    @property
    def ti_ai(self):
        """
        Parameter that is the axial induction factor exponent used in in the
            calculation of wake-added turbulence.

        Args:
            ti_ai (float): Axial induction factor exponent for wake-added
                turbulence.

        Returns:
            float: Axial induction factor exponent for wake-added turbulence.
        """
        return self._ti_ai

    @ti_ai.setter
    def ti_ai(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for ' + \
                       'ai: {}, expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._ti_ai = value
        if value != __class__.default_parameters['ai']:
            self.logger.info(
                ('Current value of ai, {0}, is not equal to tuned ' + \
                'value of {1}.').format(
                    value, __class__.default_parameters['ai']
                )
            )

    @property
    def ti_downstream(self):
        """
        Parameter that is the exponent applied to the distance downstream of an
            upstream turbine normalized by the rotor diameter used in the
            calculation of wake-added turbulence.

        Args:
            ti_downstream (float): Downstream distance exponent for
                wake-added turbulence.

        Returns:
            float: Downstream distance exponent for wake-added turbulence.
        """
        return self._ti_downstream

    @ti_downstream.setter
    def ti_downstream(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for ' + \
                       'downstream: {}, expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._ti_downstream = value
        if value != __class__.default_parameters['downstream']:
            self.logger.info(
                ('Current value of downstream, {0}, is not equal to ' + \
                'tuned value of {1}.').format(
                    value, __class__.default_parameters['downstream']
                )
            )