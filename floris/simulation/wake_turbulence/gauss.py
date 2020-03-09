# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


class Gauss(WakeTurbulence):
    """
    Gauss is a wake velocity subclass that contains objects related to the
    Gaussian wake model.

    Gauss is a subclass of
    :py:class:`floris.simulation.wake_velocity.WakeTurbulence` that is
    used to compute the wake velocity deficit based on the Gaussian
    wake model with self-similarity. The Gaussian wake model includes a
    Gaussian wake velocity deficit profile in the y and z directions
    and includes the effects of ambient turbulence, added turbulence
    from upstream wakes, as well as wind shear and wind veer. For more
    information about the Gauss wake model theory, see:

    Abkar, M. and Porte-Agel, F. "Influence of atmospheric stability on
    wind-turbine wakes: A large-eddy simulation study." *Physics of
    Fluids*, 2015.

    Bastankhah, M. and Porte-Agel, F. "A new analytical model for
    wind-turbine wakes." *Renewable Energy*, 2014.

    Bastankhah, M. and Porte-Agel, F. "Experimental and theoretical
    study of wind turbine wakes in yawed conditions." *J. Fluid
    Mechanics*, 2016.

    Niayifar, A. and Porte-Agel, F. "Analytical modeling of wind farms:
    A new approach for power prediction." *Energies*, 2016.

    Dilip, D. and Porte-Agel, F. "Wind turbine wake mitigation through
    blade pitch offset." *Energies*, 2017.

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
        An instantiated Gauss(WakeTurbulence) object.
    """

    def __init__(self, parameter_dictionary):
        super().__init__()
        self.model_string = "gauss"
        model_dictionary = parameter_dictionary[self.model_string]

        # turbulence parameters
        self.ti_initial = float(model_dictionary["initial"])
        self.ti_constant = float(model_dictionary["constant"])
        self.ti_ai = float(model_dictionary["ai"])
        self.ti_downstream = float(model_dictionary["downstream"])

    def function(self, ambient_TI, coord_ti, turbine_coord, turbine):
        """
        Using the Gaussian wake model, this method calculates and
        returns the wake velocity deficits, caused by the specified
        turbine, relative to the freestream velocities at the grid of
        points comprising the wind farm flow field.

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
        if type(value) is float:
            self._ti_initial = value
        else:
            raise ValueError(("Invalid value given for "
                              "ti_initial: {}").format(value))

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
        if type(value) is float:
            self._ti_constant = value
        else:
            raise ValueError(("Invalid value given for "
                              "ti_constant: {}").format(value))
    
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
        if type(value) is float:
            self._ti_ai = value
        else:
            raise ValueError("Invalid value given for ti_ai: {}".format(value))

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
        if type(value) is float:
            self._ti_downstream = value
        else:
            raise ValueError(("Invalid value given for "
                              "ti_downstream: {}").format(value))