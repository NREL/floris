# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from ...utilities import cosd, sind, tand
from .base_velocity_deficit import VelocityDeficit
import numpy as np


class Ishihara(VelocityDeficit):
    """
    Ishihara is a wake velocity subclass that contains objects related to the
    Gaussian wake model that include a near-wake correction.

    Ishihara is a subclass of
    :py:class:`floris.simulation.wake_velocity.VelocityDeficit` that is
    used to compute the wake velocity deficit based on the Gaussian
    wake model with self-similarity and a near wake correction. The Ishihara
    wake model includes a Gaussian wake velocity deficit profile in the y and z
    directions and includes the effects of ambient turbulence, added turbulence
    from upstream wakes, as well as wind shear and wind veer. For more info,
    see:

    Ishihara, Takeshi, and Guo-Wei Qian. "A new Gaussian-based analytical wake
    model for wind turbines considering ambient turbulence intensities and
    thrust coefficient effects." Journal of Wind Engineering and Industrial
    Aerodynamics 177 (2018): 275-292.

   Args:
        parameter_dictionary: A dictionary as generated from the
            input_reader; it should have the following key-value pairs:
            -   **ishihara**: A dictionary containing the following
                key-value pairs:

                -   **kstar**: A float that is a parameter used to
                    determine the linear relationship between the
                    turbulence intensity and the width of the Gaussian
                    wake shape.
                -   **epsilon**: A float that is a second parameter used to
                    determine the linear relationship between the
                    turbulence intensity and the width of the Gaussian
                    wake shape.
                -   **a**: constant coefficient used in calculation of              wake-added turbulence.
                -   **b**: linear coefficient used in calculation of                wake-added turbulence.
                -   **c**: near-wake coefficient used in calculation of             wake-added turbulence.

    Returns:
        An instantiated Ishihara(WaveVelocity) object.
    """
    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "ishihara"
        model_dictionary = self._get_model_dict()

        # wake model parameter
        self.kstar = model_dictionary["kstar"]
        self.epsilon = model_dictionary["epsilon"]
        self.a = model_dictionary["a"]
        self.b = model_dictionary["b"]
        self.c = model_dictionary["c"]

    def function(self, x_locations, y_locations, z_locations, turbine,
                 turbine_coord, deflection_field, flow_field):
        """
        Using the Gaussian wake model, this method calculates and
        returns the wake velocity deficits, caused by the specified
        turbine, relative to the freestream velocities at the grid of
        points comprising the wind farm flow field.

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
            deflection_field: #TODO not yet integrated into the Ishihara model
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
        # added turbulence model
        TI = turbine._turbulence_intensity

        # turbine parameters
        D = turbine.rotor_diameter

        yaw = -1 * turbine.yaw_angle  # opposite sign convention in this model
        Ct = turbine.Ct
        U_local = flow_field.u_initial
        local_x = x_locations - turbine_coord.x1
        local_y = y_locations - turbine_coord.x2
        local_z = z_locations - turbine_coord.x3  # adjust for hub height

        # coordinate info
        r = np.sqrt(local_y**2 + (local_z)**2)

        def parameter_value_from_dict(pdict, Ct, TI):
            return pdict['const'] * Ct**(pdict['Ct']) * TI**(pdict['TI'])

        kstar = parameter_value_from_dict(self.kstar, Ct, TI)
        epsilon = parameter_value_from_dict(self.epsilon, Ct, TI)
        a = parameter_value_from_dict(self.a, Ct, TI)
        b = parameter_value_from_dict(self.b, Ct, TI)
        c = parameter_value_from_dict(self.c, Ct, TI)

        k1 = np.cos(np.pi / 2 * (r / D - 0.5))**2
        k1[r / D > 0.5] = 1.0

        k2 = np.cos(np.pi / 2 * (r / D + 0.5))**2
        k2[r / D > 0.5] = 1.0

        # Representative wake width = \sigma / D
        wake_width = kstar * (local_x / D) + epsilon

        # wake velocity deficit = \Delta U (x,y,z) / U_h
        velDef = 1 / (a + b * (local_x / D) + c *
                      (1 +
                       (local_x / D))**(-2))**2 * np.exp(-(r)**2 /
                                                         (2 *
                                                          (wake_width * D)**2))

        # trim wakes to 1 D upstream to avoid artifacts
        yR = y_locations - turbine_coord.x2
        xR = yR * tand(yaw) + turbine_coord.x1 - D
        velDef[x_locations < xR] = 0

        velDef = U_local * (velDef)

        return velDef, np.zeros(np.shape(velDef)), np.zeros(np.shape(velDef))

    @property
    def kstar(self):
        """
        Parameter that is used to determine the linear relationship between the
            turbulence intensity and the width of the Gaussian wake shape.

        Args:
            kstar (float): Factor for relationship between the turbulence
                intensity and the width of the Gaussian wake shape.

        Returns:
            float: Factor for relationship between the turbulence intensity and
                the width of the Gaussian wake shape.
        """
        return self._kstar

    @kstar.setter
    def kstar(self, value):
        if type(value) is dict and set(value) == set(['const', 'Ct', 'TI']):
            self._kstar = value
        else:
            raise ValueError("Invalid value given for kstar: {}".format(value))

    @property
    def epsilon(self):
        """
        Parameter that is used to determine the linear relationship between the
            turbulence intensity and the width of the Gaussian wake shape.

        Args:
            epsilon (float): Factor for relationship between the turbulence
                intensity and the width of the Gaussian wake shape.

        Returns:
            float: Factor for relationship between the turbulence intensity and
                the width of the Gaussian wake shape.
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if type(value) is dict and set(value) == set(['const', 'Ct', 'TI']):
            self._epsilon = value
        else:
            raise ValueError("Invalid value given for " +
                             "epsilon: {}".format(value))

    @property
    def a(self):
        """
        Constant coefficient used in calculation of wake-added turbulence.

        Args:
            a (float): Constant coefficient used in calculation of wake-added
                turbulence.

        Returns:
            float: Constant coefficient used in calculation of wake-added
                turbulence.
        """
        return self._a

    @a.setter
    def a(self, value):
        if type(value) is dict and set(value) == set(['const', 'Ct', 'TI']):
            self._a = value
        else:
            raise ValueError("Invalid value given for a: {}".format(value))

    @property
    def b(self):
        """
        Linear coefficient used in calculation of wake-added turbulence.

        Args:
            b (float): Linear coefficient used in calculation of wake-added
                turbulence.

        Returns:
            float: Linear coefficient used in calculation of wake-added
                turbulence.
        """
        return self._b

    @b.setter
    def b(self, value):
        if type(value) is dict and set(value) == set(['const', 'Ct', 'TI']):
            self._b = value
        else:
            raise ValueError("Invalid value given for b: {}".format(value))

    @property
    def c(self):
        """
        Near-wake coefficient used in calculation of wake-added turbulence.

        Args:
            c (float): Near-wake coefficient used in calculation of wake-added
                turbulence.

        Returns:
            float: Near-wake coefficient used in calculation of wake-added
                turbulence.
        """
        return self._c

    @c.setter
    def c(self, value):
        if type(value) is dict and set(value) == set(['const', 'Ct', 'TI']):
            self._c = value
        else:
            raise ValueError("Invalid value given for c: {}".format(value))