# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from ....utilities import cosd, sind, tand
from ....utilities import setup_logger
from ..base_velocity_deficit import VelocityDeficit
import numpy as np


class Gauss(VelocityDeficit):
    """
    Gauss is a wake velocity subclass that contains objects related to the
    Gaussian wake model.

    Gauss is a subclass of
    :py:class:`floris.simulation.wake_velocity.WakeVelocity` that is
    used to compute the wake velocity deficit based on the Gaussian
    wake model with self-similarity. The Gaussian wake model includes a
    Gaussian wake velocity deficit profile in the y and z directions
    and includes the effects of ambient turbulence, added turbulence
    from upstream wakes, as well as wind shear and wind veer. For more
    information about the Gauss wake model theory, see:

    [1] Abkar, M. and Porte-Agel, F. "Influence of atmospheric stability on
    wind-turbine wakes: A large-eddy simulation study." *Physics of
    Fluids*, 2015.

    [2] Bastankhah, M. and Porte-Agel, F. "A new analytical model for
    wind-turbine wakes." *Renewable Energy*, 2014.

    [3] Bastankhah, M. and Porte-Agel, F. "Experimental and theoretical
    study of wind turbine wakes in yawed conditions." *J. Fluid
    Mechanics*, 2016.

    [4] Niayifar, A. and Porte-Agel, F. "Analytical modeling of wind farms:
    A new approach for power prediction." *Energies*, 2016.

    [5] Dilip, D. and Porte-Agel, F. "Wind turbine wake mitigation through
    blade pitch offset." *Energies*, 2017.

    [6] Blondel, F. and Cathelain, M. "An alternative form of the
    super-Gaussian wind turbine wake model." *Wind Energy Science Disucssions*,
    2020.

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

            -   **gauss**: A dictionary containing the following
                key-value pairs:
                -   **ka**: A float that is a parameter used to
                    determine the linear relationship between the
                    turbulence intensity and the width of the Gaussian
                    wake shape.
                -   **kb**: A float that is a second parameter used to
                    determine the linear relationship between the
                    turbulence intensity and the width of the Gaussian
                    wake shape.
                -   **alpha**: A float that is a parameter that
                    determines the dependence of the downstream
                    boundary between the near wake and far wake region
                    on the turbulence intensity.
                -   **beta**: A float that is a parameter that
                    determines the dependence of the downstream
                    boundary between the near wake and far wake region
                    on the turbine's induction factor.

    Returns:
        An instantiated Gauss object.
    """

    default_parameters = {
        'ka': 0.38,
        'kb': 0.004,
        'alpha': 0.58,
        'beta': 0.077
    }

    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.logger = setup_logger(name=__name__)
        self.model_string = "gauss"
        model_dictionary = self._get_model_dict(__class__.default_parameters)

        # wake expansion parameters
        self.ka = float(model_dictionary["ka"])
        self.kb = float(model_dictionary["kb"])

        # near wake parameters
        self.alpha = float(model_dictionary["alpha"])
        self.beta = float(model_dictionary["beta"])

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
        
        # veer (degrees)
        veer = flow_field.wind_veer

        # added turbulence model
        TI = turbine.current_turbulence_intensity

        # turbine parameters
        D = turbine.rotor_diameter
        HH = turbine.hub_height
        yaw = -1 * turbine.yaw_angle  # opposite sign convention in this model
        Ct = turbine.Ct
        U_local = flow_field.u_initial

        # wake deflection
        delta = deflection_field

        # initial velocity deficits
        uR = U_local * Ct / (2.0 * (1 - np.sqrt(1 - (Ct))))
        u0 = U_local * np.sqrt(1 - Ct)

        # initial Gaussian wake expansion
        sigma_z0 = D * 0.5 * np.sqrt(uR / (U_local + u0))
        sigma_y0 = sigma_z0 * cosd(yaw) * cosd(veer)

        # quantity that determines when the far wake starts
        x0 = D * (cosd(yaw) * (1 + np.sqrt(1 - Ct))) / (np.sqrt(2) \
            * (4 * self.alpha * TI + 2 * self.beta * (1 - np.sqrt(1 - Ct)))) \
            + turbine_coord.x1

        # wake expansion parameters
        ky = self.ka * TI + self.kb
        kz = self.ka * TI + self.kb

        # compute velocity deficit
        yR = y_locations - turbine_coord.x2
        xR = yR * tand(yaw) + turbine_coord.x1

        # velocity deficit in the near wake
        sigma_y = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * \
            D * np.sqrt(Ct / 2.) + ((x_locations - xR) / (x0 - xR)) * sigma_y0
        sigma_z = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * \
            D * np.sqrt(Ct / 2.) + ((x_locations - xR) / (x0 - xR)) * sigma_z0

        sigma_y[x_locations < xR] = 0.5 * D
        sigma_z[x_locations < xR] = 0.5 * D

        a = (cosd(veer)**2) / (2 * sigma_y**2) + \
            (sind(veer)**2) / (2 * sigma_z**2)
        b = -(sind(2 * veer)) / (4 * sigma_y**2) + \
            (sind(2 * veer)) / (4 * sigma_z**2)
        c = (sind(veer)**2) / (2 * sigma_y**2) + \
            (cosd(veer)**2) / (2 * sigma_z**2)
        totGauss = np.exp(-(a * ((y_locations - turbine_coord.x2) - delta)**2 \
                - 2 * b * ((y_locations - turbine_coord.x2) - delta) \
                * ((z_locations - HH)) + c * ((z_locations - HH))**2))

        velDef = (U_local * (1 - np.sqrt(1 - ((Ct * cosd(yaw)) \
                / (8.0 * sigma_y * sigma_z / D**2)))) * totGauss)
        velDef[x_locations < xR] = 0
        velDef[x_locations > x0] = 0

        # wake expansion in the lateral (y) and the vertical (z)
        sigma_y = ky * (x_locations - x0) + sigma_y0
        sigma_z = kz * (x_locations - x0) + sigma_z0

        sigma_y[x_locations < x0] = sigma_y0[x_locations < x0]
        sigma_z[x_locations < x0] = sigma_z0[x_locations < x0]

        # velocity deficit outside the near wake
        a = (cosd(veer)**2) / (2 * sigma_y**2) + \
            (sind(veer)**2) / (2 * sigma_z**2)
        b = -(sind(2 * veer)) / (4 * sigma_y**2) + \
            (sind(2 * veer)) / (4 * sigma_z**2)
        c = (sind(veer)**2) / (2 * sigma_y**2) + \
            (cosd(veer)**2) / (2 * sigma_z**2)
        totGauss = np.exp(-(a * ((y_locations - turbine_coord.x2) - delta)**2 \
                - 2 * b * ((y_locations - turbine_coord.x2) - delta) \
                * ((z_locations - HH)) + c * ((z_locations - HH))**2))

        # compute velocities in the far wake
        velDef1 = (U_local * (1 - np.sqrt(1 - ((Ct * cosd(yaw)) \
                / (8.0 * sigma_y * sigma_z / D**2)))) * totGauss)
        velDef1[x_locations < x0] = 0

        return np.sqrt(velDef**2 + velDef1**2), np.zeros(np.shape(velDef)), \
                       np.zeros(np.shape(velDef))

    @property
    def ka(self):
        """
        Parameter used to determine the linear relationship between the 
            turbulence intensity and the width of the Gaussian wake shape.
        Args:
            ka (float, int): Gaussian wake model coefficient.
        Returns:
            float: Gaussian wake model coefficient.
        """
        return self._ka

    @ka.setter
    def ka(self, value):
        if type(value) is float:
            self._ka = value
        elif type(value) is int:
            self._ka = float(value)
        else:
            raise ValueError("Invalid value given for ka: {}".format(value))

    @property
    def kb(self):
        """
        Parameter used to determine the linear relationship between the 
            turbulence intensity and the width of the Gaussian wake shape.
        Args:
            kb (float, int): Gaussian wake model coefficient.
        Returns:
            float: Gaussian wake model coefficient.
        """
        return self._kb

    @kb.setter
    def kb(self, value):
        if type(value) is float:
            self._kb = value
        elif type(value) is int:
            self._kb = float(value)
        else:
            raise ValueError("Invalid value given for kb: {}".format(value))

    @property
    def alpha(self):
        """
        Parameter that determines the dependence of the downstream boundary
            between the near wake and far wake region on the turbulence
            intensity.
        Args:
            alpha (float, int): Gaussian wake model coefficient.
        Returns:
            float: Gaussian wake model coefficient.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if type(value) is float:
            self._alpha = value
        elif type(value) is int:
            self._alpha = float(value)
        else:
            raise ValueError("Invalid value given for alpha: {}".format(value))

    @property
    def beta(self):
        """
        Parameter that determines the dependence of the downstream boundary
            between the near wake and far wake region on the turbine's
            induction factor.
        Args:
            beta (float, int): Gaussian wake model coefficient.
        Returns:
            float: Gaussian wake model coefficient.
        """
        return self._beta

    @beta.setter
    def beta(self, value):
        if type(value) is float:
            self._beta = value
        elif type(value) is int:
            self._beta = float(value)
        else:
            raise ValueError("Invalid value given for beta: {}".format(value))