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


class IshiharaQian(WakeTurbulence):
    """
    IshiharaQian is a wake velocity subclass that is used to compute the wake 
    velocity deficit based on the Gaussian wake model with self-similarity and 
    a near wake correction. The IshiharaQian wake model includes a Gaussian 
    wake velocity deficit profile in the spanwise and vertical directions and 
    includes the effects of ambient turbulence, added turbulence from upstream 
    wakes, as well as wind shear and wind veer. For more info, see 
    :cite:`iqt-qian2018new`.
    
    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: iqt-
    """
    default_parameters = {
        "kstar": {
            "const": 0.11,
            "Ct": 1.07,
            "TI": 0.2
        },
        "epsilon": {
            "const": 0.23,
            "Ct": -0.25,
            "TI": 0.17
        },
        "d": {
            "const": 2.3,
            "Ct": 1.2,
            "TI": 0.0
        },
        "e": {
            "const": 1.0,
            "Ct": 0.0,
            "TI": 0.1
        },
        "f": {
            "const": 0.7,
            "Ct": -3.2,
            "TI": -0.45
        }
    }

    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        All model parameters combine a constant coefficient, the thrust
        coefficient of the turbine, and the local turbulence intensity.
        Paremeter values are calculated with 
        :py:meth:`~.IshiharaQian.parameter_value_from_dict` as:

        .. code-block:: python3 

            value = pdict["const"] * Ct ** pdict["Ct"] * TI ** pdict["TI"]

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                -   **kstar** (*dict*): The parameters related to the linear
                    relationship between the turbulence intensity and the width
                    of the Gaussian wake shape.
                -   **epsilon** (*dict*): The second parameter used to
                    determine the linear relationship between the turbulence
                    intensity and the width of the Gaussian wake shape.
                -   **d** (*dict*): Constant coefficient used in calculation of
                    wake-added turbulence.
                -   **e** (*dict*): Linear coefficient used in calculation of
                    wake-added turbulence.
                -   **f** (*dict*): Near-wake coefficient used in calculation
                    of wake-added turbulence.
        """
        super().__init__(parameter_dictionary)
        self.logger = setup_logger(name=__name__)
        self.model_string = "ishihara_qian"
        model_dictionary = self._get_model_dict(__class__.default_parameters)

        # wake model parameter
        self.kstar = model_dictionary["kstar"]
        self.epsilon = model_dictionary["epsilon"]
        self.d = model_dictionary["d"]
        self.e = model_dictionary["e"]
        self.f = model_dictionary["f"]

    def function(self, ambient_TI, coord_ti, turbine_coord, turbine):
        # function(self, x_locations, y_locations, z_locations, turbine,
        #  turbine_coord, flow_field, turb_u_wake, sorted_map):
        """
        Calculates wake-added turbulence as a function of
        external conditions and wind turbine operation. This function is
        accessible through the :py:class:`~.wake.Wake` class as the
        :py:meth:`~.Wake.turbulence_function` method.

        Args:
            ambient_TI (float): TI of the background flow field.
            coord_ti (:py:class:`~.utilities.Vec3`): Coordinate where TI 
                is to be calculated (e.g. downstream wind turbines).
            turbine_coord (:py:class:`~.utilities.Vec3`): Coordinate of 
                the wind turbine adding turbulence to the flow.
            turbine (:py:class:`~.turbine.Turbine`): Wind turbine 
                adding turbulence to the flow.

        Returns:
            float: Wake-added turbulence from the current wind turbine
            (**turbine**) at location specified by (**coord_ti**).
        """
        # added turbulence model
        ti_initial = ambient_TI

        # turbine parameters
        D = turbine.rotor_diameter
        HH = turbine.hub_height
        Ct = turbine.Ct

        local_x = coord_ti.x1 - turbine_coord.x1
        local_y = coord_ti.x2 - turbine_coord.x2
        local_z = coord_ti.x3 - turbine_coord.x3
        # coordinate info
        r = np.sqrt(local_y**2 + (local_z)**2)

        kstar = self.parameter_value_from_dict(self.kstar, Ct, ti_initial)
        epsilon = self.parameter_value_from_dict(self.epsilon, Ct, ti_initial)

        d = self.parameter_value_from_dict(self.d, Ct, ti_initial)
        e = self.parameter_value_from_dict(self.e, Ct, ti_initial)
        f = self.parameter_value_from_dict(self.f, Ct, ti_initial)

        k1 = np.cos(np.pi / 2 * (r / D - 0.5))**2
        k1[r / D > 0.5] = 1.0

        k2 = np.cos(np.pi / 2 * (r / D + 0.5))**2
        k2[r / D > 0.5] = 0.0

        # Representative wake width = \sigma / D
        wake_width = kstar * (local_x / D) + epsilon

        # Added turbulence intensity = \Delta I_1 (x,y,z)
        delta = ti_initial * np.sin(np.pi * (HH - local_z) / HH)**2
        delta[local_z >= HH] = 0.0
        ti_calculation = 1 / (d + e * (local_x / D) + f *
                              (1 + (local_x / D))**(-2)) * (
                                  (k1 * np.exp(-(r - D / 2)**2 /
                                               (2 * (wake_width * D)**2))) +
                                  (k2 * np.exp(-(r + D / 2)**2 /
                                               (2 *
                                                (wake_width * D)**2)))) - delta

        return ti_calculation

    def parameter_value_from_dict(pdict, Ct, ti_initial):
        """
        Calculates model parameters using current conditions and
        model dictionaries.

        Args:
            pdict (dict): Wake turbulence parameters.
            Ct (float): Thrust coefficient of the current turbine.
            ti_initial (float): Turbulence intensity.

        Returns:
            float: Current value of model parameter.
        """
        return pdict['const'] * Ct**(pdict['Ct']) * ti_initial**(
            pdict['TI'])

    @property
    def kstar(self):
        """
        Parameter that is used to determine the linear relationship between the
        turbulence intensity and the width of the Gaussian wake shape.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            kstar (dict): Factor for relationship between the turbulence
                intensity and the width of the Gaussian wake shape with the 
                following key-value pairs:

                - **const** (*float*): The constant coefficient.
                - **Ct** (*float*): The thrust coefficient exponent.
                - **TI** (*float*): The turbulence intensity exponent.

        Returns:
            dict: Factor for relationship between the turbulence intensity and
            the width of the Gaussian wake shape.

        Raises:
            ValueError: Invalid value.
        """
        return self._kstar

    @kstar.setter
    def kstar(self, value):
        if not (
            type(value) is dict and set(value) == set(['const', 'Ct', 'TI'])
        ):
            err_msg = ('Invalid value type given for kstar: {}, expected ' + \
                       'dict with keys ["const", "Ct", "TI"]').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._kstar = value
        if value != __class__.default_parameters['kstar']:
            self.logger.info(
                ('Current value of kstar, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['kstar'])
                )

    @property
    def epsilon(self):
        """
        Parameter that is used to determine the linear relationship between the
        turbulence intensity and the width of the Gaussian wake shape.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            epsilon (dict): Factor for relationship between the turbulence
                intensity and the width of the Gaussian wake shape with the 
                following key-value pairs:

                - **const** (*float*): The constant coefficient.
                - **Ct** (*float*): The thrust coefficient exponent.
                - **TI** (*float*): The turbulence intensity exponent.

        Returns:
            dict: Factor for relationship between the turbulence intensity and
            the width of the Gaussian wake shape.

        Raises:
            ValueError: Invalid value.
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if not (
            type(value) is dict and set(value) == set(['const', 'Ct', 'TI'])
        ):
            err_msg = ('Invalid value type given for epsilon: {}, expected ' + \
                       'dict with keys ["const", "Ct", "TI"]').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._epsilon = value
        if value != __class__.default_parameters['epsilon']:
            self.logger.info(
                ('Current value of epsilon, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['epsilon'])
                )

    @property
    def d(self):
        """
        Constant coefficient used in calculation of wake-added turbulence.
        
        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            d (dict): Constant coefficient used in calculation of wake-added
                turbulence with the following key-value pairs:

                - **const** (*float*): The constant coefficient.
                - **Ct** (*float*): The thrust coefficient exponent.
                - **TI** (*float*): The turbulence intensity exponent.

        Returns:
            dict: Constant coefficient used in calculation of wake-added
            turbulence.

        Raises:
            ValueError: Invalid value.
        """
        return self._d

    @d.setter
    def d(self, value):
        if not (
            type(value) is dict and set(value) == set(['const', 'Ct', 'TI'])
        ):
            err_msg = ('Invalid value type given for d: {}, expected ' + \
                       'dict with keys ["const", "Ct", "TI"]').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._d = value
        if value != __class__.default_parameters['d']:
            self.logger.info(
                ('Current value of d, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['d'])
                )

    @property
    def e(self):
        """
        Linear coefficient used in calculation of wake-added turbulence.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            e (dict): Linear coefficient used in calculation of wake-added
                turbulence with the following key-value pairs:

                - **const** (*float*): The constant coefficient.
                - **Ct** (*float*): The thrust coefficient exponent.
                - **TI** (*float*): The turbulence intensity exponent.

        Returns:
            dict: Linear coefficient used in calculation of wake-added
            turbulence.

        Raises:
            ValueError: Invalid value.
        """
        return self._e

    @e.setter
    def e(self, value):
        if not (
            type(value) is dict and set(value) == set(['const', 'Ct', 'TI'])
        ):
            err_msg = ('Invalid value type given for e: {}, expected ' + \
                       'dict with keys ["const", "Ct", "TI"]').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._e = value
        if value != __class__.default_parameters['e']:
            self.logger.info(
                ('Current value of e, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['e'])
                )

    @property
    def f(self):
        """
        Near-wake coefficient used in calculation of wake-added turbulence.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            f (dict): Near-wake coefficient used in calculation of wake-added
                turbulence with the following key-value pairs:

                - **const** (*float*): The constant coefficient.
                - **Ct** (*float*): The thrust coefficient exponent.
                - **TI** (*float*): The turbulence intensity exponent.

        Returns:
            dict: Near-wake coefficient used in calculation of wake-added
            turbulence.

        Raises:
            ValueError: Invalid value.
        """
        return self._f

    @f.setter
    def f(self, value):
        if not (
            type(value) is dict and set(value) == set(['const', 'Ct', 'TI'])
        ):
            err_msg = ('Invalid value type given for f: {}, expected ' + \
                       'dict with keys ["const", "Ct", "TI"]').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._f = value
        if value != __class__.default_parameters['f']:
            self.logger.info(
                ('Current value of f, {0}, is not equal to tuned ' +
                'value of {1}.').format(
                    value, __class__.default_parameters['f'])
                )