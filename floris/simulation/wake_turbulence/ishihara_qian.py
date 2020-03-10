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
    Ishihara is a wake velocity subclass that contains objects related to the
    Gaussian wake model that include a near-wake correction.

    Ishihara is a subclass of
    :py:class:`floris.simulation.wake_velocity.WakeTurbulence` that is
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
                -   **d**: constant coefficient used in calculation of              wake-added turbulence.
                -   **e**: linear coefficient used in calculation of                wake-added turbulence.
                -   **f**: near-wake coefficient used in calculation of             wake-added turbulence.

    Returns:
        An instantiated Ishihara(WakeTurbulence) object.
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
        Using the Gaussian wake model, this method calculates and
        returns the wake velocity deficits, caused by the specified
        turbine, relative to the freestream velocities at the grid of
        points comprising the wind farm flow field.

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
        # # compute area overlap of wake on other turbines and update downstream
        # # turbine turbulence intensities
        # for coord_ti, turbine_ti in sorted_map:

        #     if coord_ti.x1 > turbine_coord.x1 and np.abs(
        #             turbine_coord.x2 -
        #             coord_ti.x2) < 2 * turbine.rotor_diameter:
        #         # only assess the effects of the current wake

        #         # added turbulence model
        #         ti_initial = turbine.turbulence_intensity

        #         # turbine parameters
        #         D = turbine.rotor_diameter
        #         HH = turbine.hub_height
        #         Ct = turbine.Ct

        #         local_x = x_locations - turbine_coord.x1
        #         local_y = y_locations - turbine_coord.x2
        #         local_z = z_locations - turbine_coord.x3
        #         # coordinate info
        #         r = np.sqrt(local_y**2 + (local_z)**2)

        #         def parameter_value_from_dict(pdict, Ct, ti_initial):
        #             return pdict['const'] * Ct**(pdict['Ct']) * ti_initial**(
        #                 pdict['TI'])

        #         kstar = parameter_value_from_dict(self.kstar, Ct, ti_initial)
        #         epsilon = parameter_value_from_dict(self.epsilon, Ct,
        #                                             ti_initial)

        #         d = parameter_value_from_dict(self.d, Ct, ti_initial)
        #         e = parameter_value_from_dict(self.e, Ct, ti_initial)
        #         f = parameter_value_from_dict(self.f, Ct, ti_initial)

        #         k1 = np.cos(np.pi / 2 * (r / D - 0.5))**2
        #         k1[r / D > 0.5] = 1.0

        #         k2 = np.cos(np.pi / 2 * (r / D + 0.5))**2
        #         k2[r / D > 0.5] = 0.0

        #         # Representative wake width = \sigma / D
        #         wake_width = kstar * (local_x / D) + epsilon

        #         # Added turbulence intensity = \Delta I_1 (x,y,z)
        #         delta = ti_initial * np.sin(np.pi * (HH - local_z) / HH)**2
        #         delta[local_z >= HH] = 0.0
        #         ti_calculation = 1 / (
        #             d + e * (local_x / D) + f * (1 + (local_x / D))**(-2)) * (
        #                 (k1 * np.exp(-(r - D / 2)**2 /
        #                              (2 * (wake_width * D)**2))) +
        #                 (k2 * np.exp(-(r + D / 2)**2 /
        #                              (2 * (wake_width * D)**2)))) - delta

        #         # Update turbulence intensity of downstream turbines
        #         turbine_ti.turbulence_intensity = np.sqrt(
        #             ti_calculation**2 + flow_field.turbulence_intensity**2)

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

        def parameter_value_from_dict(pdict, Ct, ti_initial):
            return pdict['const'] * Ct**(pdict['Ct']) * ti_initial**(
                pdict['TI'])

        kstar = parameter_value_from_dict(self.kstar, Ct, ti_initial)
        epsilon = parameter_value_from_dict(self.epsilon, Ct, ti_initial)

        d = parameter_value_from_dict(self.d, Ct, ti_initial)
        e = parameter_value_from_dict(self.e, Ct, ti_initial)
        f = parameter_value_from_dict(self.f, Ct, ti_initial)

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

        # Update turbulence intensity of downstream turbines
        # turbine_ti.turbulence_intensity = np.sqrt(
        #     ti_calculation**2 + flow_field.turbulence_intensity**2)
        return ti_calculation

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

        Args:
            d (float): Constant coefficient used in calculation of wake-added
                turbulence.

        Returns:
            float: Constant coefficient used in calculation of wake-added
                turbulence.
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

        Args:
            e (float): Linear coefficient used in calculation of wake-added
                turbulence.

        Returns:
            float: Linear coefficient used in calculation of wake-added
                turbulence.
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

        Args:
            f (float): Near-wake coefficient used in calculation of wake-added
                turbulence.

        Returns:
            float: Near-wake coefficient used in calculation of wake-added
                turbulence.
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