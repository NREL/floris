# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from ...utilities import Vec3
from ...utilities import sind, setup_logger
from .base_velocity_deficit import VelocityDeficit
import numpy as np
from scipy.ndimage.filters import gaussian_filter


class Curl(VelocityDeficit):
    """
    The Curl model class computes the wake velocity deficit based on the curled
    wake model developed in 
    :cite:`cvm-martinez2019aerodynamics`. The curled wake 
    model includes the change in the shape of the wake profile under yawed 
    conditions due to vortices that are shed from the rotor plane of a yawed 
    turbine. For more information about the curled wake model theory, see 
    :cite:`cvm-martinez2019aerodynamics`. For more 
    information about the impact of the curled wake behavior on wake steering, 
    see :cite:`cvm-fleming2018simulation`.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: cvm-
    """

    default_parameters = {
        "model_grid_resolution": [
            250,
            100,
            75
        ],
        "initial_deficit": 2.0,
        "dissipation": 0.06,
        "veer_linear": 0.0,
        "initial": 0.1,
        "constant": 0.73,
        "ai": 0.8,
        "downstream": -0.275
    }

    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                -   **model_grid_resolution** (*list*): A list of three floats  
                    that define the flow field grid resolution in the x, y, and
                    z directions used for the curl wake model calculations. The
                    grid resolution is specified as the number of grid points
                    in the flow field domain in the x, y, and z directions.
                -   **initial_deficit** (*float*): Parameter that, along with
                    the freestream velocity and the turbine's induction factor,
                    is used to determine the initial wake velocity deficit
                    immediately downstream of the rotor.
                -   **dissipation** (*float*): Parameter that is a scaling 
                    parameter that determines the amount of dissipation 
                    of the vortices with downstream distance.
                -   **veer_linear** (*float*): Parameter that describes the
                    amount of linear wind veer. This parameter defines the
                    linear change in the V velocity between the ground and hub
                    height, and therefore determines the slope of the change in
                    the V velocity with height.
                -   **initial** (*float*): Parameter that is the initial
                    ambient turbulence intensity, expressed as a decimal
                    fraction.
                -   **constant** (*float*): Parameter that is the constant
                    used to scale the wake-added turbulence intensity.
                -   **ai** (*float*): Parameter that is the axial induction
                    factor exponent used in in the calculation of wake-added
                    turbulence.
                -   **downstream** (*float*): Parameter that is the exponent
                    applied to the distance downstream of an upstream turbine
                    normalized by the rotor diameter used in the calculation of
                    wake-added turbulence.
        """
        super().__init__(parameter_dictionary)
        self.logger = setup_logger(name=__name__)
        self.model_string = "curl"
        model_dictionary = self._get_model_dict(__class__.default_parameters)
        self.model_grid_resolution = Vec3(
            model_dictionary["model_grid_resolution"])
        self.initial_deficit = float(model_dictionary["initial_deficit"])
        self.dissipation = float(model_dictionary["dissipation"])
        self.veer_linear = float(model_dictionary["veer_linear"])
        self.ti_initial = float(model_dictionary["initial"])
        self.ti_constant = float(model_dictionary["constant"])
        self.ti_ai = float(model_dictionary["ai"])
        self.ti_downstream = float(model_dictionary["downstream"])
        self.requires_resolution = True

    def function(self, x_locations, y_locations, z_locations, turbine,
                 turbine_coord, deflection_field, flow_field):
        """
        Using the Curl wake model, this method calculates and returns 
        the wake velocity deficits, caused by the specified turbine, 
        relative to the freestream velocities at the grid of points 
        comprising the wind farm flow field.

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
            turbine_coord (:py:obj:`floris.utilities.Vec3`): Object containing
                the coordinate of the turbine creating the wake (m).
            deflection_field (np.array): An array of floats that contains the 
                amount of wake deflection in meters in the y direction at each
                grid point of the flow field.
            flow_field (:py:class:`floris.simulation.flow_field`): Object
                containing the flow field information for the wind farm.

        Returns:
            np.array, np.array, np.array:
                Three arrays of floats that contain the wake velocity
                deficit in m/s created by the turbine relative to the freestream
                velocities for the U, V, and W components, aligned with the
                x, y, and z directions, respectively. The three arrays contain
                the velocity deficits at each grid point in the flow field.
        """
        # parameters available for tuning to match high-fidelity data
        # parameter for defining initial velocity deficit in the
        # flow field at a turbine
        initial_deficit = self.initial_deficit
        # scaling parameter that adjusts the amount of dissipation
        # of the vortexes
        dissipation = self.dissipation
        # parameter that defines the wind velocity of veer at 0 meters height
        veer_linear = self.veer_linear

        # setup x and y grid information
        x = np.linspace(np.min(x_locations), np.max(
            x_locations), int(self.model_grid_resolution.x1))
        y = np.linspace(np.min(y_locations), np.max(
            y_locations), int(self.model_grid_resolution.x2))

        # find the x-grid location closest to the current turbine
        idx = np.min(np.where(x >= turbine_coord.x1))

        # initialize the flow field
        uw = np.zeros(
            (
                int(self.model_grid_resolution.x1),
                int(self.model_grid_resolution.x2),
                int(self.model_grid_resolution.x3)
            )
        )

        # determine values to create a rotor mask for velocities
        y1 = y_locations[idx, :, :] - turbine_coord.x2
        z1 = z_locations[idx, :, :] - turbine.hub_height
        r1 = np.sqrt(y1**2 + z1**2)

        # add initial velocity deficit at the rotor to the flow field
        uw_initial = -1 * (flow_field.wind_map.grid_wind_speed * \
                           initial_deficit * turbine.aI)
        uw[idx, :, :] = gaussian_filter(
            uw_initial[idx,:,:] * (r1 <= turbine.rotor_diameter / 2), sigma=1)

        # enforce the boundary conditions
        uw[idx,  0, :] = 0.0
        uw[idx, :,  0] = 0.0
        uw[idx, -1, :] = 0.0
        uw[idx, :, -1] = 0.0

        # TODO: explain?
        uw = -1 * uw

        # parameters to simplify the code
        # diameter of the turbine rotor from the input file
        D = turbine.rotor_diameter
        Ct = turbine.Ct             # thrust coefficient of the turbine
        yaw = turbine.yaw_angle     # yaw angle of the turbine
        HH = turbine.hub_height     # hub height of the turbine
        # the free-stream velocity of the flow field
        Uinf = flow_field.wind_map.grid_wind_speed[idx,:,:]
        # the tip-speed ratio of the turbine
        TSR = turbine.tsr
        # the axial induction factor of the turbine
        aI = turbine.aI
        # initial velocities in the stream-wise, span-wise, and
        # vertical direction
        U, V, W = flow_field.u, flow_field.v, flow_field.w
        # the tilt angle of the rotor of the turbine
        tilt = turbine.tilt_angle

        # calculate the curled wake effects due to the yaw and tilt
        # of the turbine
        Gamma_Yaw = flow_field.air_density * np.pi * D / 8 * Ct * \
            turbine.average_velocity * sind(yaw)
        if turbine.yaw_angle != 0.0:
            YawFlag = 1
        else:
            YawFlag = 0
        Gamma_Tilt = flow_field.air_density * np.pi * D / 8 * Ct * \
            turbine.average_velocity * sind(tilt)
        if turbine.tilt_angle != 0.0:
            TiltFlag = 1
        else:
            TiltFlag = 0

        Gamma = Gamma_Yaw + Gamma_Tilt

        # calculate the curled wake effects due to the rotation
        # of the turbine rotor
        Gamma_wake_rotation = 2 * np.pi * D * (aI - aI**2) * Uinf / TSR

        # ======================================================================
        # add curl Elliptic
        # ======================================================================
        eps = 0.2 * D

        # distribute rotation across the blade
        z_vector = np.linspace(0, D / 2, 100)

        # length of each section dz
        dz = z_vector[1] - z_vector[0]

        # scale the circulation of each section dz
        if yaw != 0 or tilt != 0:
            Gamma0 = 4 / np.pi * Gamma
        else:
            Gamma0 = 0.0

        # loop through all the vortices from an elliptic wind distribution
        # skip the last point because it has zero circulation
        for z in z_vector[:-1]:

            # Compute the non-dimensional circulation
            Gamma = (-4 * Gamma0 * z * dz /
                     (D**2 * np.sqrt(1 - (2 * z / D)**2)))

            # locations of the tip vortices
            # top
            y_vortex_1 = turbine_coord.x2 + z * TiltFlag
            z_vortex_1 = HH + z*1.1 * YawFlag

            # bottom
            y_vortex_2 = turbine_coord.x2 - z * TiltFlag
            z_vortex_2 = HH - z*1.1 * YawFlag

            # vortex velocities
            # top
            v1, w1 = self._vortex(flow_field.y[idx, :, :] \
                                  - y_vortex_1, flow_field.z[idx, :, :] \
                                  - z_vortex_1, flow_field.x[idx, :, :] \
                                  - turbine_coord.x1, -Gamma, eps, Uinf)
            # bottom
            v2, w2 = self._vortex(flow_field.y[idx, :, :] \
                                  - y_vortex_2, flow_field.z[idx, :, :] \
                                  - z_vortex_2, flow_field.x[idx, :, :] \
                                  - turbine_coord.x1, Gamma, eps, Uinf)

            # add ground effects
            v3, w3 = self._vortex(flow_field.y[idx, :, :] - y_vortex_1, \
                                  flow_field.z[idx, :, :] + z_vortex_1, \
                                  flow_field.x[idx, :, :] - turbine_coord.x1, \
                                  Gamma, eps, Uinf)
            v4, w4 = self._vortex(flow_field.y[idx, :, :] - y_vortex_2, \
                                  flow_field.z[idx, :, :] + z_vortex_2, \
                                  flow_field.x[idx, :, :] - turbine_coord.x1, \
                                  -Gamma, eps, Uinf)

            V[idx, :, :] += v1 + v2 + v3 + v4
            W[idx, :, :] += w1 + w2 + w3 + w4

        # add wake rotation
        v5, w5 = self._vortex(flow_field.y[idx, :, :] - turbine_coord.x2, \
                    flow_field.z[idx, :, :] - turbine.hub_height, \
                    flow_field.x[idx, :, :] - turbine_coord.x1, \
                    Gamma_wake_rotation, 0.2 * D, Uinf) \
                    * (np.sqrt((flow_field.y[idx, :, :] - turbine_coord.x2)**2 \
                    + (flow_field.z[idx, :, :] - turbine.hub_height)**2) <= D/2)
        v6, w6 = self._vortex(flow_field.y[idx, :, :] - turbine_coord.x2, \
                    flow_field.z[idx, :, :] + turbine.hub_height, \
                    flow_field.x[idx, :, :] - turbine_coord.x1, \
                    -Gamma_wake_rotation, 0.2 * D, Uinf) \
                    * (np.sqrt((flow_field.y[idx, :, :] - turbine_coord.x2)**2 \
                    + (flow_field.z[idx, :, :] - turbine.hub_height)**2) <= D/2)
        V[idx, :, :] += v5 + v6
        W[idx, :, :] += w5 + w6

        # decay the vortices as they move downstream
        lmda = 15
        kappa = 0.41
        z_tmp = np.linspace(
            np.min(z_locations),
            np.max(z_locations),
            int(self.model_grid_resolution.x3)
        )
        lm = kappa * z_tmp / (1 + kappa * z_tmp / lmda)
        dudz_initial = np.gradient(U, z_tmp, axis=2)
        nu = lm**2 * np.abs(dudz_initial[0, :, :])

        for i in range(idx, len(x) - 1):
            V[i + 1, :, :] = V[idx, :, :] * eps**2 \
                / (4 * nu * (flow_field.x[i, :, :]
                             - turbine_coord.x1) / Uinf + eps**2)
            W[i + 1, :, :] = W[idx, :, :] * eps**2 \
                / (4 * nu * (flow_field.x[i, :, :]
                             - turbine_coord.x1) / Uinf + eps**2)
 
        # simple implementation of linear veer, added to the V component 
        # of the flow field
        z = np.linspace(
            np.min(z_locations),
            np.max(z_locations),
            int(self.model_grid_resolution.x3)
        )
        # z_min = HH
        # b_veer = veer_linear
        # m_veer = -b_veer / z_min
        # m_veer = -0.5/63
        # b_veer = 90/63

        # 0 = -.4/100 * 90 + b
        # v_veer = m_veer * z + b_veer
        # u = 0.01
        # v_veer = -.4/100 * z + .4*90/100
        # v_veer = -.4/100 * z_locations + .4*90/100

        # for i in range(len(z) - 1):
        #     V[:, :, i] = V[:, :, i] + v_veer[i]
        
        # V += v_veer

        # ======================================================================
        # SOLVE CURL
        # ======================================================================
        dudz_initial = np.gradient(U, axis=2) \
            / np.gradient(z_locations, axis=2)

        ti_initial = flow_field.wind_map.grid_turbulence_intensity[idx,:,:]

        # turbulence intensity parameters stored in floris.json
        ti_i = self.ti_initial
        ti_constant = self.ti_constant
        ti_ai = self.ti_ai
        ti_downstream = self.ti_downstream

        for i in range(idx + 1, len(x)):

            # compute the change in x
            dx = x[i] - x[i - 1]

            dudy = np.gradient(uw[i - 1, :, :], axis=0) \
                / np.gradient(y_locations[i - 1, :, :], axis=0)
            dudz = np.gradient(uw[i - 1, :, :], axis=1) \
                / np.gradient(z_locations[i - 1, :, :], axis=1)

            gradU = np.gradient(np.gradient(uw[i - 1, :, :], axis=0), axis=0) \
                / np.gradient(y_locations[i - 1, :, :], axis=0)**2 \
                + np.gradient(np.gradient(uw[i - 1, :, :], axis=1), axis=1) \
                / np.gradient(z_locations[i - 1, :, :], axis=1)**2

            lm = kappa * z / (1 + kappa * z / lmda)
            nu = lm**2 * np.abs(dudz_initial[i - 1, :, :])

            # turbulence intensity calculation based on Crespo et. al.
            ti_local = 10*ti_constant \
                * turbine.aI**ti_ai * ti_initial**ti_i \
                * ((x[i] - turbine_coord.x1) \
                / turbine.rotor_diameter)**ti_downstream

            # solve the marching problem for u, v, and w
            uw[i, :, :] = uw[i - 1, :, :] + (dx / (U[i - 1, :, :])) \
                * (-V[i - 1, :, :] * dudy - W[i - 1, :, :] * dudz \
                + dissipation * D * nu * ti_local * gradU)
            # enforce boundary conditions
            uw[i, :, 0] = np.zeros(len(y))
            uw[i, 0, :] = np.zeros(len(z))

        uw[x_locations < turbine_coord.x1] = 0.0

        return uw, V, W

    def _vortex(self, x, y, z, Gamma, eps, U):
        # compute the vortex velocity
        v = (Gamma / (2 * np.pi)) * (y / (x**2 + y**2)) \
            * (1 - np.exp(-(x**2 + y**2) / eps**2))
        w = -(Gamma / (2 * np.pi)) * (x / (x**2 + y**2)) \
            * (1 - np.exp(-(x**2 + y**2) / eps**2))

        return v, w

    @property
    def model_grid_resolution(self):
        """
        The flow field grid resolution in the x, y, and z directions. The
        grid resolution is specified as the number of grid points in the
        flow field domain in the x, y, and z directions.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (list): Value to set.

        Returns:
            list(float): Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._model_grid_resolution

    @model_grid_resolution.setter
    def model_grid_resolution(self, value):
        #TODO: add checker to make sure resolution is high enough
        # if type(value) is Vec3:
        #     self._model_grid_resolution = value
        # elif value is None:
        #     self._model_grid_resolution = None
        # else:
        #     raise ValueError('Invalid value given for model_grid_resolution: {}'.format(value))

        if type(value) is not Vec3 and value is not None:
            err_msg = ('Invalid value type given for ' + \
                'model_grid_resolution: {}, expected type Vec3.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._model_grid_resolution = value
        if value is not None:
            if value != Vec3(
                __class__.default_parameters['model_grid_resolution']):
                    self.logger.info(
                        ('Current value of model_grid_resolution, {0}, is ' + \
                         'not equal to tuned value of {1}.').format(
                           value,
                           __class__.default_parameters['model_grid_resolution']
                        )
                    )

    @property
    def initial_deficit(self):
        """
        Parameter that, along with the freestream velocity and the turbine's
        induction factor, is used to determine the initial wake velocity
        deficit immediately downstream of the rotor.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._initial_deficit

    @initial_deficit.setter
    def initial_deficit(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for ' + \
                       'initial_deficit: {}, expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._initial_deficit = value
        if value != __class__.default_parameters['initial_deficit']:
            self.logger.info(
                ('Current value of initial_deficit, {0}, is not equal to ' + \
                    'tuned value of {1}.').format(
                        value, __class__.default_parameters['initial_deficit']
                    )
                )

    @property
    def dissipation(self):
        """
        A scaling parameter that determines the amount of dissipation of
        the vortices with downstream distance.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._dissipation

    @dissipation.setter
    def dissipation(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for ' + \
                       'dissipation: {}, expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._dissipation = value
        if value != __class__.default_parameters['dissipation']:
            self.logger.info(
                ('Current value of dissipation, {0}, is not equal to ' + \
                    'tuned value of {1}.').format(
                        value, __class__.default_parameters['dissipation']
                    )
                )

    @property
    def veer_linear(self):
        """
        This parameter defines the linear change in the V velocity between the
        ground and hub height, and therefore determines the slope of the
        change in the V velocity with height.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._veer_linear

    @veer_linear.setter
    def veer_linear(self, value):
        if type(value) is not float:
            err_msg = ('Invalid value type given for ' + \
                       'veer_linear: {}, expected float.').format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._veer_linear = value
        if value != __class__.default_parameters['veer_linear']:
            self.logger.info(
                ('Current value of veer_linear, {0}, is not equal to ' + \
                    'tuned value of {1}.').format(
                        value, __class__.default_parameters['veer_linear']
                    )
                )
