# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from .types import Vec3

class WakeVelocity():

    def __init__(self, parameter_dictionary):
        self.requires_resolution = False
        self.model_string = None
        self.model_grid_resolution = None

        # turbulence parameters
        turbulence_intensity = parameter_dictionary["turbulence_intensity"]
        self.ti_initial = float(turbulence_intensity["initial"])
        self.ti_constant = float(turbulence_intensity["constant"])
        self.ti_ai = float(turbulence_intensity["ai"])
        self.ti_downstream = float(turbulence_intensity["downstream"])

    def __str__(self):
        return self.model_string

class Jensen(WakeVelocity):
    """
    compute the velocity deficit based on the classic Jensen/Park model. see Jensen 1983
    """
    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "jensen"
        model_dictionary = parameter_dictionary[self.model_string]
        self.we = float(model_dictionary["we"])

    def function(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, wake, flow_field):
        """
        +/- 2keX is the slope of the cone boundary for the wake

        x direction is streamwise (with the wind)
        y direction is normal to the streamwise direction and parallel to the ground
        z direction is normal the streamwise direction and normal to the ground
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
        c = (turbine.rotor_diameter /
             (2 * self.we * (x_locations - turbine_coord.x1) + turbine.rotor_diameter))**2

        # filter points upstream and beyond the upper and lower bounds of the wake
        c[x_locations - turbine_coord.x1 < 0] = 0
        c[y_locations > y_upper] = 0
        c[y_locations < y_lower] = 0
        c[z_locations > z_upper] = 0
        c[z_locations < z_lower] = 0

        return 2 * turbine.aI * c * flow_field.u_initial


class Floris(WakeVelocity):
    """
    compute the velocity deficit based on wake zones, see Gebraad et. al. 2016
    """
    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "floris"
        model_dictionary = parameter_dictionary[self.model_string]
        self.me = [float(n) for n in model_dictionary["me"]]
        self.we = float(model_dictionary["we"])
        self.aU = np.radians(float(model_dictionary["aU"]))
        self.bU = np.radians(float(model_dictionary["bU"]))
        self.mU = [float(n) for n in model_dictionary["mU"]]

    def function(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, wake, flow_field):
        
        mu = self.mU / np.cos((self.aU + self.bU * np.degrees(turbine.yaw_angle)) * np.pi / 180.0)

        # distance from wake centerline
        rY = abs(y_locations - (turbine_coord.x2 + deflection_field))
        # rZ = abs(z_locations - (turbine.hub_height))
        dx = x_locations - turbine_coord.x1

        # wake zone diameters
        nearwake = (turbine.rotor_radius + self.we * self.me[0] * dx)
        farwake = (turbine.rotor_radius + self.we * self.me[1] * dx)
        mixing = (turbine.rotor_radius + self.we * self.me[2] * dx)

        # initialize the wake field
        c = np.zeros(x_locations.shape)

        # near wake zone
        mask = rY <= nearwake
        c += mask * (turbine.rotor_diameter / (turbine.rotor_diameter + 2 * self.we * mu[0] * dx))**2
        #mask = rZ <= nearwake
        #c += mask * (radius / (radius + we * mu[0] * dx))**2

        # far wake zone
        # ^ is XOR, x^y:
        #   Each bit of the output is the same as the corresponding bit in x
        #   if that bit in y is 0, and it's the complement of the bit in x
        #   if that bit in y is 1.
        # The resulting mask is all the points in far wake zone that are not
        # in the near wake zone
        mask = (rY <= farwake) ^ (rY <= nearwake)
        c += mask * (turbine.rotor_diameter / (turbine.rotor_diameter + 2 * self.we * mu[1] * dx))**2
        #mask = (rZ <= farwake) ^ (rZ <= nearwake)
        #c += mask * (radius / (radius + we * mu[1] * dx))**2

        # mixing zone
        # | is OR, x|y:
        #   Each bit of the output is 0 if the corresponding bit of x AND
        #   of y is 0, otherwise it's 1.
        # The resulting mask is all the points in mixing zone that are not
        # in the far wake zone and not in  near wake zone
        mask = (rY <= mixing) ^ ((rY <= farwake) | (rY <= nearwake))
        c += mask * (turbine.rotor_diameter / (turbine.rotor_diameter + 2 * self.we * mu[2] * dx))**2
        #mask = (rZ <= mixing) ^ ((rZ <= farwake) | (rZ <= nearwake))
        #c += mask * (radius / (radius + we * mu[2] * dx))**2

        # filter points upstream
        c[x_locations - turbine_coord.x1 < 0] = 0

        return 2 * turbine.aI * c * flow_field.wind_speed


class Gauss(WakeVelocity):
    """
    analytical wake model based on self-similarity and Gaussian wake model
    based on Porte-Agel et. al. papers from 2015-2017
    """
    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "gauss"
        model_dictionary = parameter_dictionary[self.model_string]
        self.ka = float(model_dictionary["ka"])        # wake expansion parameter
        self.kb = float(model_dictionary["kb"])        # wake expansion parameter
        self.alpha = float(model_dictionary["alpha"])  # near wake parameter
        self.beta = float(model_dictionary["beta"])    # near wake parameter

    def function(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, wake, flow_field):
        """
        user defined model parameters:

        
        internal model parameters:


        inputs:
            x_locations: [float] - x-component of the turbine locations

            y_locations: [float] - y-component of the turbine locations

            z_locations: [float] - z-component of the turbine locations

            turbine: Turbine - the turbine where the wake effects are calculated

            turbine_coord: Vec3 - the location of the turbine above

            deflection_field: [float]

            wake: Wake - 

            flow_field: FlowField -

        """

        # veer (radians)
        # TODO: should be deg in the input file and then converted internally
        veer = flow_field.wind_veer
        
        # TODO: placeholder for now, should be computed with turbine
        TI = flow_field.turbulence_intensity

        # turbine parameters
        D = turbine.rotor_diameter
        HH = turbine.hub_height
        yaw = -1 * turbine.yaw_angle  # opposite sign convention in this model
        tilt = turbine.tilt_angle
        Ct = turbine.Ct
        U_local = flow_field.u_initial

        # wake deflection
        delta = deflection_field

        # initial velocity deficits
        uR = U_local * Ct * np.cos(tilt) * np.cos(yaw) / (2.0 * (1 - np.sqrt(1 - (Ct * np.cos(tilt) * np.cos(yaw)))))
        u0 = U_local * np.sqrt(1 - Ct)

        # initial Gaussian wake expansion
        sigma_z0 = D * 0.5 * np.sqrt(uR / (U_local + u0))
        sigma_y0 = sigma_z0 * (np.cos((yaw))) * (np.cos(veer))

        # quantity that determines when the far wake starts
        x0 = D * (np.cos(yaw) * (1 + np.sqrt(1 - Ct * np.cos(yaw)))) / (np.sqrt(2) * (4 * self.alpha * TI + 2 * self.beta * (1 - np.sqrt(1 - Ct)))) + turbine_coord.x1

        # wake expansion parameters
        ky = self.ka * TI + self.kb
        kz = self.ka * TI + self.kb

        # compute velocity deficit
        yR = y_locations - turbine_coord.x2
        xR = yR * np.tan(yaw) + turbine_coord.x1

        # velocity deficit in the near wake
        sigma_y = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * D * np.sqrt(Ct / 2.) + ((x_locations - xR) / (x0 - xR)) * sigma_y0
        sigma_z = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * D * np.sqrt(Ct / 2.) + ((x_locations - xR) / (x0 - xR)) * sigma_z0

        sigma_y[x_locations < xR] = 0.5 * D
        sigma_z[x_locations < xR] = 0.5 * D

        a = (np.cos(veer)**2) / (2 * sigma_y**2) + (np.sin(veer)**2) / (2 * sigma_z**2)
        b = -(np.sin(2 * veer)) / (4 * sigma_y**2) + (np.sin(2 * veer)) / (4 * sigma_z**2)
        c = (np.sin(veer)**2) / (2 * sigma_y**2) + (np.cos(veer)**2) / (2 * sigma_z**2)
        totGauss = np.exp(-(a * ((y_locations - turbine_coord.x2) - delta)**2 - 2 * b * ((y_locations - turbine_coord.x2) - delta) * ((z_locations - HH)) + c * ((z_locations - HH))**2))

        velDef = (U_local * (1 - np.sqrt(1 - ((Ct * np.cos(yaw)) / (8.0 * sigma_y * sigma_z / D**2)))) * totGauss)
        velDef[x_locations < xR] = 0
        velDef[x_locations > x0] = 0

        # wake expansion in the lateral (y) and the vertical (z)
        sigma_y = ky * (x_locations - x0) + sigma_y0
        sigma_z = kz * (x_locations - x0) + sigma_z0

        sigma_y[x_locations < x0] = sigma_y0[x_locations < x0]
        sigma_z[x_locations < x0] = sigma_z0[x_locations < x0]

        # velocity deficit outside the near wake
        a = (np.cos(veer)**2) / (2 * sigma_y**2) + (np.sin(veer)**2) / (2 * sigma_z**2)
        b = -(np.sin(2 * veer)) / (4 * sigma_y**2) + (np.sin(2 * veer)) / (4 * sigma_z**2)
        c = (np.sin(veer)**2) / (2 * sigma_y**2) + (np.cos(veer)**2) / (2 * sigma_z**2)
        totGauss = np.exp(-(a * ((y_locations - turbine_coord.x2) - delta)**2 - 2 * b * ((y_locations - turbine_coord.x2) - delta) * ((z_locations - HH)) + c * ((z_locations - HH))**2))

        # compute velocities in the far wake
        velDef1 = (U_local * (1 - np.sqrt(1 - ((Ct * np.cos(yaw)) / (8.0 * sigma_y * sigma_z / D**2)))) * totGauss)
        velDef1[x_locations < x0] = 0

        return np.sqrt(velDef**2 + velDef1**2)


class Curl(WakeVelocity):
    """
    this code has been adapted from Martinez et. al.
    """
    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "curl"
        model_dictionary = parameter_dictionary[self.model_string]
        self.model_grid_resolution = Vec3(model_dictionary["model_grid_resolution"])
        self.vortex_strength = float(model_dictionary["vortex_strength"])
        self.initial_deficit = float(model_dictionary["initial_deficit"])
        self.dissipation = float(model_dictionary["dissipation"])
        self.veer_linear = float(model_dictionary["veer_linear"])
        self.requires_resolution = True

    def function(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, wake, flow_field):
        """
        """

        # parameters available for tuning to match high-fidelity data
        vortex_strength = self.vortex_strength  # scaling parameter that adjusts strength of vortexes
        intial_deficit = self.initial_deficit   # parameter for defining initial velocity deficity in the flow field at a turbine
        dissipation = self.dissipation          # scaling parameter that adjusts the amount of dissipation of the vortexes
        veer_linear = self.veer_linear          # parameter that defines the wind velocity of veer at 0 meters height

        # setup x and y grid information
        x = np.linspace(np.min(x_locations), np.max(x_locations), self.model_grid_resolution.x1)
        y = np.linspace(np.min(y_locations), np.max(y_locations), self.model_grid_resolution.x2)

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
        uw_initial = -1 * (flow_field.wind_speed * intial_deficit * turbine.aI)
        uw[idx, :, :] = gaussian_filter(uw_initial * (r1 <= turbine.rotor_diameter / 2), sigma=1)
        # enforce the boundary conditions
        uw[idx, 0, :] = 0.0
        uw[idx, :, 0] = 0.0
        uw[idx, -1, :] = 0.0
        uw[idx, :, -1] = 0.0

        # TODO: explain?
        uw = -1 * uw

        # parameters to simplify the code
        # diameter of the turbine rotor from the input file
        D = turbine.rotor_diameter
        Ct = turbine.Ct                                 # thrust coefficient of the turbine
        yaw = turbine.yaw_angle                         # yaw angle of the turbine
        HH = turbine.hub_height                         # hub height of the turbine
        # the free-stream velocity of the flow field
        Uinf = flow_field.wind_speed
        # the tip-speed ratior of the turbine
        TSR = turbine.tsr
        # the axial induction factor of the turbine
        aI = turbine.aI
        # initial velocities in the stream-wise, span-wise, and vertical direction
        U, V, W = flow_field.initialize_flow_field()
        # the tilt angle of the rotor of the turbine
        tilt = turbine.tilt_angle

        # calculate the curled wake effects due to the yaw and tilt of the turbine
        Gamma_Yaw = vortex_strength * np.pi * D / 2 * Ct * \
            turbine.average_velocity * np.sin(yaw) * np.cos(yaw)**2
        if turbine.yaw_angle != 0.0:
            YawFlag = 1
        else:
            YawFlag = 0
        Gamma_Tilt = np.pi * D / 2 * Ct * turbine.average_velocity * np.sin(tilt) * \
            np.cos(tilt)**2
        if turbine.tilt_angle != 0.0:
            TiltFlag = 1
        else:
            TiltFlag = 0

        Gamma = Gamma_Yaw + Gamma_Tilt

        # calculate the curled wake effects due to the rotation of the turbine rotor
        Gamma_wake_rotation = 2 * np.pi * D * (aI - aI**2) * Uinf / TSR

        # =======================================================================
        # add curl Elliptic
        # =======================================================================
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
            z_vortex_1 = HH + z * YawFlag

            # bottom
            y_vortex_2 = turbine_coord.x2 - z * TiltFlag
            z_vortex_2 = HH - z * YawFlag

            # vortex velocities
            # top
            v1, w1 = self._vortex(flow_field.y[idx, :, :] - y_vortex_1, flow_field.z[idx, :, :] -
                                  z_vortex_1, flow_field.x[idx, :, :] - turbine_coord.x1, -Gamma, eps, Uinf)
            # bottom
            v2, w2 = self._vortex(flow_field.y[idx, :, :] - y_vortex_2, flow_field.z[idx, :, :] -
                                  z_vortex_2, flow_field.x[idx, :, :] - turbine_coord.x1, Gamma, eps, Uinf)

            # add ground effects
            v3, w3 = self._vortex(flow_field.y[idx, :, :] - y_vortex_1, flow_field.z[idx, :, :] +
                                  z_vortex_1, flow_field.x[idx, :, :] - turbine_coord.x1, Gamma, eps, Uinf)
            v4, w4 = self._vortex(flow_field.y[idx, :, :] - y_vortex_2, flow_field.z[idx, :, :] +
                                  z_vortex_2, flow_field.x[idx, :, :] - turbine_coord.x1, -Gamma, eps, Uinf)

            V[idx, :, :] += v1 + v2 + v3 + v4
            W[idx, :, :] += w1 + w2 + w3 + w4

        # add wake rotation
        v5, w5 = self._vortex(flow_field.y[idx, :, :] - turbine_coord.x2, flow_field.z[idx, :, :] -
                              turbine.hub_height, flow_field.x[idx, :, :] - turbine_coord.x, Gamma_wake_rotation, 0.2 * D, Uinf)
        v6, w6 = self._vortex(flow_field.y[idx, :, :] - turbine_coord.x2, flow_field.z[idx, :, :] +
                              turbine.hub_height, flow_field.x[idx, :, :] - turbine_coord.x, -Gamma_wake_rotation, 0.2 * D, Uinf)
        V[idx, :, :] += v5 + v6
        W[idx, :, :] += w5 + w6

        # decay the vortices as they move downstream
        lmda = 15
        kappa = 0.41
        lm = kappa * z / (1 + kappa * z / lmda)
        dudz_initial = np.gradient(U, z, axis=2)
        nu = lm**2 * np.abs(dudz_initial[0, :, :])

        for i in range(idx, len(x) - 1):
            V[i + 1, :, :] = V[idx, :, :] * eps**2 / \
                (4 * nu * (flow_field.x[i, :, :] -
                           turbine_coord.x1) / Uinf + eps**2)
            W[i + 1, :, :] = W[idx, :, :] * eps**2 / \
                (4 * nu * (flow_field.x[i, :, :] -
                           turbine_coord.x1) / Uinf + eps**2)

        # simple implementation of linear veer, added to the V component of the flow field
        z = np.linspace(
            np.min(z_locations),
            np.max(z_locations),
            self.model_grid_resolution.x3
        )
        z_min = HH
        b_veer = veer_linear
        m_veer = -b_veer / z_min

        v_veer = m_veer * z + b_veer

        for i in range(len(z) - 1):
            V[:, :, i] = V[:, :, i] + v_veer[i]

        # ===========================================================================================
        # SOLVE CURL
        # ===========================================================================================
        dudz_initial = np.gradient(U, axis=2) / \
            np.gradient(z_locations, axis=2)

        for i in range(idx + 1, len(x)):

            # compute the change in x
            dx = x[i] - x[i - 1]

            dudy = np.gradient(uw[i - 1, :, :], axis=0) / \
                np.gradient(y_locations[i - 1, :, :], axis=0)
            dudz = np.gradient(uw[i - 1, :, :], axis=1) / \
                np.gradient(z_locations[i - 1, :, :], axis=1)

            gradU = np.gradient(np.gradient(uw[i - 1, :, :], axis=0), axis=0) / np.gradient(y_locations[i - 1, :, :], axis=0)**2 \
                + np.gradient(np.gradient(uw[i - 1, :, :], axis=1), axis=1) / \
                np.gradient(z_locations[i - 1, :, :], axis=1)**2

            lm = kappa * z / (1 + kappa * z / lmda)
            nu = lm**2 * np.abs(dudz_initial[i - 1, :, :])

            # solve the marching problem for u, v, and w
            # uw[i,:,:] = uw[i-1,:,:] + (dx / (U[i-1,:,:])) * (-V[i-1,:,:]*dudy - W[i-1,:,:]*dudz + dissipation*D*(np.arctan(1/375*dx - np.pi/1.5)/(np.pi/2)+1)/2*nu*gradU)
            uw[i, :, :] = uw[i - 1, :, :] + (dx / (U[i - 1, :, :])) * (-V[i - 1, :, :]
                                                                       * dudy - W[i - 1, :, :] * dudz + dissipation * D * nu * gradU)
            # enforce boundary conditions
            uw[i, :, 0] = np.zeros(len(y))
            uw[i, 0, :] = np.zeros(len(z))

        uw[x_locations < turbine_coord.x1] = 0.0

        return uw, V, W

    def _vortex(self, x, y, z, Gamma, eps, U):
        # compute the vortex velocity
        # *np.exp(-2*nu*z/U*0.003388) # 0.003388 for 5 MW,   0.001435 for 13 MW
        v = (Gamma / (2 * np.pi)) * (y / (x**2 + y**2)) * \
            (1 - np.exp(-(x**2 + y**2) / eps**2))
        w = -(Gamma / (2 * np.pi)) * (x / (x**2 + y**2)) * (1 -
                                                            np.exp(-(x**2 + y**2) / eps**2))  # *np.exp(-2*nu*z/U*0.003388)

        return v, w
