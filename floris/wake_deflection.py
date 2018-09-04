# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np

class WakeDeflection():

    def __init__(self, type_string, parameter_dictionary):

        self.type_string = type_string

        type_map = {
            "jimenez": self._jimenez,
            "gauss_deflection": self._gauss_deflection
        }
        self.function = type_map.get(self.type_string, None)

        self.jimenez = parameter_dictionary["jimenez"]
        self.gauss_deflection = parameter_dictionary["gauss_deflection"]

        self.kd = float(self.jimenez["kd"])
        self.ad = float(self.jimenez["ad"])
        self.bd = float(self.jimenez["bd"])

        self.ka = float(self.gauss_deflection["ka"])
        self.kb = float(self.gauss_deflection["kb"])
        self.alpha = float(self.gauss_deflection["alpha"])
        self.beta = float(self.gauss_deflection["beta"])

    def _jimenez(self, x_locations, y_locations, turbine, coord, flowfield):
        # this function defines the angle at which the wake deflects in relation to the yaw of the turbine
        # this is coded as defined in the Jimenez et. al. paper

        # angle of deflection
        xi_init = (1. / 2.) * np.cos(turbine.yaw_angle) * \
            np.sin(turbine.yaw_angle) * turbine.Ct
        # xi = xi_init / (1 + 2 * self.kd * x_locations / turbine.rotor_diameter)**2
        
        x_locations = x_locations - coord.x

        # yaw displacement
        yYaw_init = ( xi_init
            * ( 15 * (2 * self.kd * x_locations / turbine.rotor_diameter + 1)**4. + xi_init**2. )
            / ((30 * self.kd / turbine.rotor_diameter) * (2 * self.kd * x_locations / turbine.rotor_diameter + 1)**5.)) \
            - (xi_init * turbine.rotor_diameter * (15 + xi_init**2.) / (30 * self.kd))

        # corrected yaw displacement with lateral offset
        deflection = yYaw_init + self.ad + self.bd * x_locations

        x = np.unique(x_locations)
        for i in range(len(x)):
            tmp = np.max(deflection[x_locations == x[i]])
            deflection[x_locations == x[i]] = tmp

        return deflection

    def _gauss_deflection(self, x_locations, y_locations, turbine, coord, flowfield):

        # =======================================================================================================
        wind_speed    = flowfield.wind_speed             # free-stream velocity (m/s)
        TI_0    = flowfield.turbulence_intensity   # turbulence intensity (%/100)
        veer    = flowfield.wind_veer                   # veer (rad), should be deg in the input file and then converted internally
        TI      = turbine.turbulence_intensity   # just a placeholder for now, should be computed with turbine
        
        # hard-coded model input data (goes in input file)
        ka      = self.ka                      # wake expansion parameter
        kb      = self.kb                      # wake expansion parameter
        alpha   = self.alpha                   # near wake parameter
        beta    = self.beta                    # near wake parameter
        ad      = self.ad                      # natural lateral deflection parameter
        bd      = self.bd                      # natural lateral deflection parameter

        # turbine parameters
        D           = turbine.rotor_diameter
        HH          = turbine.hub_height
        yaw         = -turbine.yaw_angle         # opposite sign convention in this model
        tilt        = turbine.tilt_angle
        Ct          = turbine.Ct

        # U_local = flowfield.wind_speed # just a placeholder for now, should be initialized with the flowfield
        U_local = flowfield.initial_flowfield

        # initial velocity deficits
        uR          = U_local*Ct*np.cos(tilt)*np.cos(yaw)/(2.*(1-np.sqrt(1-(Ct*np.cos(tilt)*np.cos(yaw)))))
        u0          = U_local*np.sqrt(1-Ct)

        # length of near wake
        x0      = D*(np.cos(yaw)*(1+np.sqrt(1-Ct*np.cos(yaw)))) / (np.sqrt(2)*(4*alpha*TI + 2*beta*(1-np.sqrt(1-Ct)))) + coord.x

        # wake expansion parameters
        ky      = ka*TI + kb 
        kz      = ka*TI + kb

        C0      = 1 - u0/wind_speed
        M0      = C0*(2-C0) 
        E0      = C0**2 - 3*np.exp(1./12.)*C0 + 3*np.exp(1./3.)

        # initial Gaussian wake expansion
        sigma_z0    = D*0.5*np.sqrt( uR/(U_local + u0) )
        sigma_y0    = sigma_z0*np.cos(yaw)*np.cos(veer)

        yR = y_locations - coord.y
        xR = yR*np.tan(yaw) + coord.x

        # yaw parameters (skew angle and distance from centerline)  
        theta_c0    = 2*((0.3*yaw)/np.cos(yaw))*(1-np.sqrt(1-Ct*np.cos(yaw)))    # skew angle   
        delta0      = np.tan(theta_c0)*(x0-coord.x)                            # initial wake deflection

        # deflection in the near wake
        delta_near_wake = ((x_locations-xR)/(x0-xR))*delta0 + ( ad + bd*(x_locations-coord.x) )                               
        delta_near_wake[x_locations < xR] = 0.0
        delta_near_wake[x_locations > x0] = 0.0

        # deflection in the far wake
        sigma_y = ky*( x_locations - x0 ) + sigma_y0
        sigma_z = kz*( x_locations - x0 ) + sigma_z0
        sigma_y[x_locations < x0] = sigma_y0[x_locations < x0]
        sigma_z[x_locations < x0] = sigma_z0[x_locations < x0]

        ln_deltaNum = (1.6+np.sqrt(M0))*(1.6*np.sqrt(sigma_y*sigma_z/(sigma_y0*sigma_z0)) - np.sqrt(M0))
        ln_deltaDen = (1.6-np.sqrt(M0))*(1.6*np.sqrt(sigma_y*sigma_z/(sigma_y0*sigma_z0)) + np.sqrt(M0))
        delta_far_wake = delta0 + (theta_c0*E0/5.2)*np.sqrt(sigma_y0*sigma_z0/(ky*kz*M0))*np.log(ln_deltaNum/ln_deltaDen) + ( ad + bd*(x_locations-coord.x) )  
        delta_far_wake[x_locations <= x0] = 0.0

        deflection = delta_near_wake + delta_far_wake

        return deflection
