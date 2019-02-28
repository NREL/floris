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

    def __init__(self, parameter_dictionary):
        self.model_string = None

    def __str__(self):
        return self.model_string

class Jimenez(WakeDeflection):
    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "jimenez"
        model_dictionary = parameter_dictionary[self.model_string]
        self.ad = float(model_dictionary["ad"])
        self.kd = float(model_dictionary["kd"])
        self.bd = float(model_dictionary["bd"])

    def function(self, x_locations, y_locations, turbine, coord, flow_field):
        # this function defines the angle at which the wake deflects in relation to the yaw of the turbine
        # this is coded as defined in the Jimenez et. al. paper

        # angle of deflection
        xi_init = (1. / 2.) * np.cos(turbine.yaw_angle) * \
            np.sin(turbine.yaw_angle) * turbine.Ct
        # xi = xi_init / (1 + 2 * self.kd * x_locations / turbine.rotor_diameter)**2
        
        x_locations = x_locations - coord.x1

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

class Gauss(WakeDeflection):
    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "gauss_deflection"
        model_dictionary = parameter_dictionary[self.model_string]
        self.ka = float(model_dictionary["ka"])
        self.kb = float(model_dictionary["kb"])
        self.ad = float(model_dictionary["ad"])
        self.bd = float(model_dictionary["bd"])
        self.alpha = float(model_dictionary["alpha"])
        self.beta = float(model_dictionary["beta"])

    def function(self, x_locations, y_locations, turbine, coord, flow_field):
        # =======================================================================================================
        wind_speed    = flow_field.wind_speed             # free-stream velocity (m/s)
        TI_0    = flow_field.turbulence_intensity   # turbulence intensity (%/100)
        veer    = flow_field.wind_veer                   # veer (rad), should be deg in the input file and then converted internally
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

        # U_local = flow_field.wind_speed # just a placeholder for now, should be initialized with the flow_field
        U_local = flow_field.u_initial

        # initial velocity deficits
        uR          = U_local*Ct*np.cos(tilt)*np.cos(yaw)/(2.*(1-np.sqrt(1-(Ct*np.cos(tilt)*np.cos(yaw)))))
        u0          = U_local*np.sqrt(1-Ct)

        # length of near wake
        x0      = D*(np.cos(yaw)*(1+np.sqrt(1-Ct*np.cos(yaw)))) / (np.sqrt(2)*(4*alpha*TI + 2*beta*(1-np.sqrt(1-Ct)))) + coord.x1

        # wake expansion parameters
        ky      = ka*TI + kb 
        kz      = ka*TI + kb

        C0      = 1 - u0/wind_speed
        M0      = C0*(2-C0) 
        E0      = C0**2 - 3*np.exp(1./12.)*C0 + 3*np.exp(1./3.)

        # initial Gaussian wake expansion
        sigma_z0    = D*0.5*np.sqrt( uR/(U_local + u0) )
        sigma_y0    = sigma_z0*np.cos(yaw)*np.cos(veer)

        yR = y_locations - coord.x2
        xR = yR*np.tan(yaw) + coord.x1

        # yaw parameters (skew angle and distance from centerline)  
        theta_c0    = 2*((0.3*yaw)/np.cos(yaw))*(1-np.sqrt(1-Ct*np.cos(yaw)))    # skew angle   
        delta0      = np.tan(theta_c0)*(x0-coord.x1)                            # initial wake deflection

        # deflection in the near wake
        delta_near_wake = ((x_locations-xR)/(x0-xR))*delta0 + ( ad + bd*(x_locations-coord.x1) )                               
        delta_near_wake[x_locations < xR] = 0.0
        delta_near_wake[x_locations > x0] = 0.0

        # deflection in the far wake
        sigma_y = ky*( x_locations - x0 ) + sigma_y0
        sigma_z = kz*( x_locations - x0 ) + sigma_z0
        sigma_y[x_locations < x0] = sigma_y0[x_locations < x0]
        sigma_z[x_locations < x0] = sigma_z0[x_locations < x0]

        ln_deltaNum = (1.6+np.sqrt(M0))*(1.6*np.sqrt(sigma_y*sigma_z/(sigma_y0*sigma_z0)) - np.sqrt(M0))
        ln_deltaDen = (1.6-np.sqrt(M0))*(1.6*np.sqrt(sigma_y*sigma_z/(sigma_y0*sigma_z0)) + np.sqrt(M0))
        delta_far_wake = delta0 + (theta_c0*E0/5.2)*np.sqrt(sigma_y0*sigma_z0/(ky*kz*M0))*np.log(ln_deltaNum/ln_deltaDen) + ( ad + bd*(x_locations-coord.x1) )  
        delta_far_wake[x_locations <= x0] = 0.0

        deflection = delta_near_wake + delta_far_wake

        return deflection

class Curl(WakeDeflection):
    def __init__(self, parameter_dictionary):
        super().__init__(parameter_dictionary)
        self.model_string = "curl"
        
    def function(self, x_locations, y_locations, turbine, coord, flow_field):
        return np.zeros(np.shape(x_locations)) 
