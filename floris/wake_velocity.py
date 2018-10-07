# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np

class WakeVelocity():

    def __init__(self, type_string, parameter_dictionary):

        self.type_string = type_string

        type_map = {
            "jensen": self._jensen,
            "floris": self._floris,
            "gauss": self._gauss
        }
        self.function = type_map[type_string]

        # store the parameter dictionaries
        self.jensen = parameter_dictionary["jensen"]
        self.floris = parameter_dictionary["floris"]
        self.gauss = parameter_dictionary["gauss"]
        self.turbulence_intensity = parameter_dictionary["turbulence_intensity"]

        # parse the parameters
        # jensen parameters
        self.we = float(self.jensen["we"])
        
        # floris parameters
        self.me = self.floris["me"]
        self.aU = np.radians(float(self.floris["aU"]))
        self.bU = np.radians(float(self.floris["bU"]))
        self.mU = self.floris["mU"]

        # gauss parameters
        self.ka = float(self.gauss["ka"])
        self.kb = float(self.gauss["kb"])
        self.alpha = float(self.gauss["alpha"])
        self.beta = float(self.gauss["beta"])

        # turbulence parameters
        self.ti_initial = float(self.turbulence_intensity["initial"])
        self.ti_constant = float(self.turbulence_intensity["constant"])
        self.ti_ai = float(self.turbulence_intensity["ai"])
        self.ti_downstream = float(self.turbulence_intensity["downstream"])

    def _activation_function(self, x, loc):
        sharpness = 10
        return (1 + np.tanh(sharpness * (x - loc))) / 2.

    def _jensen(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, wake, flowfield):
        """
            x direction is streamwise (with the wind)
            y direction is normal to the streamwise direction and parallel to the ground
            z direction is normal the streamwise direction and normal to the ground
        """
        # compute the velocity deficit based on the classic Jensen/Park model. see Jensen 1983
        # +/- 2keX is the slope of the cone boundary for the wake

        # define the boundary of the wake model ... y = mx + b
        m = self.we
        x = x_locations - turbine_coord.x
        b = turbine.rotor_radius

        boundary_line = m * x + b

        y_upper = boundary_line + turbine_coord.y + deflection_field
        y_lower = -1 * boundary_line + turbine_coord.y + deflection_field

        z_upper = boundary_line + turbine.hub_height
        z_lower = -1 * boundary_line + turbine.hub_height

        # calculate the wake velocity
        c = (turbine.rotor_diameter / 
             (2 * self.we * (x_locations - turbine_coord.x) + turbine.rotor_diameter))**2

        # filter points upstream and beyond the upper and lower bounds of the wake
        c[x_locations - turbine_coord.x < 0] = 0
        c[y_locations > y_upper] = 0
        c[y_locations < y_lower] = 0

        c[z_locations > z_upper] = 0
        c[z_locations < z_lower] = 0

        return 2 * turbine.aI * c * flowfield.initial_flowfield

    def _floris(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, wake, flowfield):
        # compute the velocity deficit based on wake zones, see Gebraad et. al. 2016

        # wake parameters
        me = self.me 
        aU = self.aU
        bU = self.bU
        radius = turbine.rotor_radius
        diameter = turbine.rotor_diameter
        mu = self.mU / np.cos( (aU + bU * np.degrees(turbine.yaw_angle))*np.pi/180. )
        we = self.we

        wind_speed = flowfield.wind_speed

        # distance from wake centerline
        rY = abs(y_locations - (turbine_coord.y + deflection_field))
        rZ = abs(z_locations - (turbine.hub_height))
        dx = x_locations - turbine_coord.x

        # wake zone diameters
        nearwake = (radius + we * me[0] * dx)
        farwake = (radius + we * me[1] * dx)
        mixing = (radius + we * me[2] * dx)

        # initialize the wake field
        c = np.zeros(x_locations.shape)

        # near wake zone
        mask = rY <= nearwake
        c += mask * (diameter / (diameter + 2 * we * mu[0] * dx))**2
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
        c += mask * (diameter / (diameter + 2 * we * mu[1] * dx))**2
        #mask = (rZ <= farwake) ^ (rZ <= nearwake)
        #c += mask * (radius / (radius + we * mu[1] * dx))**2

        # mixing zone
        # | is OR, x|y:
        #   Each bit of the output is 0 if the corresponding bit of x AND
        #   of y is 0, otherwise it's 1.
        # The resulting mask is all the points in mixing zone that are not
        # in the far wake zone and not in  near wake zone
        mask = (rY <= mixing) ^ ((rY <= farwake) | (rY <= nearwake))
        c += mask * (diameter / (diameter + 2 * we * mu[2] * dx))**2
        #mask = (rZ <= mixing) ^ ((rZ <= farwake) | (rZ <= nearwake))
        #c += mask * (radius / (radius + we * mu[2] * dx))**2

        # filter points upstream
        c[x_locations - turbine_coord.x < 0] = 0

        return wind_speed * 2 * turbine.aI * c
    
    def _gauss(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, wake, flowfield):

        # analytical wake model based on self-similarity and Gaussian wake model
        # based on Porte-Agel et. al. papers from 2015-2017

        # =======================================================================================================
        wind_speed    = flowfield.wind_speed            # free-stream velocity (m/s)
        TI_0    = flowfield.turbulence_intensity        # turbulence intensity (%/100)
        veer    = flowfield.wind_veer                   # veer (rad), should be deg in the input file and then converted internally
        TI      = flowfield.turbulence_intensity        # just a placeholder for now, should be computed with turbine
        
        # hard-coded model input data (goes in input file)
        ka      = self.ka           # wake expansion parameter
        kb      = self.kb           # wake expansion parameter
        alpha   = self.alpha        # near wake parameter
        beta    = self.beta         # near wake parameter
        ad      = 0.0               # natural lateral deflection parameter
        bd      = 0.0               # natural lateral deflection parameter
        aT      = 0.0               # natural vertical deflection parameter
        bT      = 0.0               # natural vertical deflection parameter

        # =======================================================================================================
                
        # turbine parameters
        D           = turbine.rotor_diameter
        HH          = turbine.hub_height
        yaw         = -turbine.yaw_angle         # opposite sign convention in this model
        tilt        = turbine.tilt_angle
        Ct          = turbine.Ct
        U_local     = flowfield.initial_flowfield

        # wake deflection
        delta = deflection_field
        
        # initial velocity deficits
        uR      = U_local*Ct*np.cos(tilt)*np.cos(yaw)/(2.*(1-np.sqrt(1-(Ct*np.cos(tilt)*np.cos(yaw)))))
        u0      = U_local*np.sqrt(1-Ct)
        
        # initial Gaussian wake expansion
        sigma_z0    = D*0.5*np.sqrt( uR/(U_local + u0) )
        sigma_y0    = sigma_z0*(np.cos((yaw)))*(np.cos(veer))

        # quantity that determines when the far wake starts
        x0      = D*(np.cos(yaw)*(1+np.sqrt(1-Ct*np.cos(yaw)))) / (np.sqrt(2)*(4*alpha*TI + 2*beta*(1-np.sqrt(1-Ct)))) + turbine_coord.x

        # wake expansion parameters
        ky      = ka*TI + kb 
        kz      = ka*TI + kb

        # initial wake velocity deficit (quantities based on Porte-Agel/Bastankah 2016 JFM)
        C0       = 1 - u0/wind_speed
        M0       = C0*(2-C0)    
        E0       = C0**2 - 3*np.exp(1./12.)*C0 + 3*np.exp(1./3.)
        
        ## COMPUTE VELOCITY DEFICIT
        yR      = y_locations - turbine_coord.y
        xR      = yR*np.tan(yaw) + turbine_coord.x
                 
        # velocity deficit in the near wake
        sigma_y = (((x0-xR)-(x_locations-xR))/(x0-xR))*0.501*D*np.sqrt(Ct/2.) + ((x_locations-xR)/(x0-xR))*sigma_y0
        sigma_z = (((x0-xR)-(x_locations-xR))/(x0-xR))*0.501*D*np.sqrt(Ct/2.) + ((x_locations-xR)/(x0-xR))*sigma_z0

        sigma_y[x_locations < xR] = 0.5*D
        sigma_z[x_locations < xR] = 0.5*D

        a = (np.cos(veer)**2)/(2*sigma_y**2) + (np.sin(veer)**2)/(2*sigma_z**2)
        b = -(np.sin(2*veer))/(4*sigma_y**2) + (np.sin(2*veer))/(4*sigma_z**2)
        c = (np.sin(veer)**2)/(2*sigma_y**2) + (np.cos(veer)**2)/(2*sigma_z**2)
        totGauss = np.exp( -( a*((y_locations-turbine_coord.y)-delta)**2 - 2*b*((y_locations-turbine_coord.y)-delta)*((z_locations-HH)) + c*((z_locations-HH))**2 ) )

        velDef = (U_local*(1-np.sqrt(1-((Ct*np.cos(yaw))/(8.0*sigma_y*sigma_z/D**2)) ) )*totGauss)
        velDef[x_locations < xR] = 0     
        velDef[x_locations > x0] = 0
              
        # wake expansion in the lateral (y) and the vertical (z) 
        sigma_y = ky*( x_locations - x0 ) + sigma_y0
        sigma_z = kz*( x_locations - x0 ) + sigma_z0

        sigma_y[x_locations < x0] = sigma_y0[x_locations < x0]
        sigma_z[x_locations < x0] = sigma_z0[x_locations < x0]

        # velocity deficit outside the near wake
        a = (np.cos(veer)**2)/(2*sigma_y**2) + (np.sin(veer)**2)/(2*sigma_z**2)
        b = -(np.sin(2*veer))/(4*sigma_y**2) + (np.sin(2*veer))/(4*sigma_z**2)
        c = (np.sin(veer)**2)/(2*sigma_y**2) + (np.cos(veer)**2)/(2*sigma_z**2)
        totGauss = np.exp( -( a*((y_locations-turbine_coord.y)-delta)**2 - 2*b*((y_locations-turbine_coord.y)-delta)*((z_locations-HH)) + c*((z_locations-HH))**2 ) )
        
        # compute velocities in the far wake
        velDef1 = (U_local*(1-np.sqrt(1-((Ct*np.cos(yaw))/(8.0*sigma_y*sigma_z/D**2)) ) )*totGauss)
        velDef1[x_locations < x0] = 0
                  
        return np.sqrt(velDef**2 + velDef1**2)
