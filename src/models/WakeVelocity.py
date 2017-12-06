"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import numpy as np
from BaseObject import BaseObject


class WakeVelocity(BaseObject):

    def __init__(self, typeString):
        super().__init__()
        self.typeString = typeString

        typeMap = {
            "jensen": self._jensen,
            "floris": self._floris
        }
        self.function = typeMap[typeString]

        # wake expansion coefficient
        self.we = .05

        # floris parameters

    
    def _activation_function(self, x, loc):
        sharpness = 10
        return (1 + np.tanh(sharpness * (x - loc))) / 2.

    def _jensen(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field):
        """
            x direction is streamwise (with the wind)
            y direction is normal to the streamwise direction and parallel to the ground
            z direction is normal the streamwise direction and normal to the ground=
        """
        # compute the velocity deficit based on the classic Jensen/Park model. see Jensen 1983
        # +/- 2keX is the slope of the cone boundary for the wake

        # define the boundary of the wake model ... y = mx + b
        m = 2 * self.we
        x = x_locations - turbine_coord.x
        b = turbine.rotorRadius
        boundary_line = m * x + b
        y_upper = boundary_line + turbine_coord.y
        y_lower = -1 * boundary_line + turbine_coord.y

        # calculate the wake velocity
        c = (turbine.rotorRadius / 
                (self.we * (x_locations - turbine_coord.x) + turbine.rotorRadius))**2

        # filter points upstream and beyond the upper and lower bounds of the wake
        c[x_locations - turbine_coord.x < 0] = 0
        c[y_locations > y_upper + deflection_field] = 0
        c[y_locations < y_lower + deflection_field] = 0

        return 2 * turbine.aI * c

    def _floris(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field):
        # compute the velocity deficit based on wake zones, see Gebraad et. al. 2016

        # wake parameters
        me = np.array([-0.5, 0.3, 1.0]) # inputData['me']
        aU = np.radians(12.0) #inputData['aU']
        bU = np.radians(1.3) #inputData['bU']
        radius = turbine.rotorRadius
        yaw = turbine.yawAngle
        mu = [0.5, 1., 5.5] / np.cos(aU + bU * yaw)

        # distance from wake centerline
        rY = abs(y_locations - (turbine_coord.y + deflection_field))
        dx = x_locations - turbine_coord.x

        # wake zone diameters
        d = np.ndarray((3,) + x_locations.shape)
        d[0] = (radius + self.we * me[0] * dx)
        d[1] = (radius + self.we * me[1] * dx)
        d[2] = (radius + self.we * me[2] * dx)

        # initialize the wake field
        c = np.zeros(x_locations.shape)

        # near wake zone
        mask = rY <= d[0]
        c += mask * (radius / (radius + self.we * mu[0] * dx))**2

        # far wake zone
        # ^ is XOR, x^y:
        #   Each bit of the output is the same as the corresponding bit in x
        #   if that bit in y is 0, and it's the complement of the bit in x
        #   if that bit in y is 1.
        # The resulting mask is all the points in far wake zone that are not
        # in the near wake zone
        mask = (rY <= d[1]) ^ (rY <= d[0])
        c += mask * (radius / (radius + self.we * mu[1] * dx))**2

        # mixing zone
        # | is OR, x|y:
        #   Each bit of the output is 0 if the corresponding bit of x AND
        #   of y is 0, otherwise it's 1.
        # The resulting mask is all the points in mixing zone that are not
        # in the far wake zone and not in  near wake zone
        mask = (rY <= d[2]) ^ ((rY <= d[1]) | (rY <= d[0]))
        c += mask * (radius / (radius + self.we * mu[2] * dx))**2

        # filter points upstream
        c[x_locations - turbine_coord.x < 0] = 0

        return 2 * turbine.aI * c

    def print_countvals(self, array): 
        unique, counts = np.unique(array, return_counts=True)
        print(dict(zip(unique, counts)))
    
    def _gauss(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field):

        # TODO: currently the deflection model, flow field quantities, model parameters is computed within this function

        # analytical wake model based on self-similarity and Gaussian wake model
        # based on Porte-Agel et. al. papers from 2015-2017

        # =======================================================================================================
        # WILL NOT BE IN THE FINAL VERSION
        # hard-coded physical input data (goes in input file)
        wind_speed    = 7.0                       # free-stream velocity (m/s)
        TI_0          = 0.1                       # turbulence intensity (%/100)
        veer          = 0.0                       # veer (rad), should be deg in the input file and then converted internally
        TI            = TI_0   # just a placeholder for now, should be computed with turbine
        
        # hard-coded model input data (goes in input file)
        ka      = 0.38371                   # wake expansion parameter
        kb      = 0.004                     # wake expansion parameter
        alpha   = 0.58                      # near wake parameter
        beta    = 0.07                      # near wake parameter
        ad      = 0.0                       # natural lateral deflection parameter
        bd      = 0.0                       # natural lateral deflection parameter
        aT      = 0.0                       # natural vertical deflection parameter
        bT      = 0.0                       # natural vertical deflection parameter

        # =======================================================================================================
                
        # turbine parameters
        D       = turbine.rotorDiameter
        HH      = turbine.hubHeight
        yaw     = -turbine.yawAngle         # opposite sign convention in this model
        tilt    = turbine.tilt
        Ct      = turbine.Ct
        U_local = turbine.windSpeed # just a placeholder for now, should be initialized with the flowfield

        # wake deflection
        delta = deflection_field
        
        # initial velocity deficits
        uR      = U_local*Ct/(2.*(1-np.sqrt(1-(Ct))))
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
        
        # yaw parameters (skew angle and distance from centerline)
        theta_c0    = ((0.3*yaw)/np.cos(yaw))*(1-np.sqrt(1-Ct*np.cos(yaw))) # skew angle  
        theta_z0    = ((0.3*tilt)/np.cos(tilt))*(1-np.sqrt(1-Ct*np.cos(tilt)))  # skew angle  
        delta0      = np.tan(theta_c0)*(x0-turbine_coord.x)
        delta_z0    = np.tan(theta_z0)*(x0-turbine_coord.x)
        
        ## COMPUTE VELOCITY DEFICIT
        yR      = y_locations - turbine_coord.y
        xR      = yR*np.tan(yaw) + turbine_coord.x
                 
        # velocity deficit in the near wake
        sigma_y = (((x0-xR)-(x_locations-xR))/(x0-xR))*0.501*D*np.sqrt(Ct/2.) + ((x_locations-xR)/(x0-xR))*sigma_y0
        sigma_z = (((x0-xR)-(x_locations-xR))/(x0-xR))*0.501*D*np.sqrt(Ct/2.) + ((x_locations-xR)/(x0-xR))*sigma_z0
        delta = ((x_locations-xR)/(x0-xR))*delta0 + ( ad + bd*(x_locations-turbine_coord.x) )   
        deltaZ = ((x_locations-xR)/(x0-xR))*delta_z0 + ( aT + bT*(x_locations-turbine_coord.x) )
        a = (np.cos(veer)**2)/(2*sigma_y**2) + (np.sin(veer)**2)/(2*sigma_z**2)
        b = -(np.sin(2*veer))/(4*sigma_y**2) + (np.sin(2*veer))/(4*sigma_z**2)
        c = (np.sin(veer)**2)/(2*sigma_y**2) + (np.cos(veer)**2)/(2*sigma_z**2)
        totGauss = np.exp( -( a*((y_locations-turbine_coord.y)-delta)**2 - 2*b*((y_locations-turbine_coord.y)-delta)*((z_locations-HH)-deltaZ) + c*((z_locations-HH)-deltaZ)**2 ) )

        velDef = (Uloc*(1-np.sqrt(1-((Ct*np.cos(yaw))/(8.0*sigma_y*sigma_z/D**2)) ) )*totGauss)
        velDef[x_locations < xR] = 0     
        velDef[x_locations > x0] = 0
              
        # wake expansion in the lateral (y) and the vertical (z) 
        sigma_y = ky*( x_locations - x0 ) + sigma_y0
        sigma_z = kz*( x_locations - x0 ) + sigma_z0

        # velocity deficit outside the near wake
        ln_deltaNum = (1.6+np.sqrt(M0))*(1.6*np.sqrt(sigma_y*sigma_z/(sigma_y0*sigma_z0)) - np.sqrt(M0))
        ln_deltaDen = (1.6-np.sqrt(M0))*(1.6*np.sqrt(sigma_y*sigma_z/(sigma_y0*sigma_z0)) + np.sqrt(M0))
        delta = delta0 + (theta_c0*E0/5.2)*np.sqrt(sigma_y0*sigma_z0/(ky*kz*M0))*np.log(ln_deltaNum/ln_deltaDen) + ( ad + bd*(x_locations-turbine_coord.x) )  
        deltaZ = delta_z0 + (theta_z0*E0/5.2)*np.sqrt(sigma_y0*sigma_z0/(ky*kz*M0))*np.log(ln_deltaNum/ln_deltaDen) + ( aT + bT*(x_locations-turbine_coord.x) )
        a = (np.cos(veer)**2)/(2*sigma_y**2) + (np.sin(veer)**2)/(2*sigma_z**2)
        b = -(np.sin(2*veer))/(4*sigma_y**2) + (np.sin(2))/(4*sigma_z**2)
        c = (np.sin(veer)**2)/(2*sigma_y**2) + (np.cos(veer)**2)/(2*sigma_z**2)
        totGauss = np.exp( -( a*((y_locations-turbine_coord.y)-delta)**2 - 2*b*((y_locations-turbine_coord.y)-delta)*((z_locations-HH)-deltaZ) + c*((z_locations-HH)-deltaZ)**2 ) )
        
        # compute velocities in the far wake
        velDef1 = (Uloc*(1-np.sqrt(1-((Ct*np.cos(yaw))/(8.0*sigma_y*sigma_z/D**2)) ) )*totGauss)
        velDef1[x_locations < x0] = 0
                  
        return np.sqrt(velDef**2 + velDef1**2)



    # def _gauss_thrust_angle(self):

