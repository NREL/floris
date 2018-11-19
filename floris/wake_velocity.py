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

class WakeVelocity():

    def __init__(self, type_string, parameter_dictionary):
        super().__init__()
        self.type_string = type_string

        type_map = {
            "jensen": self._jensen,
            "floris": self._floris,
            "gauss": self._gauss,
            "curl": self._curl
        }
        self.function = type_map[type_string]

        # store the parameter dictionaries
        self.jensen = parameter_dictionary["jensen"]
        self.floris = parameter_dictionary["floris"]
        self.gauss = parameter_dictionary["gauss"]
        self.curl = parameter_dictionary["curl"]
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

        # curl parameters
        self.grid_resolution = np.asarray(self.curl["grid_resolution"])
        self.vortex_strength = float(self.curl["vortex_strength"])
        self.initial_deficit = float(self.curl["initial_deficit"])
        self.dissipation = float(self.curl["dissipation"])
        self.veer_linear = float(self.curl["veer_linear"])

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

    def _curl(self, x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, wake, flowfield):

        # this code has been adapted from Martinez et. al.

        vortex_strength = self.vortex_strength
        intial_deficit = self.initial_deficit
        dissipation = self.dissipation
        veer_linear = self.veer_linear

        x = np.linspace(np.min(x_locations),np.max(x_locations),flowfield.grid_resolution.x)
        y = np.linspace(np.min(y_locations),np.max(y_locations),flowfield.grid_resolution.y)

        idx = np.min(np.where(x >= turbine_coord.x+0*turbine.rotor_radius))
        uw = np.zeros((flowfield.grid_resolution.x,flowfield.grid_resolution.y,flowfield.grid_resolution.z))
        
        y1 = y_locations[idx,:,:] - turbine_coord.y
        z1 = z_locations[idx,:,:] - turbine.hub_height #turbine_coord.z
        r1 = np.sqrt(y1**2 + z1**2)

        uw_initial = -(flowfield.wind_speed*intial_deficit*turbine.aI)#*np.ones((75,75)) # 1.4, 1.75

        uw[idx,:,:] = gaussian_filter(uw_initial * (r1 <= turbine.rotor_diameter/2), sigma=1) # 1, 2
        uw[idx,0,:] = 0.0
        uw[idx,:,0] = 0.0
        uw[idx,-1,:] = 0.0
        uw[idx,:,-1] = 0.0
        
        uw = -uw

        D = turbine.rotor_diameter 
        Ct = turbine.Ct 
        yaw = turbine.yaw_angle
        HH = turbine.hub_height
        Uinf = flowfield.wind_speed
        alpha = flowfield.wind_shear
        TSR = turbine.tsr 
        rho = flowfield.air_density
        aI = turbine.aI 
        U = flowfield.initial_flowfield
        V = flowfield.v
        W = flowfield.w
        tilt = turbine.tilt_angle

        X = flowfield.x 
        Y = flowfield.y 
        Z = flowfield.z

        Gamma_Yaw = vortex_strength * np.pi * D/2 * Ct * turbine.get_average_velocity() * np.sin(yaw) * np.cos(yaw)**2 # 0.35 * 
        if turbine.yaw_angle != 0.0:
            YawFlag = 1
        else:
            YawFlag = 0
        Gamma_Tilt = np.pi * D/2 * Ct * turbine.get_average_velocity() * np.sin(tilt) * np.cos(tilt)**2
        if turbine.tilt_angle != 0.0:
            TiltFlag = 1
            NuTiltFlag = 0
            if np.abs(turbine.tilt_angle) <= np.radians(2.0):
                NuTiltFlag = 1
        else:
            TiltFlag = 0
            NuTiltFlag = 1

        Gamma = Gamma_Yaw + Gamma_Tilt

        Gamma_wake_rotation = 2 * np.pi * D * (aI - aI**2) * Uinf/TSR #* 0.001 
        
        # =======================================================================
        # add curl Elliptic
        # =======================================================================
        eps = 0.2 * D

        # distribute rotation across the blade
        z_vector = np.linspace(0,D/2,100)
        #z_vector = [D/3,D/2]

        # length of each section dz
        dz = z_vector[1] - z_vector[0]

        # scale the circulation of each section dz 
        if yaw != 0 or tilt != 0:
            Gamma0 = 4 / np.pi * Gamma 
        else:
            Gamma0 = 0.0

        # loop through all the vortices from an elliptic wind distribution 
        # skip the last point because it has zero circulation
        nu = 1
        #print(np.shape(flowfield.z))
        for z in z_vector[:-1]:

            # Compute the non-dimensional circulation
            Gamma = (-4 * Gamma0 * z * dz / (D**2 * np.sqrt(1 - (2 * z/D)**2)))

            # locations of the tip vortices
            # top
            y_vortex_1 = turbine_coord.y + z*TiltFlag
            z_vortex_1 = HH + z*YawFlag

            # bottom
            y_vortex_2 = turbine_coord.y - z*TiltFlag 
            z_vortex_2 = HH - z*YawFlag 

            # vortex velocities
            # top
            v1, w1 = self._vortex(flowfield.y[idx, :, :] - y_vortex_1, flowfield.z[idx, :, :] - z_vortex_1, flowfield.x[idx, :, :]-turbine_coord.x, -Gamma, eps, nu, Uinf)
            # bottom
            v2, w2 = self._vortex(flowfield.y[idx, :, :] - y_vortex_2, flowfield.z[idx, :, :] - z_vortex_2, flowfield.x[idx, :, :]-turbine_coord.x, Gamma, eps, nu, Uinf)

            # add ground effects
            v3, w3 = self._vortex(flowfield.y[idx, :, :] - y_vortex_1, flowfield.z[idx, :, :] + z_vortex_1, flowfield.x[idx, :, :]-turbine_coord.x, Gamma, eps, nu, Uinf)
            v4, w4 = self._vortex(flowfield.y[idx, :, :] - y_vortex_2, flowfield.z[idx, :, :] + z_vortex_2, flowfield.x[idx, :, :]-turbine_coord.x, -Gamma, eps, nu, Uinf)

            V[idx, :, :] += v1 + v2 + v3 + v4
            W[idx, :, :] += w1 + w2 + w3 + w4

        # add wake rotation
        v5, w5 = self._vortex(flowfield.y[idx, :, :] - turbine_coord.y, flowfield.z[idx, :, :] - turbine.hub_height, flowfield.x[idx, :, :]-turbine_coord.x, Gamma_wake_rotation, 0.2*D, nu, Uinf)
        v6, w6 = self._vortex(flowfield.y[idx, :, :] - turbine_coord.y, flowfield.z[idx, :, :] + turbine.hub_height, flowfield.x[idx, :, :]-turbine_coord.x, -Gamma_wake_rotation, 0.2*D, nu, Uinf)
        V[idx, :, :] += v5 + v6
        W[idx, :, :] += w5 + w6
        
        lmda = 15
        kappa = 0.41
        lm = kappa * z / (1 + kappa*z/lmda)
        dudz_initial = np.gradient(U, z, axis=2)
        nu = lm**2 * np.abs(dudz_initial[0,:,:])
        
        for i in range(idx,len(x)-1):
            V[i+1, :, :] = V[idx, :, :] * eps**2 / (4 * nu * (flowfield.x[i,:,:]-turbine_coord.x)/Uinf + eps**2) # do you really mean plus?
            W[i+1, :, :] = W[idx, :, :] * eps**2 / (4 * nu * (flowfield.x[i,:,:]-turbine_coord.x)/Uinf + eps**2)
        
        z = np.linspace(np.min(z_locations),np.max(z_locations),flowfield.grid_resolution.z)
        z_min = HH
        b_veer = veer_linear
        m_veer = -b_veer/z_min

        v_veer = m_veer*z + b_veer

        for i in range(len(z)-1):
            V[:, :, i] = V[:, :, i] + v_veer[i]

        # ===========================================================================================
        # SOLVE CURL
        # ===========================================================================================
        # figure out where to start in x (based on where the turbine is located)

        yz,zy = np.meshgrid(z,y)

        idx = np.min(np.where(x >= turbine_coord.x+0*turbine.rotor_radius))
        idx1 = np.min(np.where(y >= (turbine_coord.y)))
        idx2 = np.min(np.where(z >= turbine.hub_height))

        dudz_initial = np.gradient(U, axis=2)/np.gradient(z_locations,axis=2)

        for i in range(idx+1,len(x)):
        
            # compute the change in x 
            dx = x[i] - x[i-1]
            dy = y[1] - y[0]
            dz = z[1] - z[0]

            dudy = np.gradient(uw[i-1,:,:], axis=0)/np.gradient(y_locations[i-1,:,:],axis=0)
            dudz = np.gradient(uw[i-1,:,:], axis=1)/np.gradient(z_locations[i-1,:,:],axis=1)

            gradU = np.gradient(np.gradient(uw[i-1,:,:],axis=0),axis=0)/np.gradient(y_locations[i-1,:,:],axis=0)**2 \
                  + np.gradient(np.gradient(uw[i-1,:,:],axis=1),axis=1)/np.gradient(z_locations[i-1,:,:],axis=1)**2

            lm = kappa * z / (1 + kappa*z/lmda) 
            nu = lm**2 * np.abs(dudz_initial[i-1,:,:])
            strX = 'x = ' + str(x[i])

            # solve the marching problem for u, v, and w 
            if NuTiltFlag == 1:
                uw[i,:,:] = uw[i-1,:,:] + (dx / (U[i-1,:,:])) * (-V[i-1,:,:]*dudy - W[i-1,:,:]*dudz + dissipation*D*nu*gradU)
            else:
                uw[i,:,:] = uw[i-1,:,:] + (dx / (U[i-1,:,:])) * (-V[i-1,:,:]*dudy - W[i-1,:,:]*dudz + dissipation*D*nu*gradU)
            # enforce boundary conditions
            uw[i,:,0] = np.zeros(len(y))
            uw[i,0,:] = np.zeros(len(z))
        
        #print(str(turbine_coord) + ":")
        #print("\tCp -", turbine.Cp)
        #print("\tCt -", turbine.Ct)
        #print("\tpower -", turbine.power)
        #print("\tai -", turbine.aI)
        #print("\taverage velocity -", turbine.get_average_velocity())
        #print("\tyaw angle -", turbine.yaw_angle*180/np.pi)
        #print("\tNutiltFlag is: ",NuTiltFlag)
        #plt.figure()
        #plt.plot(x,8+uw[:,idx1,idx2])
        #plt.show()
        #plt.grid()
        #plt.xlabel('nu_T')
        #plt.ylabel('z (m)')
        uw[x_locations < turbine_coord.x] = 0.0

        return uw, V, W

    def _vortex(self,x,y,z,Gamma,eps,nu,U):

        # compute the vortex velocity 
        # eps = the core size of the tip vortex 
        nu = 2 # 2 for NREL 5MW; 1.55 for SUMR13
        v =  (Gamma / (2*np.pi)) * (y / (x**2 + y**2)) * (1 - np.exp( -(x**2 + y**2)/eps**2 ))#*np.exp(-2*nu*z/U*0.003388) # 0.003388 for 5 MW,   0.001435 for 13 MW
        w = -(Gamma / (2*np.pi)) * (x / (x**2 + y**2)) * (1 - np.exp( -(x**2 + y**2)/eps**2 ))#*np.exp(-2*nu*z/U*0.003388)

        return v,w