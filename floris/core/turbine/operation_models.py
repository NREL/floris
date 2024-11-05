
from __future__ import annotations

import copy
import os
from abc import abstractmethod
from pathlib import Path
from typing import (
    Any,
    Dict,
    Final,
)

import numpy as np
from attrs import define, field
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import fsolve

from floris.core import BaseClass
from floris.core.rotor_velocity import (
    average_velocity,
    compute_tilt_angles_for_floating_turbines,
    rotor_velocity_air_density_correction,
    rotor_velocity_tilt_cosine_correction,
    rotor_velocity_yaw_cosine_correction,
    tum_rotor_velocity_yaw_correction,
)
from floris.type_dec import (
    NDArrayFloat,
    NDArrayObject,
)
from floris.utilities import cosd


POWER_SETPOINT_DEFAULT = 1e12
POWER_SETPOINT_DISABLED = 0.001


@define
class BaseOperationModel(BaseClass):
    """
    Base class for turbine operation models. All turbine operation models must implement static
    power(), thrust_coefficient(), and axial_induction() methods, which are called by power() and
    thrust_coefficient() through the interface in the turbine.py module.

    Args:
        BaseClass (_type_): _description_

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
    """
    @staticmethod
    @abstractmethod
    def power() -> None:
        raise NotImplementedError("BaseOperationModel.power")

    @staticmethod
    @abstractmethod
    def thrust_coefficient() -> None:
        raise NotImplementedError("BaseOperationModel.thrust_coefficient")

    @staticmethod
    @abstractmethod
    def axial_induction() -> None:
        # TODO: Consider whether we can make a generic axial_induction method
        # based purely on thrust_coefficient so that we don't need to implement
        # axial_induciton() in individual operation models.
        raise NotImplementedError("BaseOperationModel.axial_induction")

@define
class SimpleTurbine(BaseOperationModel):
    """
    Static class defining an actuator disk turbine model that is fully aligned with the flow. No
    handling for yaw or tilt angles.

    As with all turbine submodules, implements only static power() and thrust_coefficient() methods,
    which are called by power() and thrust_coefficient() on turbine.py, respectively. This class is
    not intended to be instantiated; it simply defines a library of static methods.
    """

    def power(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        # Construct power interpolant
        power_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["power"],
            fill_value=0.0,
            bounds_error=False,
        )

        # Compute the power-effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"]
        )

        # Compute power
        power = power_interpolator(rotor_effective_velocities) * 1e3 # Convert to W

        return power

    def thrust_coefficient(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        # Construct thrust coefficient interpolant
        thrust_coefficient_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["thrust_coefficient"],
            fill_value=0.0001,
            bounds_error=False,
        )

        # Compute the effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        # TODO: Do we need an air density correction here?

        thrust_coefficient = thrust_coefficient_interpolator(rotor_average_velocities)
        thrust_coefficient = np.clip(thrust_coefficient, 0.0001, 0.9999)

        return thrust_coefficient

    def axial_induction(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):

        thrust_coefficient = SimpleTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            average_method=average_method,
            cubature_weights=cubature_weights,
        )

        return (1 - np.sqrt(1 - thrust_coefficient))/2


@define
class CosineLossTurbine(BaseOperationModel):
    """
    Static class defining an actuator disk turbine model that may be misaligned with the flow.
    Nonzero tilt and yaw angles are handled via cosine relationships, with the power lost to yawing
    defined by the cosine of the yaw misalignment raised to the power of cosine_loss_exponent_yaw.
    This turbine submodel is the default, and matches the turbine model in FLORIS v3.

    As with all turbine submodules, implements only static power() and thrust_coefficient() methods,
    which are called by power() and thrust_coefficient() on turbine.py, respectively. This class is
    not intended to be instantiated; it simply defines a library of static methods.
    """

    def power(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        tilt_interp: NDArrayObject,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        correct_cp_ct_for_tilt: bool = False,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        # Construct power interpolant
        power_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["power"],
            fill_value=0.0,
            bounds_error=False,
        )

        # Compute the power-effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"]
        )

        rotor_effective_velocities = rotor_velocity_yaw_cosine_correction(
            cosine_loss_exponent_yaw=power_thrust_table["cosine_loss_exponent_yaw"],
            yaw_angles=yaw_angles,
            rotor_effective_velocities=rotor_effective_velocities,
        )

        rotor_effective_velocities = rotor_velocity_tilt_cosine_correction(
            tilt_angles=tilt_angles,
            ref_tilt=power_thrust_table["ref_tilt"],
            cosine_loss_exponent_tilt=power_thrust_table["cosine_loss_exponent_tilt"],
            tilt_interp=tilt_interp,
            correct_cp_ct_for_tilt=correct_cp_ct_for_tilt,
            rotor_effective_velocities=rotor_effective_velocities,
        )

        # Compute power
        power = power_interpolator(rotor_effective_velocities) * 1e3 # Convert to W

        return power

    def thrust_coefficient(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        tilt_interp: NDArrayObject,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        correct_cp_ct_for_tilt: bool = False,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        # Construct thrust coefficient interpolant
        thrust_coefficient_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["thrust_coefficient"],
            fill_value=0.0001,
            bounds_error=False,
        )

        # Compute the effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        # TODO: Do we need an air density correction here?
        thrust_coefficient = thrust_coefficient_interpolator(rotor_average_velocities)
        thrust_coefficient = np.clip(thrust_coefficient, 0.0001, 0.9999)

        # Apply tilt and yaw corrections
        # Compute the tilt, if using floating turbines
        old_tilt_angles = copy.deepcopy(tilt_angles)
        tilt_angles = compute_tilt_angles_for_floating_turbines(
            tilt_angles=tilt_angles,
            tilt_interp=tilt_interp,
            rotor_effective_velocities=rotor_average_velocities,
        )
        # Only update tilt angle if requested (if the tilt isn't accounted for in the Ct curve)
        tilt_angles = np.where(correct_cp_ct_for_tilt, tilt_angles, old_tilt_angles)

        thrust_coefficient = (
            thrust_coefficient
            * cosd(yaw_angles)
            * cosd(tilt_angles - power_thrust_table["ref_tilt"])
        )

        return thrust_coefficient

    def axial_induction(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        tilt_interp: NDArrayObject,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        correct_cp_ct_for_tilt: bool = False,
        **_ # <- Allows other models to accept other keyword arguments
    ):

        thrust_coefficient = CosineLossTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            yaw_angles=yaw_angles,
            tilt_angles=tilt_angles,
            tilt_interp=tilt_interp,
            average_method=average_method,
            cubature_weights=cubature_weights,
            correct_cp_ct_for_tilt=correct_cp_ct_for_tilt
        )

        misalignment_loss = cosd(yaw_angles) * cosd(tilt_angles - power_thrust_table["ref_tilt"])
        return 0.5 / misalignment_loss * (1 - np.sqrt(1 - thrust_coefficient * misalignment_loss))



@define
class TUMLossTurbine(BaseOperationModel):
    """
    Static class defining a wind turbine model that may be misaligned with the flow.
    Nonzero tilt and yaw angles are handled via the model presented in
    https://doi.org/10.5194/wes-2023-133 .

    The method requires C_P, C_T look-up tables as functions of tip speed ratio and blade pitch
    angle, available here:
    "../floris/turbine_library/LUT_iea15MW.npz" for the IEA 3.4 MW (Bortolotti et al., 2019)
    As with all turbine submodules, implements only static power() and thrust_coefficient() methods,
    which are called by power() and thrust_coefficient() on turbine.py, respectively.
    There are also two new functions, i.e. compute_local_vertical_shear() and control_trajectory().
    These are called by thrust_coefficient() and power() to compute the vertical shear and predict
    the turbine status in terms of tip speed ratio and pitch angle.
    This class is not intended to be instantiated; it simply defines a library of static methods.

    TODO: Should the turbine submodels each implement axial_induction()?
    """

    def compute_local_vertical_shear(velocities,avg_velocities):
        num_rows, num_cols = avg_velocities.shape
        shear = np.zeros_like(avg_velocities)
        for i in np.arange(num_rows):
            for j in np.arange(num_cols):
                mean_speed = np.mean(velocities[i,j,:,:],axis=0)
                if len(mean_speed) % 2 != 0: # odd number
                    u_u_hh     = mean_speed/mean_speed[int(np.floor(len(mean_speed)/2))]
                else:
                    u_u_hh     = (
                        mean_speed
                        /(mean_speed[int((len(mean_speed)/2))]
                          +mean_speed[int((len(mean_speed)/2))-1]
                          )/2
                    )
                zg_R = np.linspace(-1,1,len(mean_speed)+2)
                polifit_k  = np.polyfit(zg_R[1:-1],1-u_u_hh,1)
                shear[i,j] = -polifit_k[0]
        return shear

    def control_trajectory(
            rotor_average_velocities,
            yaw_angles,
            tilt_angles,
            air_density,
            R,
            shear,
            sigma,
            cd,
            cl_alfa,
            beta,
            power_setpoints,
            power_thrust_table
        ):
        if power_setpoints is None:
            power_demanded = (
                np.ones_like(tilt_angles)*power_thrust_table["rated_power"]
                /power_thrust_table["generator_efficiency"]
            )
        else:
            power_demanded = power_setpoints/power_thrust_table["generator_efficiency"]

        def find_cp(sigma,cd,cl_alfa,gamma,delta,k,cosMu,sinMu,tsr,theta,MU,ct):
            #add a small misalignment in case MU = 0 to avoid division by 0
            if MU == 0:  
                MU = 1e-6
                sinMu = np.sin(MU)
                cosMu = np.cos(MU)
            a = 1-((1+np.sqrt(1-ct-1/16*sinMu**2*ct**2))/(2*(1+1/16*ct*sinMu**2)))
            SG = np.sin(np.deg2rad(gamma))
            CG = np.cos(np.deg2rad(gamma))
            SD = np.sin(np.deg2rad(delta))
            CD = np.cos(np.deg2rad(delta))
            k_1s = -1*(15*np.pi/32*np.tan((MU+sinMu*(ct/2))/2))

            p = sigma*((np.pi*cosMu**2*tsr*cl_alfa*(a - 1)**2
                         - (tsr*cd*np.pi*(
                                CD**2*CG**2*SD**2*k**2
                                + 3*CD**2*SG**2*k**2
                                - 8*CD*tsr*SG*k + 8*tsr**2
                            ))/16
                         - (np.pi*tsr*sinMu**2*cd)/2 - (2*np.pi*cosMu*tsr**2*cl_alfa*theta)/3
                         + (np.pi*cosMu**2*k_1s**2*tsr*a**2*cl_alfa)/4
                         + (2*np.pi*cosMu*tsr**2*a*cl_alfa*theta)/3
                         + (2*np.pi*CD*cosMu*tsr*SG*cl_alfa*k*theta)/3
                         + ((CD**2*cosMu**2*tsr*cl_alfa*k**2*np.pi*(a - 1)**2*(CG**2*SD**2 + SG**2))
                            /(4*sinMu**2))
                         - (2*np.pi*CD*cosMu*tsr*SG*a*cl_alfa*k*theta)/3
                         + ((CD**2*cosMu**2*k_1s**2*tsr*a**2*cl_alfa*k**2*np.pi
                             *(3*CG**2*SD**2 + SG**2))
                            /(24*sinMu**2))
                         - (np.pi*CD*CG*cosMu**2*k_1s*tsr*SD*a*cl_alfa*k)/sinMu
                         + (np.pi*CD*CG*cosMu**2*k_1s*tsr*SD*a**2*cl_alfa*k)/sinMu
                         + (np.pi*CD*CG*cosMu*k_1s*tsr**2*SD*a*cl_alfa*k*theta)/(5*sinMu)
                         - (np.pi*CD**2*CG*cosMu*k_1s*tsr*SD*SG*a*cl_alfa*k**2*theta)/(10*sinMu)
                         )/(2*np.pi))
            return p


        def get_ct(x,*data):
            sigma,cd,cl_alfa,gamma,delta,k,cosMu,sinMu,tsr,theta,MU = data
            #add a small misalignment in case MU = 0 to avoid division by 0
            if MU == 0:  
                MU = 1e-6
                sinMu = np.sin(MU)
                cosMu = np.cos(MU)
            CD = np.cos(np.deg2rad(delta))
            CG = np.cos(np.deg2rad(gamma))
            SD = np.sin(np.deg2rad(delta))
            SG = np.sin(np.deg2rad(gamma))
            a = (1- ( (1+np.sqrt(1-x-1/16*x**2*sinMu**2))/(2*(1+1/16*x*sinMu**2))) )
            k_1s = -1*(15*np.pi/32*np.tan((MU+sinMu*(x/2))/2))
            I1 = -(np.pi*cosMu*(tsr - CD*SG*k)*(a - 1)
                   + (CD*CG*cosMu*k_1s*SD*a*k*np.pi*(2*tsr - CD*SG*k))/(8*sinMu))/(2*np.pi)
            I2 = (np.pi*sinMu**2 + (np.pi*(CD**2*CG**2*SD**2*k**2
                                           + 3*CD**2*SG**2*k**2 - 8*CD*tsr*SG*k
                                           + 8*tsr**2))/12)/(2*np.pi)

            return (sigma*(cd+cl_alfa)*(I1) - sigma*cl_alfa*theta*(I2)) - x


        ## Define function to get tip speed ratio
        def get_tsr(x,*data):
            (air_density,R,sigma,shear,cd,cl_alfa,beta,gamma,tilt,u,pitch_in,omega_lut_pow,
             torque_lut_omega,cp_i,pitch_i,tsr_i) = data

            omega_lut_torque = omega_lut_pow*np.pi/30

            omega   = x*u/R
            omega_rpm = omega*30/np.pi

            torque_nm = np.interp(omega,omega_lut_torque,torque_lut_omega)

            mu    = np.arccos(np.cos(np.deg2rad(gamma))*np.cos(np.deg2rad(tilt)))
            data  = (sigma,cd,cl_alfa,gamma,tilt,shear,np.cos(mu),np.sin(mu),x,
                     np.deg2rad(pitch_in)+np.deg2rad(beta),mu)
            x0    = 0.1
            [ct,infodict,ier,mesg] = fsolve(get_ct, x0,args=data,full_output=True,factor=0.1)
            cp = find_cp(
                sigma,
                cd,
                cl_alfa,
                gamma,
                tilt,
                shear,
                np.cos(mu),
                np.sin(mu),
                x,
                np.deg2rad(pitch_in)+np.deg2rad(beta),
                mu,
                ct
            )

            mu    = np.arccos(np.cos(np.deg2rad(0))*np.cos(np.deg2rad(tilt)))
            data  = (sigma,cd,cl_alfa,0,tilt,shear,np.cos(mu),np.sin(mu),x,
                     np.deg2rad(pitch_in)+np.deg2rad(beta),mu)
            x0    = 0.1
            [ct,infodict,ier,mesg] = fsolve(get_ct, x0,args=data,full_output=True,factor=0.1)
            cp0 = find_cp(
                sigma,
                cd,
                cl_alfa,
                0,
                tilt,
                shear,
                np.cos(mu),
                np.sin(mu),
                x,
                np.deg2rad(pitch_in)+np.deg2rad(beta),
                mu,
                ct
            )

            eta_p = cp/cp0

            interp   = RegularGridInterpolator((np.squeeze((tsr_i)),
                                np.squeeze((pitch_i))), cp_i,
                                               bounds_error=False, fill_value=None)

            Cp_now = interp((x,pitch_in),method='cubic')
            cp_g1 =  Cp_now*eta_p
            aero_pow = 0.5*air_density*(np.pi*R**2)*(u)**3*cp_g1
            electric_pow = torque_nm*(omega_rpm*np.pi/30)

            y = aero_pow - electric_pow
            return y

        ## Define function to get pitch angle
        def get_pitch(x,*data):
            (air_density,R,sigma,shear,cd,cl_alfa,beta,gamma,tilt,u,omega_rated,omega_lut_torque,
             torque_lut_omega,cp_i,pitch_i,tsr_i) = data

            omega_rpm   = omega_rated*30/np.pi
            tsr     = omega_rated*R/(u)

            pitch_in = np.deg2rad(x)
            torque_nm = np.interp(omega_rpm,omega_lut_torque*30/np.pi,torque_lut_omega)

            mu    = np.arccos(np.cos(np.deg2rad(gamma))*np.cos(np.deg2rad(tilt)))
            data  = (sigma,cd,cl_alfa,gamma,tilt,shear,np.cos(mu),np.sin(mu),tsr,
                     (pitch_in)+np.deg2rad(beta),mu)
            x0    = 0.1
            [ct,infodict,ier,mesg] = fsolve(get_ct, x0,args=data,full_output=True,factor=0.1)
            cp = find_cp(
                sigma,
                cd,
                cl_alfa,
                gamma,
                tilt,
                shear,
                np.cos(mu),
                np.sin(mu),
                tsr,
                (pitch_in)+np.deg2rad(beta),
                mu,
                ct
            )

            mu    = np.arccos(np.cos(np.deg2rad(0))*np.cos(np.deg2rad(tilt)))
            data  = (sigma,cd,cl_alfa,0,tilt,shear,np.cos(mu),np.sin(mu),tsr,
                     (pitch_in)+np.deg2rad(beta),mu)
            x0    = 0.1
            [ct,infodict,ier,mesg] = fsolve(get_ct, x0,args=data,full_output=True,factor=0.1)
            cp0 = find_cp(
                sigma,
                cd,
                cl_alfa,
                0,
                tilt,
                shear,
                np.cos(mu),
                np.sin(mu),
                tsr,
                (pitch_in)+np.deg2rad(beta),
                mu,
                ct
            )

            eta_p = cp/cp0

            interp   = RegularGridInterpolator(
                (np.squeeze((tsr_i)), np.squeeze((pitch_i))),
                cp_i,
                bounds_error=False,
                fill_value=None
            )

            Cp_now = interp((tsr,x),method='cubic')
            cp_g1 =  Cp_now*eta_p
            aero_pow = 0.5*air_density*(np.pi*R**2)*(u)**3*cp_g1
            electric_pow = torque_nm*(omega_rpm*np.pi/30)

            y = aero_pow - electric_pow
            return y

        pkgroot = Path(os.path.dirname(os.path.abspath(__file__))).resolve().parents[1]
        lut_file = pkgroot / "turbine_library" / "LUT_iea15MW.npz"
        LUT         = np.load(lut_file)
        cp_i = LUT['cp_lut']
        pitch_i = LUT['pitch_lut']
        tsr_i = LUT['tsr_lut']
        idx = np.squeeze(np.where(cp_i == np.max(cp_i)))

        tsr_opt   = tsr_i[idx[0]]
        pitch_opt = pitch_i[idx[1]]
        max_cp    = cp_i[idx[0],idx[1]]

        omega_cut_in = 0     # RPM
        omega_max    = power_thrust_table["rated_rpm"] # RPM 
        rated_power_aero  = power_thrust_table["rated_power"]/power_thrust_table["generator_efficiency"]  # MW
        #%% Compute torque-rpm relation and check for region 2-and-a-half
        Region2andAhalf = False

        omega_array = np.linspace(omega_cut_in,omega_max,161)*np.pi/30 # rad/s
        Q = (0.5*air_density*omega_array**2*R**5 * np.pi * max_cp ) / tsr_opt**3

        Paero_array = Q*omega_array

        if Paero_array[-1] < rated_power_aero: # then we have region 2and1/2
            Region2andAhalf = True
            Q_extra = rated_power_aero/(omega_max*np.pi/30)
            Q = np.append(Q,Q_extra)
            (Paero_array[-1]/(0.5*air_density*np.pi*R**2*max_cp))**(1/3)
            omega_array = np.append(omega_array,omega_array[-1])
            Paero_array = np.append(Paero_array,rated_power_aero)
        else: # limit aero_power to the last Q*omega_max
            rated_power_aero = Paero_array[-1]

        u_rated = (rated_power_aero/(0.5*air_density*np.pi*R**2*max_cp))**(1/3)
        u_array = np.linspace(3,25,45)
        idx = np.argmin(np.abs(u_array-u_rated))
        if u_rated > u_array[idx]:
            u_array = np.insert(u_array,idx+1,u_rated)
        else:
            u_array = np.insert(u_array,idx,u_rated)

        pow_lut_omega = Paero_array
        omega_lut_pow = omega_array*30/np.pi
        torque_lut_omega = Q
        omega_lut_torque = omega_lut_pow

        num_rows, num_cols = tilt_angles.shape

        omega_rated = np.zeros_like(rotor_average_velocities)
        u_rated     = np.zeros_like(rotor_average_velocities)
        for i in np.arange(num_rows):
            for j in np.arange(num_cols):
                omega_rated[i,j] = (
                    np.interp(power_demanded[i,j],pow_lut_omega,omega_lut_pow)
                    *np.pi/30 #rad/s
                )
                u_rated[i,j] = (power_demanded[i,j]/(0.5*air_density*np.pi*R**2*max_cp))**(1/3)

        pitch_out = np.zeros_like(rotor_average_velocities)
        tsr_out = np.zeros_like(rotor_average_velocities)

        for i in np.arange(num_rows):
            yaw  = yaw_angles[i,:]
            tilt = tilt_angles[i,:]
            k = shear[i,:]
            for j in np.arange(num_cols):
                u_v = rotor_average_velocities[i,j]
                if u_v > u_rated[i,j]:
                    tsr_v = omega_rated[i,j]*R/u_v*np.cos(np.deg2rad(yaw[j]))**0.5
                else:
                    tsr_v = tsr_opt*np.cos(np.deg2rad(yaw[j]))
                if Region2andAhalf: # fix for interpolation
                    omega_lut_torque[-1] = omega_lut_torque[-1]+1e-2
                    omega_lut_pow[-1]    = omega_lut_pow[-1]+1e-2

                data = (air_density,R,sigma,k[j],cd,cl_alfa,beta,yaw[j],tilt[j],u_v,pitch_opt,
                        omega_lut_pow,torque_lut_omega,cp_i,pitch_i,tsr_i)
                [tsr_out_soluzione,infodict,ier,mesg] = fsolve(
                    get_tsr,tsr_v,args=data,full_output=True
                )
                # check if solution was possible. If not, we are in region 3
                if (np.abs(infodict['fvec']) > 10 or tsr_out_soluzione < 4):
                    tsr_out_soluzione = 1000

                # save solution
                tsr_outO = tsr_out_soluzione
                omega    = tsr_outO*u_v/R

                # check if we are in region 2 or 3
                if omega < omega_rated[i,j]: # region 2
                    # Define optimum pitch
                    pitch_out0 = pitch_opt

                else: # region 3
                    tsr_outO = omega_rated[i,j]*R/u_v
                    data = (air_density,R,sigma,k[j],cd,cl_alfa,beta,yaw[j],tilt[j],u_v,
                            omega_rated[i,j],omega_array,Q,cp_i,pitch_i,tsr_i)
                    # solve aero-electrical power balance with TSR from rated omega
                    [pitch_out_soluzione,infodict,ier,mesg] = fsolve(
                        get_pitch,u_v,args=data,factor=0.1,full_output=True,
                    xtol=1e-10,maxfev=2000)
                    if pitch_out_soluzione < pitch_opt:
                        pitch_out_soluzione = pitch_opt
                    pitch_out0 = pitch_out_soluzione
                #%% COMPUTE CP AND CT GIVEN THE PITCH AND TSR FOUND ABOVE
                pitch_out[i,j]         = np.squeeze(pitch_out0)
                tsr_out[i,j]           = tsr_outO

        return pitch_out, tsr_out

    def power(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        power_setpoints: NDArrayFloat,
        tilt_interp: NDArrayObject,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        correct_cp_ct_for_tilt: bool = False,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        # # Construct power interpolant
        # power_interpolator = interp1d(
        #     power_thrust_table["wind_speed"],
        #     power_thrust_table["power"],
        #     fill_value=0.0,
        #     bounds_error=False,
        # )

        # sign convention. in the tum model, negative tilt creates tower clearance
        tilt_angles = -tilt_angles

        # Compute the power-effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"]
        )

        # Compute power
        def get_ct(x,*data):
            sigma,cd,cl_alfa,gamma,delta,k,cosMu,sinMu,tsr,theta,R,MU = data
            #add a small misalignment in case MU = 0 to avoid division by 0
            if MU == 0:  
                MU = 1e-6
                sinMu = np.sin(MU)
                cosMu = np.cos(MU)
            CD = np.cos(np.deg2rad(delta))
            CG = np.cos(np.deg2rad(gamma))
            SD = np.sin(np.deg2rad(delta))
            SG = np.sin(np.deg2rad(gamma))
            a = (1- ( (1+np.sqrt(1-x-1/16*x**2*sinMu**2))/(2*(1+1/16*x*sinMu**2))) )
            k_1s = -1*(15*np.pi/32*np.tan((MU+sinMu*(x/2))/2))
            I1 = -(np.pi*cosMu*(tsr - CD*SG*k)*(a - 1)
                   + (CD*CG*cosMu*k_1s*SD*a*k*np.pi*(2*tsr - CD*SG*k))/(8*sinMu))/(2*np.pi)
            I2 = (np.pi*sinMu**2 + (np.pi*(CD**2*CG**2*SD**2*k**2
                                           + 3*CD**2*SG**2*k**2 - 8*CD*tsr*SG*k
                                           + 8*tsr**2))/12)/(2*np.pi)

            return (sigma*(cd+cl_alfa)*(I1) - sigma*cl_alfa*theta*(I2)) - x

        def find_cp(sigma,cd,cl_alfa,gamma,delta,k,cosMu,sinMu,tsr,theta,R,MU,ct):
            #add a small misalignment in case MU = 0 to avoid division by 0
            if MU == 0:  
                MU = 1e-6
                sinMu = np.sin(MU)
                cosMu = np.cos(MU)
            a = 1-((1+np.sqrt(1-ct-1/16*sinMu**2*ct**2))/(2*(1+1/16*ct*sinMu**2)))
            SG = np.sin(np.deg2rad(gamma))
            CG = np.cos(np.deg2rad(gamma))
            SD = np.sin(np.deg2rad(delta))
            CD = np.cos(np.deg2rad(delta))
            k_1s = -1*(15*np.pi/32*np.tan((MU+sinMu*(ct/2))/2))

            p = sigma*((np.pi*cosMu**2*tsr*cl_alfa*(a - 1)**2
                         - (tsr*cd*np.pi*(CD**2*CG**2*SD**2*k**2 + 3*CD**2*SG**2*k**2
                            - 8*CD*tsr*SG*k + 8*tsr**2))/16
                         - (np.pi*tsr*sinMu**2*cd)/2 - (2*np.pi*cosMu*tsr**2*cl_alfa*theta)/3
                         + (np.pi*cosMu**2*k_1s**2*tsr*a**2*cl_alfa)/4
                         + (2*np.pi*cosMu*tsr**2*a*cl_alfa*theta)/3
                         + (2*np.pi*CD*cosMu*tsr*SG*cl_alfa*k*theta)/3
                         + ((CD**2*cosMu**2*tsr*cl_alfa*k**2*np.pi*(a - 1)**2*(CG**2*SD**2 + SG**2))
                            /(4*sinMu**2))
                         - (2*np.pi*CD*cosMu*tsr*SG*a*cl_alfa*k*theta)/3
                         + ((CD**2*cosMu**2*k_1s**2*tsr*a**2*cl_alfa*k**2*np.pi*(3*CG**2*SD**2
                             + SG**2))
                            /(24*sinMu**2))
                         - (np.pi*CD*CG*cosMu**2*k_1s*tsr*SD*a*cl_alfa*k)/sinMu
                         + (np.pi*CD*CG*cosMu**2*k_1s*tsr*SD*a**2*cl_alfa*k)/sinMu
                         + (np.pi*CD*CG*cosMu*k_1s*tsr**2*SD*a*cl_alfa*k*theta)/(5*sinMu)
                         - (np.pi*CD**2*CG*cosMu*k_1s*tsr*SD*SG*a*cl_alfa*k**2*theta)/(10*sinMu)
                        )/(2*np.pi)
                      )
            return p

        num_rows, num_cols = tilt_angles.shape
        (average_velocity(velocities))

        shear = TUMLossTurbine.compute_local_vertical_shear(velocities,average_velocity(velocities))

        beta = power_thrust_table["beta"]
        cd = power_thrust_table["cd"]
        cl_alfa = power_thrust_table["cl_alfa"]

        sigma = power_thrust_table["rotor_solidity"]
        R = power_thrust_table["rotor_diameter"]/2

        air_density = power_thrust_table["ref_air_density"]

        pitch_out, tsr_out = TUMLossTurbine.control_trajectory(
            rotor_effective_velocities,
            yaw_angles,
            tilt_angles,
            air_density,
            R,
            shear,
            sigma,
            cd,
            cl_alfa,
            beta,
            power_setpoints,
            power_thrust_table
        )

        MU = np.arccos(np.cos(np.deg2rad((yaw_angles)))*np.cos(np.deg2rad((tilt_angles))))
        cosMu = (np.cos(MU))
        sinMu = (np.sin(MU))
        tsr_array = (tsr_out)
        theta_array = (np.deg2rad(pitch_out+beta))

        x0 = 0.2

        p = np.zeros_like(average_velocity(velocities))

        for i in np.arange(num_rows):
            yaw  = yaw_angles[i,:]
            tilt = tilt_angles[i,:]
            k = shear[i,:]
            cMu  = cosMu[i,:]
            sMu  = sinMu[i,:]
            Mu   = MU[i,:]
            for j in np.arange(num_cols):
                # Break below command over multiple lines
                data = (
                    sigma,
                    cd,
                    cl_alfa,
                    yaw[j],
                    tilt[j],
                    k[j],
                    cMu[j],
                    sMu[j],
                    (tsr_array[i,j]),
                    (theta_array[i,j]),
                    R,
                    Mu[j]
                )
                ct, info, ier, msg = fsolve(get_ct, x0,args=data,full_output=True)
                if ier == 1:
                    p[i,j] = np.squeeze(find_cp(
                        sigma,
                        cd,
                        cl_alfa,
                        yaw[j],
                        tilt[j],
                        k[j],
                        cMu[j],
                        sMu[j],
                        (tsr_array[i,j]),
                        (theta_array[i,j]),
                        R,
                        Mu[j],
                        ct
                    ))
                else:
                    p[i,j] = -1e3

    ############################################################################

        yaw_angles = np.zeros_like(yaw_angles)
        MU = np.arccos(np.cos(np.deg2rad((yaw_angles)))*np.cos(np.deg2rad((tilt_angles))))
        cosMu = (np.cos(MU))
        sinMu = (np.sin(MU))

        p0 = np.zeros_like((average_velocity(velocities)))

        for i in np.arange(num_rows):
            yaw  = yaw_angles[i,:]
            tilt = tilt_angles[i,:]
            k = shear[i,:]
            cMu  = cosMu[i,:]
            sMu  = sinMu[i,:]
            Mu   = MU[i,:]
            for j in np.arange(num_cols):
                data = (sigma,cd,cl_alfa,yaw[j],tilt[j],k[j],cMu[j],sMu[j],(tsr_array[i,j]),
                        (theta_array[i,j]),R,Mu[j])
                ct, info, ier, msg = fsolve(get_ct, x0,args=data,full_output=True)
                if ier == 1:
                    p0[i,j] = np.squeeze(find_cp(
                        sigma,
                        cd,
                        cl_alfa,
                        yaw[j],
                        tilt[j],
                        k[j],
                        cMu[j],
                        sMu[j],
                        (tsr_array[i,j]),
                        (theta_array[i,j]),
                        R,
                        Mu[j],
                        ct
                    ))
                else:
                    p0[i,j] = -1e3

        ratio = p/p0

    ############################################################################

        pkgroot = Path(os.path.dirname(os.path.abspath(__file__))).resolve().parents[1]
        lut_file = pkgroot / "turbine_library" / "LUT_iea15MW.npz"
        LUT         = np.load(lut_file)
        cp_i = LUT['cp_lut']
        pitch_i = LUT['pitch_lut']
        tsr_i = LUT['tsr_lut']
        interp_lut = RegularGridInterpolator(
            (tsr_i,pitch_i),
            cp_i,
            bounds_error=False,
            fill_value=None
        )

        power_coefficient = np.zeros_like(average_velocity(velocities))
        for i in np.arange(num_rows):
            for j in np.arange(num_cols):
                cp_interp = interp_lut(np.array([(tsr_array[i,j]),(pitch_out[i,j])]),method='cubic')
                power_coefficient[i,j] = np.squeeze(cp_interp*ratio[i,j])

        print('Tip speed ratio' + str(tsr_array))
        print('Pitch out: ' + str(pitch_out))
        power = (
            0.5*air_density*(rotor_effective_velocities)**3*np.pi*R**2
            *(power_coefficient)*power_thrust_table["generator_efficiency"]
        )
        return power

    def thrust_coefficient(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        power_setpoints: NDArrayFloat,
        tilt_interp: NDArrayObject,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        correct_cp_ct_for_tilt: bool = False,
        **_ # <- Allows other models to accept other keyword arguments
    ):

        # sign convention. in the tum model, negative tilt creates tower clearance
        tilt_angles = -tilt_angles

        # Compute the effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )


        # Apply tilt and yaw corrections
        # Compute the tilt, if using floating turbines
        old_tilt_angles = copy.deepcopy(tilt_angles)
        tilt_angles = compute_tilt_angles_for_floating_turbines(
            tilt_angles=tilt_angles,
            tilt_interp=tilt_interp,
            rotor_effective_velocities=rotor_average_velocities,
        )
        # Only update tilt angle if requested (if the tilt isn't accounted for in the Ct curve)
        tilt_angles = np.where(correct_cp_ct_for_tilt, tilt_angles, old_tilt_angles)

        def get_ct(x,*data):
            sigma,cd,cl_alfa,gamma,delta,k,cosMu,sinMu,tsr,theta,R,MU = data
            #add a small misalignment in case MU = 0 to avoid division by 0
            if MU == 0:  
                MU = 1e-6
                sinMu = np.sin(MU)
                cosMu = np.cos(MU)
            CD = np.cos(np.deg2rad(delta))
            CG = np.cos(np.deg2rad(gamma))
            SD = np.sin(np.deg2rad(delta))
            SG = np.sin(np.deg2rad(gamma))
            a = (1- ( (1+np.sqrt(1-x-1/16*x**2*sinMu**2))/(2*(1+1/16*x*sinMu**2))) )
            k_1s = -1*(15*np.pi/32*np.tan((MU+sinMu*(x/2))/2))
            I1 = -(np.pi*cosMu*(tsr - CD*SG*k)*(a - 1)
                   + (CD*CG*cosMu*k_1s*SD*a*k*np.pi*(2*tsr - CD*SG*k))/(8*sinMu))/(2*np.pi)
            I2 = (np.pi*sinMu**2 + (np.pi*(CD**2*CG**2*SD**2*k**2
                                           + 3*CD**2*SG**2*k**2 - 8*CD*tsr*SG*k
                                           + 8*tsr**2))/12)/(2*np.pi)

            return (sigma*(cd+cl_alfa)*(I1) - sigma*cl_alfa*theta*(I2)) - x

        beta = power_thrust_table["beta"]
        cd = power_thrust_table["cd"]
        cl_alfa = power_thrust_table["cl_alfa"]

        sigma = power_thrust_table["rotor_solidity"]
        R = power_thrust_table["rotor_diameter"]/2

        shear = TUMLossTurbine.compute_local_vertical_shear(velocities,average_velocity(velocities))

        air_density = power_thrust_table["ref_air_density"] # CHANGE

        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"]
        )

        pitch_out, tsr_out = TUMLossTurbine.control_trajectory(
            rotor_effective_velocities,
            yaw_angles,
            tilt_angles,
            air_density,
            R,
            shear,
            sigma,
            cd,
            cl_alfa,
            beta,
            power_setpoints,
            power_thrust_table
        )

        num_rows, num_cols = tilt_angles.shape

        (average_velocity(velocities))
        MU = np.arccos(np.cos(np.deg2rad((yaw_angles)))*np.cos(np.deg2rad((tilt_angles))))
        cosMu = (np.cos(MU))
        sinMu = (np.sin(MU))
        # u = np.squeeze(u)
        theta_array = (np.deg2rad(pitch_out+beta))
        tsr_array = (tsr_out)

        x0 = 0.2

        thrust_coefficient1 = np.zeros_like(average_velocity(velocities))
        for i in np.arange(num_rows):
            yaw  = yaw_angles[i,:]
            tilt = tilt_angles[i,:]
            cMu  = cosMu[i,:]
            sMu  = sinMu[i,:]
            Mu   = MU[i,:]
            for j in np.arange(num_cols):
                data = (sigma,cd,cl_alfa,yaw[j],tilt[j],shear[i,j],cMu[j],sMu[j],(tsr_array[i,j]),
                        (theta_array[i,j]),R,Mu[j])
                ct = fsolve(get_ct, x0,args=data)
                thrust_coefficient1[i,j] = np.squeeze(np.clip(ct, 0.0001, 0.9999))


        yaw_angles = np.zeros_like(yaw_angles)
        MU = np.arccos(np.cos(np.deg2rad((yaw_angles)))*np.cos(np.deg2rad((tilt_angles))))
        cosMu = (np.cos(MU))
        sinMu = (np.sin(MU))

        thrust_coefficient0 = np.zeros_like(average_velocity(velocities))

        for i in np.arange(num_rows):
            yaw  = yaw_angles[i,:]
            tilt = tilt_angles[i,:]
            cMu  = cosMu[i,:]
            sMu  = sinMu[i,:]
            Mu   = MU[i,:]
            for j in np.arange(num_cols):
                data = (sigma,cd,cl_alfa,yaw[j],tilt[j],shear[i,j],cMu[j],sMu[j],(tsr_array[i,j]),
                        (theta_array[i,j]),R,Mu[j])
                ct = fsolve(get_ct, x0,args=data)
                thrust_coefficient0[i,j] = np.squeeze(ct) #np.clip(ct, 0.0001, 0.9999)

        ############################################################################

        ratio = thrust_coefficient1/thrust_coefficient0

        pkgroot = Path(os.path.dirname(os.path.abspath(__file__))).resolve().parents[1]
        lut_file = pkgroot / "turbine_library" / "LUT_iea15MW.npz"
        LUT         = np.load(lut_file)
        ct_i = LUT['ct_lut']
        pitch_i = LUT['pitch_lut']
        tsr_i = LUT['tsr_lut']
        interp_lut = RegularGridInterpolator(
            (tsr_i,pitch_i),
            ct_i,
            bounds_error=False,
            fill_value=None
        )#*0.9722085500886761)


        thrust_coefficient = np.zeros_like(average_velocity(velocities))

        for i in np.arange(num_rows):
            for j in np.arange(num_cols):
                ct_interp = interp_lut(np.array([(tsr_array[i,j]),(pitch_out[i,j])]),method='cubic')
                thrust_coefficient[i,j] = np.squeeze(ct_interp*ratio[i,j])

        return thrust_coefficient

    def axial_induction(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        power_setpoints: NDArrayFloat,
        tilt_interp: NDArrayObject,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        correct_cp_ct_for_tilt: bool = False,
        **_ # <- Allows other models to accept other keyword arguments
    ):

        # sign convention. in the tum model, negative tilt creates tower clearance
        tilt_angles = -tilt_angles

        # Compute the effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )


        # Apply tilt and yaw corrections
        # Compute the tilt, if using floating turbines
        old_tilt_angles = copy.deepcopy(tilt_angles)
        tilt_angles = compute_tilt_angles_for_floating_turbines(
            tilt_angles=tilt_angles,
            tilt_interp=tilt_interp,
            rotor_effective_velocities=rotor_average_velocities,
        )
        # Only update tilt angle if requested (if the tilt isn't accounted for in the Ct curve)
        tilt_angles = np.where(correct_cp_ct_for_tilt, tilt_angles, old_tilt_angles)

        def get_ct(x,*data):
            sigma,cd,cl_alfa,gamma,delta,k,cosMu,sinMu,tsr,theta,R,MU = data
            #add a small misalignment in case MU = 0 to avoid division by 0
            if MU == 0:  
                MU = 1e-6
                sinMu = np.sin(MU)
                cosMu = np.cos(MU)
            CD = np.cos(np.deg2rad(delta))
            CG = np.cos(np.deg2rad(gamma))
            SD = np.sin(np.deg2rad(delta))
            SG = np.sin(np.deg2rad(gamma))
            a = (1- ( (1+np.sqrt(1-x-1/16*x**2*sinMu**2))/(2*(1+1/16*x*sinMu**2))) )
            k_1s = -1*(15*np.pi/32*np.tan((MU+sinMu*(x/2))/2))
            I1 = -(np.pi*cosMu*(tsr - CD*SG*k)*(a - 1)
                   + (CD*CG*cosMu*k_1s*SD*a*k*np.pi*(2*tsr - CD*SG*k))/(8*sinMu))/(2*np.pi)
            I2 = (np.pi*sinMu**2 + (np.pi*(CD**2*CG**2*SD**2*k**2
                                           + 3*CD**2*SG**2*k**2 - 8*CD*tsr*SG*k
                                           + 8*tsr**2))/12)/(2*np.pi)

            return (sigma*(cd+cl_alfa)*(I1) - sigma*cl_alfa*theta*(I2)) - x

        beta = power_thrust_table["beta"]
        cd = power_thrust_table["cd"]
        cl_alfa = power_thrust_table["cl_alfa"]

        sigma = power_thrust_table["rotor_solidity"]
        R = power_thrust_table["rotor_diameter"]/2

        shear = TUMLossTurbine.compute_local_vertical_shear(velocities,average_velocity(velocities))

        air_density = power_thrust_table["ref_air_density"] # CHANGE
        
        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"]
        )
        
        pitch_out, tsr_out = TUMLossTurbine.control_trajectory(
            rotor_effective_velocities,
            yaw_angles,
            tilt_angles,
            air_density,
            R,
            shear,
            sigma,
            cd,
            cl_alfa,
            beta,
            power_setpoints,
            power_thrust_table
        )

        num_rows, num_cols = tilt_angles.shape

        (average_velocity(velocities))
        MU = np.arccos(np.cos(np.deg2rad((yaw_angles)))*np.cos(np.deg2rad((tilt_angles))))
        cosMu = (np.cos(MU))
        sinMu = (np.sin(MU))
        # u = np.squeeze(u)
        theta_array = (np.deg2rad(pitch_out+beta))
        tsr_array = (tsr_out)

        x0 = 0.2

        thrust_coefficient1 = np.zeros_like(average_velocity(velocities))
        for i in np.arange(num_rows):
            yaw  = yaw_angles[i,:]
            tilt = tilt_angles[i,:]
            cMu  = cosMu[i,:]
            sMu  = sinMu[i,:]
            Mu   = MU[i,:]
            for j in np.arange(num_cols):
                data = (sigma,cd,cl_alfa,yaw[j],tilt[j],shear[i,j],cMu[j],sMu[j],(tsr_array[i,j]),
                        (theta_array[i,j]),R,Mu[j])
                ct = fsolve(get_ct, x0,args=data)
                thrust_coefficient1[i,j] = np.squeeze(np.clip(ct, 0.0001, 0.9999))


        yaw_angles = np.zeros_like(yaw_angles)
        MU = np.arccos(np.cos(np.deg2rad((yaw_angles)))*np.cos(np.deg2rad((tilt_angles))))
        cosMu = (np.cos(MU))
        sinMu = (np.sin(MU))

        thrust_coefficient0 = np.zeros_like(average_velocity(velocities))

        for i in np.arange(num_rows):
            yaw  = yaw_angles[i,:]
            tilt = tilt_angles[i,:]
            cMu  = cosMu[i,:]
            sMu  = sinMu[i,:]
            Mu   = MU[i,:]
            for j in np.arange(num_cols):
                data = (sigma,cd,cl_alfa,yaw[j],tilt[j],shear[i,j],cMu[j],sMu[j],(tsr_array[i,j]),
                        (theta_array[i,j]),R,Mu[j])
                ct = fsolve(get_ct, x0,args=data)
                thrust_coefficient0[i,j] = np.squeeze(ct) #np.clip(ct, 0.0001, 0.9999)

        ############################################################################

        ratio = thrust_coefficient1/thrust_coefficient0

        pkgroot = Path(os.path.dirname(os.path.abspath(__file__))).resolve().parents[1]
        lut_file = pkgroot / "turbine_library" / "LUT_iea15MW.npz"
        LUT         = np.load(lut_file)
        ct_i = LUT['ct_lut']
        pitch_i = LUT['pitch_lut']
        tsr_i = LUT['tsr_lut']
        interp_lut = RegularGridInterpolator(
            (tsr_i,pitch_i),
            ct_i,
            bounds_error=False,
            fill_value=None
        )#*0.9722085500886761)

        axial_induction = np.zeros_like(average_velocity(velocities))

        for i in np.arange(num_rows):
            for j in np.arange(num_cols):
                ct_interp = interp_lut(np.array([(tsr_array[i,j]),(pitch_out[i,j])]),method='cubic')
                ct = ct_interp*ratio[i,j]
                a  = (1- ( (1+np.sqrt(1-ct-1/16*ct**2*sMu[j]**2))/(2*(1+1/16*ct*sMu[j]**2))) )
                axial_induction[i,j] = np.squeeze(np.clip(a, 0.0001, 0.9999))

        return axial_induction



@define
class SimpleDeratingTurbine(BaseOperationModel):
    """
    power_thrust_table is a dictionary (normally defined on the turbine input yaml)
    that contains the parameters necessary to evaluate power(), thrust(), and axial_induction().
    Any specific parameters for derating can be placed here. (they can be added to the turbine
    yaml). For this operation model to receive those arguements, they'll need to be
    added to the kwargs dictionaries in the respective functions on turbine.py. They won't affect
    the other operation models.
    """
    def power(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        power_setpoints: NDArrayFloat | None,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        base_powers = SimpleTurbine.power(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            air_density=air_density,
            average_method=average_method,
            cubature_weights=cubature_weights
        )
        if power_setpoints is None:
            return base_powers
        else:
            return np.minimum(base_powers, power_setpoints)

        # TODO: would we like special handling of zero power setpoints
        # (mixed with non-zero values) to speed up computation in that case?

    def thrust_coefficient(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        power_setpoints: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        base_thrust_coefficients = SimpleTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            average_method=average_method,
            cubature_weights=cubature_weights
        )
        if power_setpoints is None:
            return base_thrust_coefficients
        else:
            # Assume thrust coefficient scales directly with power
            base_powers = SimpleTurbine.power(
                power_thrust_table=power_thrust_table,
                velocities=velocities,
                air_density=air_density
            )
            power_fractions = power_setpoints / base_powers
            thrust_coefficients = power_fractions * base_thrust_coefficients
            return np.minimum(base_thrust_coefficients, thrust_coefficients)

    def axial_induction(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        power_setpoints: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        thrust_coefficient = SimpleDeratingTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            air_density=air_density,
            power_setpoints=power_setpoints,
            average_method=average_method,
            cubature_weights=cubature_weights,
        )

        return (1 - np.sqrt(1 - thrust_coefficient))/2

@define
class MixedOperationTurbine(BaseOperationModel):

    def power(
        yaw_angles: NDArrayFloat,
        power_setpoints: NDArrayFloat,
        **kwargs
    ):
        # Yaw angles mask all yaw_angles not equal to zero
        yaw_angles_mask = yaw_angles != 0.0
        power_setpoints_mask = power_setpoints < POWER_SETPOINT_DEFAULT
        neither_mask = np.logical_not(yaw_angles_mask) & np.logical_not(power_setpoints_mask)

        if (power_setpoints_mask & yaw_angles_mask).any():
            raise ValueError((
                "Power setpoints and yaw angles are incompatible."
                "If yaw_angles entry is nonzero, power_setpoints must be greater than"
                " or equal to {0}.".format(POWER_SETPOINT_DEFAULT)
            ))

        powers = np.zeros_like(power_setpoints)
        powers[yaw_angles_mask] += CosineLossTurbine.power(
            yaw_angles=yaw_angles,
            **kwargs
        )[yaw_angles_mask]
        powers[power_setpoints_mask] += SimpleDeratingTurbine.power(
            power_setpoints=power_setpoints,
            **kwargs
        )[power_setpoints_mask]
        powers[neither_mask] += SimpleTurbine.power(
            **kwargs
        )[neither_mask]

        return powers

    def thrust_coefficient(
        yaw_angles: NDArrayFloat,
        power_setpoints: NDArrayFloat,
        **kwargs
    ):
        yaw_angles_mask = yaw_angles != 0.0
        power_setpoints_mask = power_setpoints < POWER_SETPOINT_DEFAULT
        neither_mask = np.logical_not(yaw_angles_mask) & np.logical_not(power_setpoints_mask)

        if (power_setpoints_mask & yaw_angles_mask).any():
            raise ValueError((
                "Power setpoints and yaw angles are incompatible."
                "If yaw_angles entry is nonzero, power_setpoints must be greater than"
                " or equal to {0}.".format(POWER_SETPOINT_DEFAULT)
            ))

        thrust_coefficients = np.zeros_like(power_setpoints)
        thrust_coefficients[yaw_angles_mask] += CosineLossTurbine.thrust_coefficient(
            yaw_angles=yaw_angles,
            **kwargs
        )[yaw_angles_mask]
        thrust_coefficients[power_setpoints_mask] += SimpleDeratingTurbine.thrust_coefficient(
            power_setpoints=power_setpoints,
            **kwargs
        )[power_setpoints_mask]
        thrust_coefficients[neither_mask] += SimpleTurbine.thrust_coefficient(
            **kwargs
        )[neither_mask]

        return thrust_coefficients

    def axial_induction(
        yaw_angles: NDArrayFloat,
        power_setpoints: NDArrayFloat,
        **kwargs
    ):
        yaw_angles_mask = yaw_angles != 0.0
        power_setpoints_mask = power_setpoints < POWER_SETPOINT_DEFAULT
        neither_mask = np.logical_not(yaw_angles_mask) & np.logical_not(power_setpoints_mask)

        if (power_setpoints_mask & yaw_angles_mask).any():
            raise ValueError((
                "Power setpoints and yaw angles are incompatible."
                "If yaw_angles entry is nonzero, power_setpoints must be greater than"
                " or equal to {0}.".format(POWER_SETPOINT_DEFAULT)
            ))

        axial_inductions = np.zeros_like(power_setpoints)
        axial_inductions[yaw_angles_mask] += CosineLossTurbine.axial_induction(
            yaw_angles=yaw_angles,
            **kwargs
        )[yaw_angles_mask]
        axial_inductions[power_setpoints_mask] += SimpleDeratingTurbine.axial_induction(
            power_setpoints=power_setpoints,
            **kwargs
        )[power_setpoints_mask]
        axial_inductions[neither_mask] += SimpleTurbine.axial_induction(
            **kwargs
        )[neither_mask]

        return axial_inductions

@define
class AWCTurbine(BaseOperationModel):
    """
    power_thrust_table is a dictionary (normally defined on the turbine input yaml)
    that contains the parameters necessary to evaluate power(), thrust(), and axial_induction().

    Feel free to put any Helix tuning parameters into here (they can be added to the turbine yaml).
    Also, feel free to add any commanded inputs to power(), thrust_coefficient(), or
    axial_induction(). For this operation model to receive those arguments, they'll need to be
    added to the kwargs dictionaries in the respective functions on turbine.py. They won't affect
    the other operation models.
    """

    def power(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        awc_modes: str,
        awc_amplitudes: NDArrayFloat | None,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        base_powers = SimpleTurbine.power(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            air_density=air_density,
            average_method=average_method,
            cubature_weights=cubature_weights
        )

        if (awc_modes == 'helix').any():
            if np.any(np.isclose(
                base_powers/1000,
                np.max(power_thrust_table['power'])
                )):
                raise UserWarning(
                    'The selected wind speed is above or near rated wind speed. '
                    '`AWCTurbine` operation model is not designed '
                    'or verified for above-rated conditions.'
                    )
            return base_powers * (1 - (
                power_thrust_table['helix_power_b']
                + power_thrust_table['helix_power_c']*base_powers
                )
                *awc_amplitudes**power_thrust_table['helix_a']
            ) # TODO: Should probably add max function here
        if (awc_modes == 'baseline').any():
            return base_powers
        else:
            raise UserWarning(
                'Active wake mixing strategies other than the `helix` strategy '
                'have not yet been implemented in FLORIS. Returning baseline power.'
                )


    def thrust_coefficient(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        awc_modes: str,
        awc_amplitudes: NDArrayFloat | None,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        base_thrust_coefficients = SimpleTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            average_method=average_method,
            cubature_weights=cubature_weights
        )
        if (awc_modes == 'helix').any():
            return base_thrust_coefficients * (1 - (
                power_thrust_table['helix_thrust_b']
                + power_thrust_table['helix_thrust_c']*base_thrust_coefficients
                )
                *awc_amplitudes**power_thrust_table['helix_a']
            )
        if (awc_modes == 'baseline').any():
            return base_thrust_coefficients
        else:
            raise UserWarning(
                'Active wake mixing strategies other than the `helix` strategy '
                'have not yet been implemented in FLORIS. Returning baseline power.'
                )

    def axial_induction(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        awc_modes: str,
        awc_amplitudes: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        thrust_coefficient = AWCTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            awc_modes=awc_modes,
            awc_amplitudes=awc_amplitudes,
            average_method=average_method,
            cubature_weights=cubature_weights,
        )

        return (1 - np.sqrt(1 - thrust_coefficient))/2

@define
class PeakShavingTurbine():

    def power(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        turbulence_intensities: NDArrayFloat,
        air_density: float,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        base_powers = SimpleTurbine.power(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            air_density=air_density,
            average_method=average_method,
            cubature_weights=cubature_weights
        )

        # Get fraction by thrust
        base_thrust_coefficients = SimpleTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            average_method=average_method,
            cubature_weights=cubature_weights
        )
        peak_shaving_thrust_coefficients = PeakShavingTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            turbulence_intensities=turbulence_intensities,
            average_method=average_method,
            cubature_weights=cubature_weights
        )

        # Compute equivalent axial inductions
        base_ais = (1 - np.sqrt(1 - base_thrust_coefficients))/2
        peak_shaving_ais = (1 - np.sqrt(1 - peak_shaving_thrust_coefficients))/2

        # Power proportion
        power_fractions = (
            (peak_shaving_thrust_coefficients * (1-peak_shaving_ais))
            / (base_thrust_coefficients * (1-base_ais))
        )

        # Apply fraction to power and return
        powers = power_fractions * base_powers

        return powers

    def thrust_coefficient(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        turbulence_intensities: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):
        base_thrust_coefficients = SimpleTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            average_method=average_method,
            cubature_weights=cubature_weights
        )

        peak_normal_thrust_prime = np.max(
            np.array(power_thrust_table["wind_speed"])**2
            * np.array(power_thrust_table["thrust_coefficient"])
        )
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )
        # Replace zeros with small values to avoid division by zero
        rotor_average_velocities = np.maximum(rotor_average_velocities, 0.01)
        max_allowable_thrust_coefficient = (
            (1-power_thrust_table["peak_shaving_fraction"])
            * peak_normal_thrust_prime
            / rotor_average_velocities**2
        )

        # Apply TI mask
        max_allowable_thrust_coefficient = np.where(
            (
                turbulence_intensities.mean(
                    axis=tuple([2 + i for i in range(turbulence_intensities.ndim - 2)])
                )
                >= power_thrust_table["peak_shaving_TI_threshold"]
            ),
            max_allowable_thrust_coefficient,
            base_thrust_coefficients
        )

        thrust_coefficient = np.minimum(base_thrust_coefficients, max_allowable_thrust_coefficient)

        return thrust_coefficient

    def axial_induction(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        turbulence_intensities: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_ # <- Allows other models to accept other keyword arguments
    ):

        thrust_coefficient = PeakShavingTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            turbulence_intensities=turbulence_intensities,
            average_method=average_method,
            cubature_weights=cubature_weights,
        )

        return (1 - np.sqrt(1 - thrust_coefficient))/2
