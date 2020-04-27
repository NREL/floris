# functions to couple floris with CCBlade and a controller

import floris.tools as wfct
import matplotlib.pyplot as plt
import os
from os import path
import numpy as np
import pickle
import copy
from ..utilities import setup_logger

logging_dict = {
    "console": {
        "enable": True,
        "level": "INFO"
    },
    "file": {
        "enable": False,
        "level": "INFO"
    }
}
logger = setup_logger(name=__name__, logging_dict=logging_dict)

try:
    from ccblade import CCAirfoil, CCBlade
except ImportError:
    err_msg = ('It appears you do not have CCBlade installed. ' + \
        'Please refer to http://wisdem.github.io/CCBlade/ for ' + \
        'guidance on how to properly install the module.')
    logger.error(err_msg, stack_info=True)
    raise ImportError(err_msg)

# from ccblade import CCAirfoil, CCBlade

from scipy import interpolate


# Some useful constants
degRad = np.pi/180.
rpmRadSec = 2.0*(np.pi)/60.0
base_R = 63. # Actual NREL 5MW radius

# Function returns a scaled NREL 5MW rotor object from CC-Blade
def CCrotor(Rtip=base_R, Rhub=1.5, hubHt=90.0, shearExp = 0.2, rho = 1.225, mu = 1.81206e-5, path_to_af='5MW_AFFiles'):


    r = (Rtip/base_R)*np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
                    28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
                    56.1667, 58.9000, 61.6333])
    chord = (Rtip/base_R)*np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
                        3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
    theta = np.array([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
                        6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
    B = 3  # number of blades

    # In this initial version, hard-code to be NREL 5MW
    afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
    basepath = path_to_af

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = afinit(path.join(basepath, 'Cylinder1.dat'))
    airfoil_types[1] = afinit(path.join(basepath, 'Cylinder2.dat'))
    airfoil_types[2] = afinit(path.join(basepath, 'DU40_A17.dat'))
    airfoil_types[3] = afinit(path.join(basepath, 'DU35_A17.dat'))
    airfoil_types[4] = afinit(path.join(basepath, 'DU30_A17.dat'))
    airfoil_types[5] = afinit(path.join(basepath, 'DU25_A17.dat'))
    airfoil_types[6] = afinit(path.join(basepath, 'DU21_A17.dat'))
    airfoil_types[7] = afinit(path.join(basepath, 'NACA64_A17.dat'))

    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    af = [0]*len(r)
    for i in range(len(r)):
        af[i] = airfoil_types[af_idx[i]]


    tilt = -5.0
    precone = 2.5
    yaw = 0.0

    nSector = 8  # azimuthal discretization

    rotor = CCBlade(r, chord, theta, af, Rhub, Rtip, B, rho, mu,
                    precone, tilt, yaw, shearExp, hubHt, nSector)

    return rotor

# Return the demanded generator torque for a given gen speed
# This is based on the torque controller within SOWFA and using the control parameters within the SOWFA example
def trq_cont(turbine_dict, genSpeedF):
    """
    Compue the torque control at a given gen speed (based on SOWFA)
    """
    #print(genSpeedF,turbine_dict['Region2StartGenSpeed'])
    # Region 1.
    if genSpeedF < turbine_dict['CutInGenSpeed']:
        # print('in region 1...')
        torqueGenCommanded = turbine_dict['CutInGenTorque']
    # # Region 1-1/2.
    elif ((genSpeedF >= turbine_dict['CutInGenSpeed']) and (genSpeedF < turbine_dict['Region2StartGenSpeed'])):
        # print('in region 1.5...')
        dGenSpeed = genSpeedF - turbine_dict['CutInGenSpeed']
        Region2StartGenTorque = turbine_dict['KGen'] * turbine_dict['Region2StartGenSpeed'] * turbine_dict['Region2StartGenSpeed']
        torqueSlope = (Region2StartGenTorque - turbine_dict['CutInGenTorque']) / ( turbine_dict['Region2StartGenSpeed'] - turbine_dict['CutInGenSpeed'] )
        torqueGenCommanded = turbine_dict['CutInGenTorque'] + torqueSlope*dGenSpeed
    # # Region 2.
    elif ((genSpeedF >= turbine_dict['Region2StartGenSpeed']) and (genSpeedF < turbine_dict['Region2EndGenSpeed'])):
        # print('in region 2...')
        torqueGenCommanded = turbine_dict['KGen'] * genSpeedF * genSpeedF
    # # Region 2-1/2.
    elif ((genSpeedF >= turbine_dict['Region2EndGenSpeed']) and (genSpeedF < turbine_dict['RatedGenSpeed'])):
        # print('in region 2.5...')
        dGenSpeed = genSpeedF - turbine_dict['Region2EndGenSpeed']
        Region2EndGenTorque = turbine_dict['KGen'] * turbine_dict['Region2EndGenSpeed'] * turbine_dict['Region2EndGenSpeed']
        torqueSlope = (turbine_dict['RatedGenTorque'] - Region2EndGenTorque) / ( turbine_dict['RatedGenSpeed'] - turbine_dict['Region2EndGenSpeed'] )
        torqueGenCommanded = Region2EndGenTorque + torqueSlope*dGenSpeed
    # # Region 3.
    elif (genSpeedF >= turbine_dict['RatedGenSpeed']):
        # print('in region 3...')
        torqueGenCommanded = turbine_dict['RatedGenTorque']


    # Limit to the rated torque
    torqueGenCommanded = np.min([torqueGenCommanded,turbine_dict['RatedGenTorque']]) 

    return torqueGenCommanded


# Update the PI pitch controller
# This is based on the pitch controller within SOWFA and using the control parameters within the SOWFA example
def pitch_control(turbine_dict, rotSpeedF, pitch_prev, dt,intSpeedError):
    min_pitch = 0.0
    max_pitch = 90.0
    
    # Set the gain scheduling variable.
    GK = 1.0 / (1.0 + (pitch_prev*degRad)/turbine_dict['PitchK'])

    # Store the old value of speed error.
    # speedErrorLast = sped_prev 

    # Compute the low speed shaft speed error.
    speedError = rotSpeedF - turbine_dict['RatedRotSpeed']*rpmRadSec # in rad/s

    # Numerically integrate the speed error over time.
    intSpeedError= intSpeedError + speedError * dt

    # Numerically take the deriviative of speed error w.r.t time.
    #scalar derivSpeedError = (speedError[i] - speedErrorLast) / dt;

    # Saturate the integrated speed error based on pitch saturation.
    intSpeedError = np.max([intSpeedError, min_pitch/(GK*turbine_dict['PitchControlKI'])])
    intSpeedError = np.min([intSpeedError, max_pitch/(GK*turbine_dict['PitchControlKI'])])

    # Compute the pitch components from the proportional, integral,
    # and derivative parts and sum them.
    pitchP = GK * turbine_dict['PitchControlKP'] * speedError
    pitchI = GK * turbine_dict['PitchControlKI']  * intSpeedError
    # scalar pitchD = GK * PitchControlKD[j] * derivSpeedError;
    pitchCommanded = pitchP + pitchI#  + pitchD;

    # Saturate the pitch based on the pitch limits of the pitch
    # actuator.
    pitchCommanded = np.min([np.max([pitchCommanded, min_pitch]), max_pitch])

    # print('pitch commanded = ', pitchCommanded,max_pitch)

    # Return the commanded pitch
    return pitchCommanded, intSpeedError

# Given a controller paramaterized by turbing dict, return a new turbine_dict
# With values scaled according to changes in D and turbine rating (in MW)
def scale_controller_and_rotor(turbine_dict_in, R_In=base_R, turbine_rating=5):

    # Copy the dict
    turbine_dict = copy.deepcopy(turbine_dict_in)

    # Save the R value
    turbine_dict['TipRad'] = R_In

    # Scale the rotation speed inverse to the new radius
    turbine_dict['CutInGenSpeed'] = (base_R/R_In) * turbine_dict['CutInGenSpeed']
    turbine_dict['Region2StartGenSpeed'] = (base_R/R_In) * turbine_dict['Region2StartGenSpeed']
    turbine_dict['Region2EndGenSpeed'] = (base_R/R_In) * turbine_dict['Region2EndGenSpeed']
    turbine_dict['RatedGenSpeed'] = (base_R/R_In) * turbine_dict['RatedGenSpeed']
    turbine_dict['RatedRotSpeed'] = (base_R/R_In) * turbine_dict['RatedRotSpeed']

    # Scale the cut in generator torque (not necessary, this is always 0)
    # turbine_dict['CutInGenTorque'] = (base_R/R_In) * turbine_dict['CutInGenTorque']

    # Scale kGen by the 5th power of radius
    turbine_dict['KGen'] = (R_In/base_R)**5 * turbine_dict['KGen']

    # Scale the rator torque according to the rated speed and power
    turbine_dict['RatedGenTorque'] = (turbine_rating * 1E6) / (turbine_dict['RatedRotSpeed'] * turbine_dict['GBRatio'] * np.pi/30.0 * turbine_dict['GenEfficiency'])

    # Save rating for conviencce
    turbine_dict['RatedMW'] = turbine_rating

    # Get the scaled rotor
    rotor = CCrotor(R_In)

    return turbine_dict, rotor


# Given a controller paramaterization, show the torque curve
def show_torque_curve(turbine_dict, ax,label='_nolegend_'):

    # Based on the details in SOWFA case, show the torque curve
    gen_speed_sweep = np.arange(0,turbine_dict['RatedRotSpeed'] * turbine_dict['GBRatio'],1.)
    gen_torque = np.array([trq_cont(turbine_dict, gf) for gf in gen_speed_sweep])
    # trq_opt = np.array([gf*gf*turbine_dict['KGen'] for gf in gen_speed_sweep])

    # ax.plot(gen_speed_sweep,trq_opt,'k--',label='Optimal')
    ax.plot(gen_speed_sweep,gen_torque,label=label)
    ax.set_xlabel('Gen Speed (RPM)')
    ax.set_ylabel('Gen Torque (Nm)')
    ax.grid(True)
    ax.set_title('Torque Curve')
    ax.legend()

# Generate a set of look-up tables the controller/steady state can use to find a cp/cq/ct
# for a given pitch angle and TSR
def generate_base_lut(rotor, turbine_dict):
    
    # These dicts (keyed on yaw)
    cp_dict = dict()
    ct_dict = dict()
    cq_dict = dict()

    # for now, assume only one yaw angle, perhaps expand later
    yaw = 0.0

    # Mesh the grid and flatten the arrays
    fixed_rpm = 10 # RPM
    Rtip = turbine_dict['TipRad']
    TSR_initial = np.arange(0.5,15,0.5)
    pitch_initial = np.arange(0,25,0.5)
    ws_array = (fixed_rpm * (np.pi / 30.) * Rtip)  / TSR_initial
    ws_mesh, pitch_mesh = np.meshgrid(ws_array, pitch_initial)
    ws_flat = ws_mesh.flatten()
    pitch_flat = pitch_mesh.flatten()
    omega_flat = np.ones_like(pitch_flat) * fixed_rpm
    tsr_flat = (fixed_rpm * (np.pi / 30.) * Rtip)  / ws_flat

    # Get values from cc-blade
    P, T, Q, M, CP, CT, CQ, CM = rotor.evaluate(ws_flat, omega_flat, pitch_flat, coefficients=True)


    # Reshape Cp, Ct and Cq
    CP = np.reshape(CP, (len(pitch_initial), len(TSR_initial)))
    CT = np.reshape(CT, (len(pitch_initial), len(TSR_initial)))
    CQ = np.reshape(CQ, (len(pitch_initial), len(TSR_initial)))

    # # Form the interpolant functions
    cp_interp = interpolate.interp2d(TSR_initial, pitch_initial, CP, kind='cubic')
    ct_interp = interpolate.interp2d(TSR_initial, pitch_initial, CT, kind='cubic')
    cq_interp = interpolate.interp2d(TSR_initial, pitch_initial, CQ, kind='cubic')

    # Add to the dictionaries
    cp_dict[yaw] = cp_interp
    ct_dict[yaw] = ct_interp
    cq_dict[yaw] = cq_interp

    # Save dictionaries
    pickle.dump([cp_dict,ct_dict,cq_dict],open('cp_ct_cq_lut.p','wb'))




def get_aero_torque(rotor,ws,rot_speed,fluidDensity,R,pitch_angle=0.0):
    P, T, Q, M, Cp, Ct, cq, CM = rotor.evaluate([ws], 
                                                    [rot_speed/rpmRadSec], 
                                                    [pitch_angle], 
                                                    coefficients=True)
    # print(cq[0])
    return 0.5 * fluidDensity * (np.pi * R**2) * cq[0] * R * ws**2

# For a given rotor/controller/wind speed, get the steady state value
def get_steady_state(turbine_dict,rotor,ws,dt=0.5,sim_time=5,title=None,show_plot=False):

    # Save some convience terms
    fluidDensity = 1.225 #TODO Get from SOWFA
    R = turbine_dict['TipRad']
    GBRatio = turbine_dict['GBRatio']

    # Determine the drivetrain inertia
    drivetrain_inertia =  turbine_dict['NumBl'] * turbine_dict['BladeIner'] + turbine_dict['HubIner'] + turbine_dict['GBRatio']*turbine_dict['GBRatio']*turbine_dict['GenIner']

    # Simulation parameters
    # sim_length = sim_time/dt

    # Try to determine a good initial rotor speed
    rot_sweep = np.linspace(turbine_dict['CutInGenSpeed'] * rpmRadSec / GBRatio,turbine_dict['RatedRotSpeed'] * rpmRadSec,15)
    gen_sweep = rot_sweep * GBRatio / rpmRadSec
    aero_sweep = np.array( [get_aero_torque(rotor,ws,r_speed,fluidDensity,R) for r_speed in rot_sweep])
    gt_sweep = np.array([trq_cont(turbine_dict,gs) for gs in gen_sweep])
    torque_error = np.abs(aero_sweep * turbine_dict['GBEfficiency'] - GBRatio * gt_sweep)

    # If max exceeded, use max
    if ( np.max(aero_sweep * turbine_dict['GBEfficiency']) > np.max(gt_sweep * GBRatio)):
        init_rotor = turbine_dict['RatedRotSpeed'] * rpmRadSec
    else: # Use the minimum
        idx = np.argmin(torque_error)
        init_rotor = rot_sweep[idx] 

    # Initialize the pitch (if at max RPM)
    if ((init_rotor == rot_sweep[-1] ) or (init_rotor==turbine_dict['RatedRotSpeed'] * rpmRadSec)):
        pitch_sweep = np.linspace(0,20,50)
        aero_sweep = np.array( [get_aero_torque(rotor,ws,init_rotor,fluidDensity,R,pitch_angle=p) for p in pitch_sweep])
        gt_sweep = np.array([trq_cont(turbine_dict,gen_sweep[-1]) for p in pitch_sweep])
        torque_error = np.abs(aero_sweep * turbine_dict['GBEfficiency'] - GBRatio * gt_sweep)
        idx = np.argmin(torque_error)
        init_pitch = pitch_sweep[idx] 

        # And force the intspeed error warm
        GK = 1.0 / (1.0 + (init_pitch*degRad)/turbine_dict['PitchK'])
        intSpeedError = init_pitch / (GK * turbine_dict['PitchControlKI'])


    else:
        init_pitch = 0.0
        # Initialize int speed error as 0
        intSpeedError = 0.0


    #Aero torque assuming pitch is 0


    # Create the arrays
    t_array = nt_array = np.arange(0,sim_time,dt)
    pitch = np.ones_like(t_array) * init_pitch
    rot_speed = np.ones_like(t_array) * init_rotor # represent rot speed in rad / s
    gen_speed = np.ones_like(t_array) * init_rotor * GBRatio / rpmRadSec # represent gen speed in rpm
    aero_torque = np.ones_like(t_array) * 1000.0
    gen_torque = np.ones_like(t_array) * trq_cont(turbine_dict, gen_speed[0])
    gen_power = np.ones_like(t_array) * 0.0
    tsr_array = np.ones_like(t_array) * 0.0
    cq_array = np.ones_like(t_array) * 0.0
    cp_array = np.ones_like(t_array) * 0.0
    ct_array = np.ones_like(t_array) * 0.0


    # Load the Cp,Ct,Cq tables
    cp_dict,ct_dict,cq_dict = pickle.load(open('cp_ct_cq_lut.p','rb'))

    # Select the 0-yaw LUT
    cq_lut = cq_dict[0]

    # Now loop through and get the values
    re_run = True
    max_re_run = 5
    num_re_run = 0
    while(re_run and (num_re_run < max_re_run)):
        for i in range(1,len(t_array)):

            #print('Control time step = ', i, 'out of ', len(t_array))

            # Calculate TSR
            tsr = (R* (rot_speed[i-1]/rpmRadSec) * np.pi/ 30.0) / ws

            # Update the aero torque
            #cq = cq_lut(tsr,pitch[i-1])
            try:
                P, T, Q, M, Cp, Ct, cq, CM = rotor.evaluate([ws], 
                                                        [rot_speed[i-1]/rpmRadSec], 
                                                        [pitch[i-1]], 
                                                        coefficients=True)
            except:
                print('CC BLADE PROBLEM')
                if i > 0:
                    return gen_power[i-1],cp_array[i-1],ct_array[i-1]
                else:
                    print('...no data')
                    return np.nan,np.nan,np.nan
                    
                
            aero_torque[i] = 0.5 * fluidDensity * (np.pi * R**2) * cq * R * ws**2

            # Save these values for plotting
            cq_array[i] = cq
            cp_array[i] = Cp[0]
            ct_array[i] = Ct[0]
            tsr_array[i] = tsr

            # Update the rotor speed and generator speed
            rot_speed[i] = rot_speed[i-1] + (dt/drivetrain_inertia)*(aero_torque[i] * turbine_dict['GBEfficiency'] - GBRatio * gen_torque[i-1])
            gen_speed[i] = rot_speed[i] * GBRatio / rpmRadSec

            # Update the gen torque
            gen_torque[i] = trq_cont(turbine_dict, gen_speed[i])

            # Update the blade pitch
            pitch[i], intSpeedError = pitch_control(turbine_dict, rot_speed[i], pitch[i-1], dt,intSpeedError)

            # Calculate the power
            gen_power[i] = gen_speed[i] * np.pi/30.0 * gen_torque[i] * turbine_dict['GenEfficiency']
        # Determine if need to re_run
        if (gen_power[-1] <= turbine_dict['RatedMW'] * 1e6 * 1.001) and (np.abs(gen_torque[-1] - aero_torque[-1]/GBRatio) < 100):
            re_run = False
        else:
            print('Re Run %s' % title)
            re_run = True
            num_re_run = num_re_run + 1
            pitch[0] = pitch[-1]
            rot_speed[0] = rot_speed[-1]
            gen_speed[0] = gen_speed[-1]
            aero_torque[0] = aero_torque[-1]
            gen_torque[0] = gen_torque[-1]
            gen_power[0] = gen_power[-1]
            tsr_array[0] = tsr_array[-1]
            cq_array[0] = cq_array[-1]

    if show_plot:
        fig, axarr = plt.subplots(5,1,sharex=True)

        if title is not None:
            fig.suptitle(title, fontsize=16)

        ax = axarr[0]
        ax.plot(t_array[1:], tsr_array[1:],label='TSR')
        ax.fill_between(t_array[1:],6,8,color='gray',alpha=0.2)
        ax.legend()
        # ax.set_ylim([5,9])
        ax.grid(True)

        ax = axarr[1]
        ax.plot(t_array[1:],pitch[1:],label='Pitch')
        ax.grid(True)
        ax.legend()

        ax = axarr[2]
        ax.plot(t_array[1:],gen_torque[1:],label='Gen Torque')
        ax.plot(t_array[1:],aero_torque[1:]/GBRatio,label='Aero Torque / GB',color='r')
        ax.grid(True)
        ax.legend()


        ax = axarr[3]
        ax.plot(t_array[1:],rot_speed[1:]/rpmRadSec,label='Rotor Speed (RPM)')
        ax.axhline(turbine_dict['RatedRotSpeed'],color='r',label='Rated')
        ax.grid(True)
        ax.legend()

        ax = axarr[4]
        ax.plot(t_array[1:],gen_power[1:]/1E6,label='Power')
        ax.axhline(turbine_dict['RatedMW'],color='r')
        ax.plot()
        ax.grid(True)
        ax.legend()

    # Return the steady values
    return gen_power[-1],Cp[0],Ct[0]


def get_wind_sweep_steady_values(turbine_dict,rotor,ws_array=np.arange(3,21,1.)):

    # Get the steady values
    pow_array = list()
    cp_array = list()
    ct_array = list()

    for ws in ws_array:
        print(ws)
        p,cp,ct = get_steady_state(turbine_dict,rotor,ws)
        pow_array.append(p)
        cp_array.append(cp)
        ct_array.append(ct)

    return ws_array, np.array(pow_array), np.array(cp_array),np.array(ct_array)