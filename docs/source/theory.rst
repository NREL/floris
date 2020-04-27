.. _theory:

Theory Reference
----------------

FLORIS implements a 3D version of the Jensen, original FLORIS (Gebraad et. al.
2016), Gaussian, Curl and Gauss-Curl-Hybrid wake models.

 
Jensen
======
The Jensen wake model is defined in:
1. Jensen, N. O. (1983). A note on wind generator interaction. Risø
        National Laboratory.

Multi Zone Wake Model
=====================
The multi-zone wake model is defined in:

1. Gebraad, P. M. O. et al., "A Data-driven model for wind plant power
   optimization by yaw control." *Proc. American Control Conference*,
   Portland, OR, 2014.

2. Gebraad, P. M. O. et al., "Wind plant power optimization through
   yaw control using a parametric model for wake effects - a CFD
   simulation study." *Wind Energy*, 2016.


Gaussian Wake Models
====================
Several gaussian wake models are now implemented within FLORIS.  

Literature on the Gaussian model can be found in the following papers:

1. Niayifar, A. and Porté-Agel, F.: A new analytical model for wind farm
   power prediction, in: Journal of Physics: Conference Series, vol. 625,
   012039, IOP Publishing, 2015.

2. Dilip, D. and Porté-Agel, F.: Wind Turbine Wake Mitigation through Blade
   Pitch Offset, Energies, 10, 757, 2017.

3. Abkar, M. and Porté-Agel, F.: Influence of atmospheric stability on
   wind-turbine wakes: A large-eddy simulation study, Physics of Fluids,
   27, 035 104, 2015.

4. Bastankhah, M. and Porté-Agel, F.: A new analytical model for
   wind-turbine wakes, Renewable Energy, 70, 116–123, 2014.

5. Bastankhah, M. and Porté-Agel, 5 F.: Experimental and theoretical study of
   wind turbine wakes in yawed conditions, Journal of FluidMechanics, 806,
   506–541, 2016.
   
6. Blondel, F. and Cathelain, M. "An alternative form of the
   super-Gaussian wind turbine wake model." *Wind Energy Science
   Disucssions*, 2020.

7. Ishihara, Takeshi, and Guo-Wei Qian. "A new Gaussian-based
   analytical wake model for wind turbines considering ambient turbulence
   intensities and thrust coefficient effects." *Journal of Wind
   Engineering and Industrial Aerodynamics* 177 (2018): 275-292.

Gauss Curl Hybrid Model
=======================

The Gauss-Curly-Hybrid model combines with gaussian wake models to model
second-order effects of wake steering using curl-based methods

1. King, J., Fleming, P., King, R., Martínez-Tossas, L. A., Bay, C. J,
   Mudafort, R., and Simley, E.: Controls-Oriented Model for Secondary
   Effects of Wake Steering, *Wind Energ. Sci. Discuss.*, 
   https://doi.org/10.5194/wes-2020-3, in review, 2020.

Jimenez Wake Model
==================
Describe the jimenez model of wake deflection is defined in:

1. Jiménez, Ángel, Antonio Crespo, and Emilio Migoya. "Application of
        a LES technique to characterize the wake deflection of a wind turbine
        in yaw." Wind energy 13.6 (2010): 559-572.


Curled Wake Model
=================
The curled wake model solves a linearized version of the 
Reynolds-averaged Navier-Stokes equations to obtain
the wake velocity deficit.
This is the computationally most expensive option (1-10 seconds)
in the FLORIS
framework because it solves a parabolic system of equations.
This model can be used when accurate modeling of yawed wakes 
is of interest
and comptutational expense is not a priority.

1. Martinez-Tossas, L. A., Annoni, J., Fleming, P. A., and Churchfield, M. J.: 
   The aerodynamics of the curled wake: a simplified model in view of flow control, 
   Wind Energ. Sci., 4, 127–138, 2019.

Other models to describe
========================
possible other models to describe
- Flow field initialization ... boundary layer/log law?
- combination schemes
