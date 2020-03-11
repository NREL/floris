.. _theory:

Theory Reference
----------------

FLORIS implements a 3D version of the Jensen, original FLORIS (Gebraad et. al.
2016), Gaussian, and Curl wake model.

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
   
6. Martinez-Tossas, L. A., Annoni, J., Fleming, P. A., and Churchfield, M. J.: 
   The aerodynamics of the curled wake: a simplified model in view of flow control, 
   Wind Energ. Sci., 4, 127–138, 2019.
 


Curled Wake Model
===============
The curled wake model solves a linearized version of the 
Reynolds-averaged Navier-Stokes equations to obtain
the wake velocity deficit.
This is the computationally most expensive option (1-10 seconds)
in the FLORIS
framework because it solves a parabolic system of equations.
This model can be used when accurate modeling of yawed wakes 
is of interest
and comptutational expense is not a priority.

Gauss Wake Model
================
Describe the gauss wake model here

Multi Zone Wake Model
=====================
Describe the multi zone wake model here

Jensen
======
Describe the jensen wake model here

Jimenez Wake Model
==================
Describe the jimenez wake model here

Other models to describe
========================
possible other models to describe
- Flow field initialization ... boundary layer/log law?
- combination schemes
