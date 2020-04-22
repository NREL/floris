.. _theory:

Theory Reference
----------------

FLORIS implements a 3D version of the Jensen, original FLORIS (Gebraad et. al.
2016), Gaussian, and Curl wake model.

Literature on the Gaussian model can be found in :cite:`bastankhah2014new,abkar2015influence,niayifar2016analytical,bastankhah2016experimental,dilip2017wind` and :cite:`martinez2019aerodynamics`.

.. bibliography:: refs.bib
   :cited:
   :style: unsrt
   :filter: docname in docnames



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
