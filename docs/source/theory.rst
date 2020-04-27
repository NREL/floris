.. _theory:

Theory Reference
----------------
The wake velocity deficit, deflection, and turbulence models implemented in
the FLORIS framework are listed below along with accompanying publications.

Jensen
======
The Jensen wake model is defined in :cite:`thy-jensen1983note`.

Multi Zone Wake Model
=====================
The multi-zone wake model is defined in :cite:`thy-gebraad2014data`,
:cite:`thy-gebraad2016wind`.

Gaussian Wake Models
====================
Several Gaussian wake models are now implemented within FLORIS.
Literature on the Gaussian model can be found in
:cite:`thy-bastankhah2014new`,
:cite:`thy-abkar2015influence`,
:cite:`thy-niayifar2016analytical`,
:cite:`thy-bastankhah2016experimental`,
:cite:`thy-dilip2017wind`,
:cite:`thy-blondel2020alternative`
and :cite:`thy-qian2018new`.

Gauss Curl Hybrid Model
=======================

The Gauss-Curl-Hybrid model combines Gaussian wake models to capture
second-order effects of wake steering using curl-based methods, as
described in :cite:`thy-King2019Controls`.

Jimenez Wake Model
==================
The Jimenez model of wake deflection is defined in
:cite:`thy-jimenez2010application`.

Curled Wake Model
=================
The curled wake model solves a linearized version of the Reynolds-averaged
Navier-Stokes equations to obtain the wake velocity deficit. This is the
computationally most expensive option (1-10 seconds) in the FLORIS
framework because it solves a parabolic system of equations.
This model can be used when accurate modeling of yawed wakes 
is of interest and comptutational expense is not a priority.
See :cite:`thy-martinez2019aerodynamics` for a full description.

References:
===========
   .. bibliography:: /source/zrefs.bib
      :style: unsrt
      :filter: docname in docnames
      :keyprefix: thy-