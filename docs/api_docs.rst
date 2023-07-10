API Documentation
=================

FLORIS is divided into two primary packages.
:py:mod:`floris.simulation` is the core code that models the wind turbines
and wind farms. It is low-level code that generally is nto accessed
by typical users. :py:mod:`floris.tools` is the set of analysis routines
that define, drive, and post process a simulation. This is where
more users will interface with the software.

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   floris.logging_manager
   floris.simulation
   floris.tools
   floris.type_dec
   floris.turbine_library
   floris.utilities
