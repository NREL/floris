# API Documentation

FLORIS is divided into two primary packages.
{py:mod}`floris.simulation` is the core code that models the wind turbines
and wind farms. It is low-level code that generally is not accessed
by typical users. {py:mod}`floris.tools` is the set of analysis routines
that define, drive, and post process a simulation. This is where
more users will interface with the software.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   floris.flow_visualization
   floris.floris_model
   floris.wind_data
   floris.uncertain_floris_model
   floris.turbine_library
   floris.parallel_floris_model
   floris.optimization
   floris.layout_visualization
   floris.cut_plane
   floris.core
   floris.convert_turbine_v3_to_v4
   floris.convert_floris_input_v3_to_v4
   floris.utilities
   floris.type_dec
   floris.logging_manager
```
