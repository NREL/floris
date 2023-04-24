---
kernelspec:
  name: python3
  display_name: python3
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
    jupytext_version: 1.13.8
---

# Wake Model Comparison

This page will explain the different wake models available in FLORIS and show how the results will
differ between each model.

## Setup

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt

from floris.tools import FlorisInterface
from floris.tools.visualization import (
    calculate_horizontal_plane_with_turbines,
    visualize_cut_plane,
    plot_rotor_values,
    show_plots,
)
```

## Intro to Wake Models

### Wake Model Availability and Explainers

#### Wake Velocity Deflection

#### Gauss
The Gauss deflection model is a blend of the models described in
:cite:`gdm-bastankhah2016experimental` and :cite:`gdm-King2019Controls` for calculating
the deflection field in turbine wakes.

#### Jensen

#### Wake Velocity

#### Wake Turbulence

#### Wake Combination


### Shorthand Model Naming

#### Jensen

#### GCH

#### CC

## Model Setup

```{code-cell}
def get_plot_parameters(fi):
    horizontal_plane = fi.calculate_horizontal_plane(
        x_resolution=200,
        y_resolution=100,
        height=90.0,
        yaw_angles=np.array([[[25.,0.,0.]]]),
    )
    y_plane = fi.calculate_y_plane(
        x_resolution=200,
        z_resolution=100,
        crossstream_dist=630.0,
        yaw_angles=np.array([[[25.,0.,0.]]]),
    )
    cross_plane = fi.calculate_cross_plane(
        y_resolution=100,
        z_resolution=100,
        downstream_dist=630.0,
        yaw_angles=np.array([[[25.,0.,0.]]]),
    )
    return horizontal_plane, y_plane, cross_plane
```

```{code-cell}
solver_settings = {
    "type": "turbine_grid",
    "turbine_grid_points": 10
}

fi_cc = FlorisInterface("../examples/inputs/cc.yaml")
fi_gch = FlorisInterface("../examples/inputs/gch.yaml")
fi_jensen = FlorisInterface("../examples/inputs/jensen.yaml")
# fi_turbopark = FlorisInterface("../examples/inputs/turbopark.yaml")

fi_cc.reinitialize(solver_settings=solver_settings)
fi_gch.reinitialize(solver_settings=solver_settings)
fi_jensen.reinitialize(solver_settings=solver_settings)
# fi_turbopark.reinitialize(solver_settings=solver_settings)

fi_cc.calculate_wake()
fi_gch.calculate_wake()
fi_jensen.calculate_wake()
# fi_turbopark.calculate_wake()
```

```{code-cell}
horizontal_cc, y_cc, cross_cc = get_plot_parameters(fi_cc)
horizontal_gch, y_gch, cross_gch = get_plot_parameters(fi_gch)
horizontal_jensen, y_jensen, cross_jensen = get_plot_parameters(fi_jensen)
# horizontal_turbopark, y_turbopark, cross_turbopark = get_plot_parameters(fi_turbopark)
```

## Plot the Results

In the below plot we demonstrate the horizontal wake profile.

```{code-cell}
fig, ax_list = plt.subplots(3, 1, figsize=(10, 8), dpi=200, sharex=True, sharey=True)
ax_list = ax_list.flatten()

visualize_cut_plane(horizontal_cc, ax=ax_list[0], title="CC", fontsize=12)
visualize_cut_plane(horizontal_gch, ax=ax_list[1], title="GCH", fontsize=12)
visualize_cut_plane(horizontal_jensen, ax=ax_list[2], title="Jensen", fontsize=12)
# horizontal_plane_scan_turbine = calculate_horizontal_plane_with_turbines(
#     fi,
#     x_resolution=20,
#     y_resolution=10,
#     yaw_angles=np.array([[[25.,0.,0.]]]),
# )

fig.suptitle("Horizontal Wake Profile", fontsize=16)
fig.tight_layout()
```

In the below plots we show the turbine grids for each turbine.

```{code-cell}
fig, axes, _ , _ = plot_rotor_values(
    fi_cc.floris.flow_field.u,
    wd_index=0,
    ws_index=0,
    n_rows=1,
    n_cols=3,
    return_fig_objects=True
)
fig.suptitle("Rotor Plane Visualization for CC, 10x10 Resolution")
show_plots()

fig, axes, _ , _ = plot_rotor_values(
    fi_gch.floris.flow_field.u,
    wd_index=0,
    ws_index=0,
    n_rows=1,
    n_cols=3,
    return_fig_objects=True
)
fig.suptitle("Rotor Plane Visualization for GCH, 10x10 Resolution")
show_plots()

fig, axes, _ , _ = plot_rotor_values(
    fi_jensen.floris.flow_field.u,
    wd_index=0,
    ws_index=0,
    n_rows=1,
    n_cols=3,
    return_fig_objects=True
)
fig.suptitle("Rotor Plane Visualization for Jensen, 10x10 Resolution")
show_plots()
```
