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
import pandas as pd
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

#### Wake Deflection

#### Gauss
The Gauss deflection model is a blend of the models described in
{cite:t}`bastankhah2016experimental` and {cite:t}`King2019Controls` for calculating
the deflection field in turbine wakes.

#### Jimenez

Jiménez wake deflection model, dervied from {cite:t}`jimenez2010application`.

#### Wake Velocity

##### Jensen

The Jensen model computes the wake velocity deficit based on the classic Jensen/Park model
{cite:t}`jensen1983note`.

##### Gauss

The Gaussian velocity model is implemented based on {cite:t}`bastankhah2016experimental` and
{cite:t}`niayifar2016analytical`

##### Cumulative Curl (CC)

The cumulative curl model is an implementation of the model described in {cite:t}`gdm-bay_2022`,
which itself is based on the cumulative model of  {cite:t}`bastankhah_2021`

##### TurbOPark

Model based on the TurbOPark model. For model details see https://github.com/OrstedRD/TurbOPark,
https://github.com/OrstedRD/TurbOPark/blob/main/TurbOPark%20description.pdf, and Nygaard, Nicolai
Gayle, et al. "Modelling cluster wakes and wind farm blockage." 2020.

#### Wake Turbulence

##### CrespoHernandez

CrespoHernandez is a wake-turbulence model that is used to compute additional variability introduced
to the flow field by operation of a wind turbine. Implementation of the model follows the original
formulation and limitations outlined in {cite:t}`crespo1996turbulence`.

#### Wake Combination

##### FLS

FLS uses freestream linear superposition to apply the wake velocity deficits to the freestream
flow field.

##### MAX

MAX uses the maximum wake velocity deficit to add to the base flow field. For more information,
refer to {cite:t}`gunn2016limitations`.

##### SOSFS

SOSFS uses sum of squares freestream superposition to combine the wake velocity deficits to the base
flow field.

### Shorthand Model Naming

#### Jensen

The Jensen model uses the Jensen velocity model, Jimenez deflection model, CrespoHernandez
turbulence model, and SOSFS combination model.

#### GCH

The Gauss-Curl-Hybrid model combines Gaussian wake models to capture second-order effects of wake
steering using curl-based methods, as described in {cite:t}`King2019Controls`

#### CC

The CC model uses the Cumulative Curl velocity model, CrespoHernandez turbulence model, the Gaussian
deflection model, and the SOSFS combination model.

## Model Setup

Here, we'll define a method to consistently calculate the varying cut planes for plotting later on.
We'll also set up each of the three models that we'll compare to highlight the differences in how
they operate.

```{code-cell}
def get_plot_parameters(fi):
    """Calculates the horizontal, streamwise, and rotor planes for a given FlorisInterface (fi)
    object.
    """

    horizontal_plane = fi.calculate_horizontal_plane(
        x_resolution=200,
        y_resolution=100,
        height=90.0,
        yaw_angles=np.array([[[25.,0.,0., 25.,0.,0., 25.,0.,0.]]]),
    )
    y_plane = fi.calculate_y_plane(
        x_resolution=200,
        z_resolution=100,
        crossstream_dist=630.0,
        # yaw_angles=np.array([[[25.,0.,0.],[25.,0.,0.],[25.,0.,0.]]]),
    )
    cross_plane = fi.calculate_cross_plane(
        y_resolution=100,
        z_resolution=100,
        downstream_dist=630.0,
        # yaw_angles=np.array([[[50.,0.,0.],[50.,0.,0.],[50.,0.,0.]]]),
    )
    return horizontal_plane, y_plane, cross_plane
```

```{code-cell}
# Use a more granular grid than the default
# Note: only a 3x3 rotor grid can be initialized
solver_settings = {
    "type": "turbine_grid",
    "turbine_grid_points": 10
}

# Update the layout to be a staggered 3x3 turbine grid
layout_x = np.arange(3) * np.ones((3, 3)) * 700
layout_x[1] += 250
layout_x[2] += 500
layout_x = layout_x.flatten()

layout_y = (np.arange(0, 601, 300).reshape(-1, 1) * np.ones((3, 3))).flatten()

# Define each of the CC, GCH, and Jensen models
fi_cc = FlorisInterface("../examples/inputs/cc.yaml")
fi_gch = FlorisInterface("../examples/inputs/gch.yaml")
fi_jensen = FlorisInterface("../examples/inputs/jensen.yaml")

# Reinitialize with the more granular grid and new layouts
fi_cc.reinitialize(
    solver_settings=solver_settings, layout_x=layout_x, layout_y=layout_y
)
fi_gch.reinitialize(
    solver_settings=solver_settings, layout_x=layout_x, layout_y=layout_y
)
fi_jensen.reinitialize(
    solver_settings=solver_settings, layout_x=layout_x, layout_y=layout_y
)
```

```{code-cell}
# Run the models
fi_cc.calculate_wake()
fi_gch.calculate_wake()
fi_jensen.calculate_wake()

# Get the cut planes
horizontal_cc, y_cc, cross_cc = get_plot_parameters(fi_cc)
horizontal_gch, y_gch, cross_gch = get_plot_parameters(fi_gch)
horizontal_jensen, y_jensen, cross_jensen = get_plot_parameters(fi_jensen)
```

## Impacts on Performance
### Plot the Results

In the below plot we demonstrate the horizontal wake profile.

```{code-cell}
fig = plt.figure(figsize=(10, 8), dpi=200)
ax_list = fig.subplots(3, 1, sharex=True, sharey=True).flatten()

visualize_cut_plane(horizontal_cc, ax=ax_list[0])
visualize_cut_plane(horizontal_gch, ax=ax_list[1])
visualize_cut_plane(horizontal_jensen, ax=ax_list[2])

ax_list[0].set_title("CC", fontsize=12)
ax_list[1].set_title("GCH", fontsize=12)
ax_list[2].set_title("Jensen", fontsize=12)

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
fig.suptitle("CC Rotor Plane Visualization, 10x10 Resolution")
show_plots()

fig, axes, _ , _ = plot_rotor_values(
    fi_gch.floris.flow_field.u,
    wd_index=0,
    ws_index=0,
    n_rows=1,
    n_cols=3,
    return_fig_objects=True
)
fig.suptitle("GCH Rotor Plane Visualization, 10x10 Resolution")
show_plots()

fig, axes, _ , _ = plot_rotor_values(
    fi_jensen.floris.flow_field.u,
    wd_index=0,
    ws_index=0,
    n_rows=1,
    n_cols=3,
    return_fig_objects=True
)
fig.suptitle("Jensen Rotor Plane Visualization, 10x10 Resolution")
show_plots()
```

### AEP Comparison

```{code-cell}
# Gather the AEPs and display in a table
frequency = np.array([[1.0]])
results_df = pd.DataFrame(
    [
        ["CC", fi_cc.get_farm_AEP(frequency)],
        ["GCH", fi_gch.get_farm_AEP(frequency)],
        ["Jensen", fi_jensen.get_farm_AEP(frequency)],
    ],
    columns=["Model", "AEP (GWh)"],
).set_index("Model") / 1e6

# Compute the potential (same for all models)
potential = fi_cc.get_farm_AEP(frequency, no_wake=True) / 1e6

results_df.loc[:, "Wake Losses (GWh)"] = np.full(3, potential) - results_df["AEP (GWh)"]
results_df.style.format(precision=2, thousands=",")
```

```{bibliography}
```
