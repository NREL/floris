"""Provides the plotting of individual rotor values across a variety of scenarios"""
from __future__ import annotations

import math
from itertools import product

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_rotor_values(
    values: np.ndarray,
    titles: np.ndarray,
    max_width: int = 4,
    cmap: str = "coolwarm",
    return_fig_objects: bool = False,
    save_path: str | None = None,
) -> None | tuple[plt.figure, plt.axes]:
    """Plots the gridded turbine rotor values. This is intended to be used for
    understanding the differences between two sets of values, so each subplot can be
    used for inspection of what values are differing, and under what conditions.

    Parameters:
        values (np.ndarray): The 5-dimensional array of values to plot. Should be:
            N wind directions x N wind speeds x N turbines X N rotor points X N rotor points.
        titles (np.ndarray): The string values to label each plot, and should be of the
            same shape as `values`.
        max_width (int): The maximum number of subplots in one row, default 4.
        cmap (str): The matplotlib colormap to be used, default "coolwarm".
        return_fig_objects (bool): Indicator to return the primary figure objects for
            further editing, default False.
        save_path (str | None): Where to save the figure, if a value is provided.

    Returns:
        None | tuple[plt.figure, plt.axes, plt.axis, plt.colorbar]: If
        `return_fig_objects` is `False, then `None` is returned`, otherwise the primary
        figure objects are returned for custom editing.
    """

    cmap = plt.cm.get_cmap(name=cmap)

    n_wd, n_ws, *_ = values.shape

    vmin = values.min()
    vmax = values.max()

    rounded_vmin = round(math.floor(vmin) * 2) / 2
    if vmin % 1 >= 0.5:
        rounded_vmin += 0.5

    rounded_vmax = round(math.ceil(vmax) * 2) / 2
    if vmax % 1 <= 0.5:
        rounded_vmax -= 0.5

    bounds = np.linspace(rounded_vmin, rounded_vmax, 11)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    n_plots = n_wd * n_ws
    extra = 0
    if n_plots <= max_width:
        nrows, ncols = 1, n_plots
    else:
        nrows, extra = divmod(n_plots, max_width)
        if extra > 0:
            nrows += 1
            extra = max_width - extra
        ncols = max_width

    fig = plt.figure(dpi=200, figsize=(16, 16))
    axes = fig.subplots(nrows, ncols)

    for ax, t, (i, j) in zip(
        axes.flatten(), titles.flatten(), product(range(n_wd), range(n_ws))
    ):
        ax.imshow(values[i, j], cmap=cmap, norm=norm)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(t)

    if extra > 0:
        for ax in axes[-1][-extra:]:
            fig.delaxes(ax)

    cbar_ax = fig.add_axes([0.05, 0.125, 0.03, 0.75])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if return_fig_objects:
        return fig, axes, cbar_ax, cb
