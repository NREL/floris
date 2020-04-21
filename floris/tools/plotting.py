# Copyright 2020 NREL
 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
 
# See https://floris.readthedocs.io for documentation
 

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class PlotDefaults():
    """
    This class sets journal-ready styles for plots.
    """

    def __init__(self):

        sns.set_style("ticks")
        sns.set_context("paper", font_scale=1.5)

        # color palette from colorbrewer (up to 4 colors, good for print and black&white printing)
        # color_brewer_palette = ['#e66101', '#5e3c99', '#fdb863', '#b2abd2']

        #most journals: 300dpi
        plt.rcParams['savefig.dpi'] = 300

        #most journals: 9 cm (or 3.5 inch) for single column width and 18.5 cm (or 7.3 inch) for double column width.
        plt.rcParams['figure.autolayout'] = False
        plt.rcParams['figure.figsize'] = 7.3, 4
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['font.size'] = 32
        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['lines.markersize'] = 8
        plt.rcParams['legend.fontsize'] = 14


def data_plot(x,
              y,
              color='b',
              label='_nolegend_',
              x_bins=None,
              x_radius=None,
              ax=None,
              show_scatter=True,
              show_bin_points=True,
              show_confidence=True,
              min_vals=1,
              seaborn=False,
              show_80=False):
    """
    Plot data to a single axis (no subfigrures). Method includes flags
    to provide additional statistical context in plot (e.g. scatter,
    confidnece, etc.)

    Args:
        x (np.array): abscissa data.
        y (np.array): ordinate data.
        color (str, optional): line color.
            Defaults to 'b'.
        label (str, optional): line label used in legend.
            Defaults to '_nolegend_'.
        x_bins (np.array, optional): bin limits for abscissa data.
            Defaults to None.
        x_radius (float, optional): bin width.
            Defaults to None.
        ax (:py:class:`matplotlib.pyplot.axes`, optional):
            axes handle for plotting. Defaults to None.
        show_scatter (bool, optional): flag to control scatter plot.
            Defaults to True.
        show_bin_points (bool, optional): flag to control plot of bins.
            Defaults to True.
        show_confidence (bool, optional): flag to control plot of
            confidence interval. Defaults to True.
        min_vals (int, optional): minimum number of values required to
            merit plotting. Defaults to 1.
        seaborn (bool, optional): flag to control plotting library.
            Defaults to False.
        show_80 (bool, optional): flag to control plot of points above
            the 80th percentile. Defaults to False.
            #TODO generalize to show_percentile?

    Returns:
    Only returns values if `show_confidence` flag is active (True),
    otherwise returns (np.nan).

        x_bins (np.array): bin limits
        median_vals (np.array): median values of data in bins
        lower (np.array): lower limit of data in bins
        upper (np.array): upper limit of data in bins
    """
    if (not ax) and (not seaborn):
        fig, ax = plt.subplots()

    if seaborn:
        show_bin_points = False
        show_scatter = False

    df = pd.DataFrame({'x': x, 'y': y})

    if df.shape[0] > 0:

        # If bins not provided, just use ints
        if x_bins is None:
            x_bins = np.arange(df['x'].astype(int).min(),
                               df['x'].astype(int).max(), 1)

        # if no radius provided, use bins to determine
        if x_radius is None:
            x_radius = (x_bins[1] - x_bins[0]) / 2.0

        # now loop over bins and determine stats
        median_vals = np.zeros_like(x_bins) * np.nan
        # median_vals = np.zeros_like(x_bins) * np.nan
        count_vals = np.zeros_like(x_bins) * np.nan
        lower = np.zeros_like(x_bins) * np.nan
        upper = np.zeros_like(x_bins) * np.nan
        vals_80_up = np.zeros_like(x_bins) * np.nan
        vals_80_down = np.zeros_like(x_bins) * np.nan
        # p_down_vals = np.zeros_like(x_bins) * np.nan

        for x_idx, x_cent in enumerate(x_bins):

            df_sub = df[(df.x >= x_cent - x_radius)
                        & (df.x <= x_cent + x_radius)]

            #TODO this conditional statement contains a lot of stuff to be cleaned up. Why all the commented content?
            if df_sub.shape[0] > min_vals:

                # Get statistics via bootstrapping
                n_bs = 40
                boot_frac = 1.0
                med_array = np.zeros(n_bs)
                for i_bs in range(n_bs):
                    # Random subset the df
                    df_rand = df_sub.sample(frac=boot_frac, replace=True)
                    # med_array[i_bs] = np.median(df_rand.y)
                    med_array[i_bs] = np.mean(df_rand.y)

                # median_vals[x_idx] = np.nanmedian(df_sub.y)
                median_vals[x_idx] = np.mean(df_sub.y)
                vals_80_down[x_idx], vals_80_up[x_idx] = np.percentile(
                    df_sub.y, [50 + 0.5 * 80., 50 - 0.5 * 80.])
                # mean_vals[x_idx] = np.median(ratio_array)
                count_vals[x_idx] = df_sub.shape[0]
                # ci_vals[x_idx] = scipy.stats.sem(ratio_array, ddof=1) * 1.96 # df_sub.y.apply(lambda x: scipy.stats.sem(x, ddof=1) * 1.96)
                # p_up_vals[x_idx] = p_up_func(ratio_array)# df_sub.y.apply(p_up_func)
                # p_down_vals[x_idx] = p_down_func(ratio_array)#df_sub.y.apply(p_down_func)
                # Get the confidence bounds
                confidence = 95
                conf_bounds = [50 + 0.5 * confidence, 50 - 0.5 * confidence]
                # lower[x_idx], upper[x_idx] = (2*med_vals[x_idx]-np.percentile(med_array, conf_bounds))
                lower[x_idx], upper[x_idx] = np.percentile(
                    med_array, conf_bounds)

        # # Plot the underlying points
        if show_scatter:
            ax.scatter(df['x'],
                       df['y'],
                       color=color,
                       label='_nolegend_',
                       alpha=1.0,
                       s=35,
                       marker='.')
        if show_bin_points:
            ax.scatter(x_bins,
                       median_vals,
                       color=color,
                       s=count_vals,
                       label='_nolegend_',
                       alpha=0.6,
                       marker='s')
        if show_80:
            ax.plot(x_bins,
                    vals_80_down,
                    '--',
                    color=color,
                    label='_nolegend_')
            ax.plot(x_bins, vals_80_up, '--', color=color, label='_nolegend_')

        # Plot the main trend
        if not seaborn:
            ax.plot(x_bins, median_vals, label=label, color=color)
        else:
            plt.plot(x_bins, median_vals, label=label, color=color)

        if show_confidence:
            if not seaborn:
                ax.fill_between(x_bins,
                                lower,
                                upper,
                                alpha=0.2,
                                color=color,
                                label='_nolegend_')
            else:
                plt.fill_between(x_bins,
                                 lower,
                                 upper,
                                 alpha=0.2,
                                 color=color,
                                 label='_nolegend_')

        return x_bins, median_vals, lower, upper

    else:
        ax.plot(0, 0, label=label, color=color)

        return np.nan, np.nan, np.nan, np.nan


def stacked_plot(x, groups, x_bins, ax, color_array=None):
    """
    Plot stacked histograms of data according to specified groups.

    Args:
        x (np.array): abscissa data.
        groups (list): groups of data provided by pd.Groupby()
            #TODO right?
        x_bins (np.array, optional): bin limits for abscissa data.
            Defaults to None.
        ax (:py:class:`matplotlib.pyplot.axes`, optional):
            axes handle for plotting. Defaults to None.
        color_array (list, optional): list of color specifiers.
            Defaults to None.
    """

    x_radius = (x_bins[1] - x_bins[0]) / 2.0

    # ind = np.arange(len(x_bins))

    group_vals = np.unique(groups)
    num_groups = len(group_vals)

    p_array = np.zeros((num_groups, len(x_bins)))

    for x_idx, x_cent in enumerate(x_bins):

        x_mask = (x >= x_cent - x_radius) \
                    & (x < x_cent + x_radius)

        # y_bin = y[x_mask]
        g_bin = groups[x_mask]
        num_points = len(g_bin)

        if num_points > 0:
            for g_idx, g in enumerate(group_vals):
                p_array[g_idx, x_idx] = np.sum(
                    g_bin == g)  # / float(num_points)
    p = list()

    if not color_array is None:
        p.append(
            ax.bar(x_bins,
                   p_array[0, :],
                   width=x_radius * 1.5,
                   color=color_array[0]))
    else:
        p.append(ax.bar(x_bins, p_array[0, :], width=x_radius * 1.5))

    for g_idx in range(1, num_groups):
        if not color_array is None:
            p.append(
                ax.bar(x_bins,
                       p_array[g_idx, :],
                       bottom=p_array[g_idx - 1, :],
                       width=x_radius * 1.5,
                       color=color_array[g_idx]))
        else:
            p.append(
                ax.bar(x_bins,
                       p_array[g_idx, :],
                       bottom=p_array[g_idx - 1, :],
                       width=x_radius * 1.5))
    #ax.set_xticks(ind,x_bins)
    ax.legend(group_vals)
    # return group_vals,p_array


def stacked_percent_plot(x, groups, x_bins, ax, color_array=None):
    """
    Plot stacked percentage plot (normalized stacked histogram)
    according to specified groups.

    Args:
        x (np.array): abscissa data.
        groups (list): groups of data provided by pd.Groupby()
            #TODO right?
        x_bins (np.array, optional): bin limits for abscissa data.
            Defaults to None.
        ax (:py:class:`matplotlib.pyplot.axes`, optional):
            axes handle for plotting. Defaults to None.
        color_array (list, optional): list of color specifiers.
            Defaults to None.
    """
    x_radius = (x_bins[1] - x_bins[0]) / 2.0
    # ind = np.arange(len(x_bins))

    group_vals = np.unique(groups)
    num_groups = len(group_vals)

    p_array = np.zeros((num_groups, len(x_bins)))

    for x_idx, x_cent in enumerate(x_bins):

        x_mask = (x >= x_cent - x_radius) \
                    & (x < x_cent + x_radius)

        # y_bin = y[x_mask]
        g_bin = groups[x_mask]
        num_points = len(g_bin)

        if num_points > 0:
            for g_idx, g in enumerate(group_vals):
                p_array[g_idx, x_idx] = np.sum(g_bin == g) / float(num_points)
    p = list()

    if not color_array is None:
        p.append(
            ax.bar(x_bins,
                   p_array[0, :],
                   width=x_radius * 1.5,
                   color=color_array[0]))
    else:
        p.append(ax.bar(x_bins, p_array[0, :], width=x_radius * 1.5))
    for g_idx in range(1, num_groups):
        if not color_array is None:
            p.append(
                ax.bar(x_bins,
                       p_array[g_idx, :],
                       bottom=p_array[g_idx - 1, :],
                       width=x_radius * 1.5,
                       color=color_array[g_idx]))
        else:
            p.append(
                ax.bar(x_bins,
                       p_array[g_idx, :],
                       bottom=p_array[g_idx - 1, :],
                       width=x_radius * 1.5))
    #ax.set_xticks(ind,x_bins)
    ax.legend(group_vals, bbox_to_anchor=(1.0, 1.0))
    # return group_vals,p_array