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
import matplotlib.pyplot as plt
import pandas as pd

def gaussian(x, mu, sig):
    """
    Compute gaussian function, from
    https://stackoverflow.com/questions/14873203/plotting-of-1-dimensional-gaussian-distribution-function.

    Args:
        x (np.array): Input variable to Gaussian.
        mu (float): Mean value.
        sig (float): Standard deviation.

    Returns:
        np.array: The resulting Gaussian distribution.
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def _convert_to_numpy_array(series):
    """
    Convert an input series to NumPy array. Currently, this function
    checks if an object has a `values` attribute and returns that if it does.
    Otherwise, it returns the given input if that input is a `np.ndarray`.

    Args:
        series (pd.Series): Series to convert.

    Returns:
        np.array: Converted Series.
    """
    if hasattr(series, 'values'):
        return series.values
    elif isinstance(series, np.ndarray):
        return series


def _calculate_bootstrap_iterations(n):
    """
    Calculate number of bootstrap iterations given length.
    # TODO: What are `bootstrap iterations`?

    Args:
        n (int): Number of points.

    Returns:
        int: Number of bootstrap iterations.
    """
    maximum = 10000
    minimum = 2000
    return int(np.round(max(min(n * np.log10(n), maximum), minimum)))


def _calculate_lower_and_upper_bound(bootstrap_array,
                                     percentiles,
                                     central_estimate=None,
                                     method='simple_percentile'):
    """
    Given resultant bootstrap output array, compute lower and upper bound 
    of confidence interval.

    Args:
        bootstrap_array (np.array): array of bootrapped results
            percentiles (np.array): percentile values
            central_estimate (float, optional): if not using simple percentile,
            need to provide the central estimated result. Defaults to None.
        method (str, optional): method for computing bounds. Defaults to
            'simple_percentile'.

    Returns:
        float, float: 
        
            -   lower ci bound
            -   upper ci bound
    """
    if method is 'simple_percentile':
        upper, lower = np.percentile(bootstrap_array, percentiles)
    else:
        lower, upper = (2 * central_estimate -
                        np.percentile(bootstrap_array, percentiles))
    return lower, upper


def _get_confidence_bounds(confidence):
    """

    Get the upper and lower confidence bounds given a desired confidence level.

    Args:
        confidence (float): [description]
        # TODO: ^^

    Returns:
        float, float: 
        
            -   upper confidence bound
            -   lower confidence bound
    """
    return [50 + 0.5 * confidence, 50 - 0.5 * confidence]



def energy_ratio(ref_pow_base, test_pow_base):
    """
    Compute the balanced energy ratio for a single binned wind direction

    Single version, not comparing controller on controller off

    This function is called to compute a single balanced 
    energy ratio calculation for a particular wind direction bin.  Note 
    the reference turbine should not be the turbine implementing 
    control, but should be an unaffected nearby turbine, or a synthetic 
    power estimate from a measurement.  See :cite:`ers-fleming2019initial` and 
    :cite:`ers-fleming2019continued` for more information.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: ers-

    Args:
        ref_pow_base (np.array): Array of baseline reference turbine 
            power.
        test_pow_base (np.array): Array of baseline test turbine power.


    Returns:
        tuple: tuple containing:

            -   **ratio_base** (*float*): Baseline energy ratio.
            -   **counts_base** (*float*): Number of points in baseline.
    """


    if len(ref_pow_base) == 0:
        return np.nan, np.nan


    # Weighted sums
    weight_sum_ref_base = np.sum(ref_pow_base)
    weight_sum_test_base = np.sum(test_pow_base)

    if ((weight_sum_ref_base==0) or (weight_sum_test_base==0)):
        return np.nan, np.nan


    ratio_base = weight_sum_test_base / weight_sum_ref_base


    # Get the counts
    counts_base = len(ref_pow_base)


    return ratio_base, counts_base


def calculate_balanced_energy_ratio(reference_power_baseline,
                                    test_power_baseline,
                                    wind_direction_array_baseline,
                                    wind_direction_bins,
                                    confidence=95,
                                    n_boostrap=None,
                                    wind_direction_bin_p_overlap=None,
                                    ):
    """
    Calculate a balanced energy ratio for each wind direction bin.

    Single version, not divided into baseline and controlled

    Calculate a balanced energy ratio for each wind direction bin.  A 
    reference and test turbine are provided for the ratio, as well as 
    wind speed and wind directions. These data are further divided into 
    baseline and controlled conditions.  The balanced energy ratio 
    function is called and used to ensure a similar distribution of 
    wind speeds is used in the computation, per wind direction bin, for 
    baseline and controlled results.  Resulting arrays, including upper 
    and lower uncertainty bounds computed through bootstrapping, are 
    returned.  Note the reference turbine should not be the turbine 
    implementing control, but should be an unaffected nearby turbine, 
    or a synthetic power estimate from a measurement  See 
    :cite:`ers-fleming2019initial` and 
    :cite:`ers-fleming2019continued` for more information.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: ers-

    Args:
        reference_power_baseline (np.array): Array of power of 
            reference turbine in baseline conditions.
        test_power_baseline (np.array): Array of power of test turbine 
            in baseline conditions.
        wind_speed_array_baseline (np.array): Array of wind speeds in 
            baseline conditions.
        wind_direction_array_baseline (np.array): Array of wind 
            directions in baseline case.
        wind_direction_bins (np.array): Wind directions bins.
        confidence (int, optional): Confidence level to use.  Defaults 
            to 95.
        n_boostrap (int, optional): Number of bootstaps, if none, 
            _calculate_bootstrap_iterations is called.  Defaults to 
            None.
        wind_direction_bin_p_overlap (np.array, optional): Percentage 
            overlap between wind direction bin. Defaults to None.

    Returns:
        tuple: tuple containing:

            **ratio_array_base** (*np.array*): Baseline energy ratio at each wind direction bin.
            **lower_ratio_array_base** (*np.array*): Lower confidence bound of baseline energy ratio at each wind direction bin.
            **upper_ratio_array_base** (*np.array*): Upper confidence bound of baseline energy ratio at each wind direction bin.
            **counts_ratio_array_base** (*np.array*): Counts per wind direction bin in baseline.
    """

    # Ensure that input arrays are np.ndarray
    reference_power_baseline = _convert_to_numpy_array(
        reference_power_baseline)
    test_power_baseline = _convert_to_numpy_array(test_power_baseline)
    wind_direction_array_baseline = _convert_to_numpy_array(
        wind_direction_array_baseline)

    # Handle no overlap specificed (assume non-overlap)
    if wind_direction_bin_p_overlap is None:
        wind_direction_bin_p_overlap = 0

    # Compute binning radius (is this right?)
    wind_direction_bin_radius = (1.0 + wind_direction_bin_p_overlap / 100.) * (
        wind_direction_bins[1]-wind_direction_bins[0])/2.0


    ratio_array_base = np.zeros(len(wind_direction_bins)) * np.nan
    lower_ratio_array_base = np.zeros(len(wind_direction_bins)) * np.nan
    upper_ratio_array_base = np.zeros(len(wind_direction_bins)) * np.nan
    counts_ratio_array_base = np.zeros(len(wind_direction_bins)) * np.nan

    for i, wind_direction_bin in enumerate(wind_direction_bins):

        wind_dir_mask_baseline = (wind_direction_array_baseline >= wind_direction_bin - wind_direction_bin_radius) \
            & (wind_direction_array_baseline < wind_direction_bin + wind_direction_bin_radius)

        reference_power_baseline_wd = reference_power_baseline[wind_dir_mask_baseline]
        test_power_baseline_wd = test_power_baseline[wind_dir_mask_baseline]
        wind_dir_array_baseline_wd = wind_direction_array_baseline[wind_dir_mask_baseline]
        baseline_weight = gaussian(wind_dir_array_baseline_wd, wind_direction_bin,wind_direction_bin_radius/2.0 )
        baseline_weight = baseline_weight / np.sum(baseline_weight)

        if (len(reference_power_baseline_wd) == 0):
            continue

        # compute the energy ratio
        # ratio_array_base[i], counts_ratio_array_base[i] = energy_ratio(reference_power_baseline_wd, test_power_baseline_wd)

        # Get the bounds through boot strapping
        # determine the number of bootstrap iterations if not given
        if n_boostrap is None:
            n_boostrap = _calculate_bootstrap_iterations(
                len(reference_power_baseline_wd))

        ratio_base_bs = np.zeros(n_boostrap)
        for i_bs in range(n_boostrap):

            # random resampling w/ replacement
            #ind_bs = np.random.randint(
            #     len(reference_power_baseline_wd), size=len(reference_power_baseline_wd))
            ind_bs = np.random.choice(
                len(reference_power_baseline_wd), size=len(reference_power_baseline_wd),p=baseline_weight)
            reference_power_binned_baseline = reference_power_baseline_wd[ind_bs]
            test_power_binned_baseline = test_power_baseline_wd[ind_bs]

            # compute the energy ratio
            ratio_base_bs[i_bs], _ = energy_ratio(reference_power_binned_baseline, test_power_binned_baseline)

        # Get the confidence bounds
        percentiles = _get_confidence_bounds(confidence)

        # Compute the central over from the bootstrap runs
        ratio_array_base[i] = np.mean(ratio_base_bs)

        lower_ratio_array_base[i], upper_ratio_array_base[i] = _calculate_lower_and_upper_bound(
            ratio_base_bs, percentiles, central_estimate=ratio_array_base[i], method='simple_percentile')
        
    return ratio_array_base, lower_ratio_array_base, upper_ratio_array_base, counts_ratio_array_base


def plot_energy_ratio(reference_power_baseline,
                      test_power_baseline,
                      wind_speed_array_baseline,
                      wind_direction_array_baseline,
                      wind_direction_bins,
                      confidence=95,
                      n_boostrap=None,
                      wind_direction_bin_p_overlap=None,
                      ax=None,
                      base_color='b',
                      label_array=None,
                      label_pchange=None,
                      plot_simple=False,
                      plot_ratio_scatter=False,
                      marker_scale=1.,
                      show_count=True,
                      hide_controlled_case=False
                      ):
    """
    Plot the single energy ratio.

    Function mainly acts as a wrapper to call 
    calculate_balanced_energy_ratio and plot the results.

    Args:
        reference_power_baseline (np.array): Array of power 
            of reference turbine in baseline conditions.
        test_power_baseline (np.array): Array of power of 
            test turbine in baseline conditions.
        wind_speed_array_baseline (np.array): Array of wind 
            speeds in baseline conditions.
        wind_direction_array_baseline (np.array): Array of 
            wind directions in baseline case.
        wind_direction_bins (np.array): Wind directions bins.
        confidence (int, optional): Confidence level to use.  
            Defaults to 95.
        n_boostrap (int, optional): Number of bootstaps, if 
            none, _calculate_bootstrap_iterations is called.  Defaults 
            to None.
        wind_direction_bin_p_overlap (np.array, optional): 
            Percentage overlap between wind direction bin. Defaults to 
            None.
        axarr ([axes], optional): list of axes to plot to. 
            Defaults to None.
        base_color (str, optional): Color of baseline in 
            plots. Defaults to 'b'.
        label_array ([str], optional): List of labels to 
            apply Defaults to None.
        label_pchange ([type], optional): Label for 
            percentage change. Defaults to None.
        plot_simple (bool, optional): Plot only the ratio, no 
            confidence. Defaults to False.
        plot_ratio_scatter (bool, optional): Include scatter 
            plot of values, sized to indicate counts. Defaults to False.
        marker_scale ([type], optional): Marker scale. 
            Defaults to 1.
        show_count (bool, optional): Show the counts as scatter plot
        hide_controlled_case (bool, optional): Option to hide the control case from plots, for demonstration

    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, sharex=True)

    if label_array is None:
        label_array = ['Baseline']

    if label_pchange is None:
        label_pchange = 'Energy Gain'

    ratio_array_base, lower_ratio_array_base, upper_ratio_array_base, \
        counts_ratio_array_base,  = calculate_balanced_energy_ratio \
            (reference_power_baseline, test_power_baseline, \
                wind_direction_array_baseline, wind_direction_bins, \
                    confidence=95,n_boostrap=None, \
                        wind_direction_bin_p_overlap = 
                        wind_direction_bin_p_overlap, )

    if plot_simple:
        ax.plot(wind_direction_bins, ratio_array_base,
                label=label_array[0], color=base_color, ls='--')
        ax.axhline(1, color='k')
        ax.set_ylabel('Energy Ratio (-)')

    else:

        ax.plot(wind_direction_bins, ratio_array_base,
                label=label_array[0], color=base_color, ls='-', marker='.')
        ax.fill_between(wind_direction_bins, lower_ratio_array_base,
                        upper_ratio_array_base, alpha=0.3, color=base_color, label='_nolegend_')
        if show_count:
            ax.scatter(wind_direction_bins, ratio_array_base, s=counts_ratio_array_base,
                    label='_nolegend_', color=base_color, marker='o', alpha=0.2)
        ax.axhline(1, color='k')
        ax.set_ylabel('Energy Ratio (-)')


        ax.grid(True)
        ax.set_xlabel('Wind Direction (Deg)')
