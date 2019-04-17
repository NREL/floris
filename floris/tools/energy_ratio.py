
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class PlotDefaults():
    def __init__(self):
        """
        This class sets journal-ready styles for plots.
        """
        
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

def _convert_to_numpy_array(series):
    if hasattr(series, 'values'):
        return series.values
    elif isinstance(series, np.ndarray):
        return series

# def _ratio_of_mean(x, y):
#     """
#     Arguments
#         x: numerator
#         y: denominator
#     """
#     return np.mean(x) / np.mean(y)

def _calculate_bootstrap_iterations(n):
    maximum = 10000
    minimum = 2000
    return int(np.round(max(min(n * np.log10(n), maximum), minimum)))

def _calculate_lower_and_upper_bound(bootstrap_array, percentiles, central_estimate=None,method='simple_percentile'):
    if method is 'simple_percentile':
        lower, upper = np.percentile(bootstrap_array, percentiles)
    else:
        lower, upper = (2 * central_estimate - np.percentile(bootstrap_array, percentiles))
    return lower, upper


def _get_confidence_bounds(confidence):
        return [50 + 0.5 * confidence, 50 - 0.5 * confidence]


def energy_ratio(ref_pow_base,test_pow_base, ws_base,
                ref_pow_con,test_pow_con, ws_con):

    """

    """

    # First derive the weighting functions by wind speed
    ws_unique_base = np.unique(ws_base)
    ws_unique_con = np.unique(ws_con)
    ws_unique = np.intersect1d(ws_unique_base,ws_unique_con )

    if len(ws_unique)==0:
        return np.nan, np.nan, np.nan, np.nan

    # Mask down to the items in both sides
    base_mask = np.isin(ws_base,ws_unique)
    con_mask = np.isin(ws_con,ws_unique)
    ref_pow_base = ref_pow_base[base_mask]
    test_pow_base = test_pow_base[base_mask]
    ws_base = ws_base[base_mask]
    ref_pow_con = ref_pow_con[con_mask]
    test_pow_con = test_pow_con[con_mask]
    ws_con = ws_con[con_mask]

    ws_unique_base, counts_base = np.unique(ws_base, return_counts=True)
    ws_unique_con, counts_con = np.unique(ws_con, return_counts=True)
    total_counts = counts_base + counts_con

    # Make the weights per wind speed
    weights_base = counts_con.astype(float) / total_counts.astype(float)
    weights_con = counts_base.astype(float) / total_counts.astype(float)

    # Make a weighting array
    lut_base = np.zeros(np.max(ws_unique)+1)
    lut_base[ws_unique] = weights_base
    weight_array_base = lut_base[ws_base]
    lut_con = np.zeros(np.max(ws_unique)+1)
    lut_con[ws_unique] = weights_con
    weight_array_con = lut_con[ws_con]
    
    # Weighted sums 
    weight_sum_ref_base = np.sum(ref_pow_base * weight_array_base)
    weight_sum_test_base = np.sum(test_pow_base * weight_array_base)
    weight_sum_ref_con = np.sum(ref_pow_con * weight_array_con)
    weight_sum_test_con = np.sum(test_pow_con * weight_array_con)

    # Ratio and diff
    ratio_base = weight_sum_test_base / weight_sum_ref_base
    ratio_con = weight_sum_test_con / weight_sum_ref_con
    ratio_diff = ratio_con - ratio_base
    p_change= 100. * ratio_diff / ratio_base

    return ratio_base, ratio_con, ratio_diff, p_change


def calculate_balanced_energy_ratio(reference_power_baseline,
                            test_power_baseline,
                            wind_speed_array_baseline,
                            wind_direction_array_baseline,
                            reference_power_controlled,
                            test_power_controlled,
                            wind_speed_array_controlled,
                            wind_direction_array_controlled,
                            wind_direction_bins,
                            confidence=95,
                            n_boostrap=None,
                            wind_direction_bin_p_overlap=None,
                            ):

    # Ensure that input arrays are np.ndarray
    reference_power_baseline = _convert_to_numpy_array(reference_power_baseline)
    test_power_baseline = _convert_to_numpy_array(test_power_baseline)
    wind_speed_array_baseline = _convert_to_numpy_array(wind_speed_array_baseline)
    wind_direction_array_baseline = _convert_to_numpy_array(wind_direction_array_baseline)

    reference_power_controlled = _convert_to_numpy_array(reference_power_controlled)
    test_power_controlled = _convert_to_numpy_array(test_power_controlled)
    wind_speed_array_controlled = _convert_to_numpy_array(wind_speed_array_controlled)
    wind_direction_array_controlled = _convert_to_numpy_array(wind_direction_array_controlled)

    # Handle no overlap specificed (assume non-overlap)
    if wind_direction_bin_p_overlap is None:
        wind_direction_bin_p_overlap = 0

    # Compute binning radius (is this right?)
    wind_direction_bin_radius = (1.0 + wind_direction_bin_p_overlap / 100.) *  (wind_direction_bins[1]-wind_direction_bins[0])/2.0

    ratio_array_base = np.zeros(len(wind_direction_bins)) * np.nan
    lower_ratio_array_base = np.zeros(len(wind_direction_bins))* np.nan
    upper_ratio_array_base = np.zeros(len(wind_direction_bins))* np.nan
    counts_ratio_array_base = np.zeros(len(wind_direction_bins))* np.nan
    
    ratio_array_con = np.zeros(len(wind_direction_bins))* np.nan
    lower_ratio_array_con = np.zeros(len(wind_direction_bins))* np.nan
    upper_ratio_array_con = np.zeros(len(wind_direction_bins))* np.nan
    counts_ratio_array_con = np.zeros(len(wind_direction_bins))* np.nan

    diff_array = np.zeros(len(wind_direction_bins))* np.nan
    lower_diff_array = np.zeros(len(wind_direction_bins))* np.nan
    upper_diff_array = np.zeros(len(wind_direction_bins))* np.nan
    counts_diff_array = np.zeros(len(wind_direction_bins))* np.nan

    p_change_array = np.zeros(len(wind_direction_bins))* np.nan
    lower_p_change_array = np.zeros(len(wind_direction_bins))* np.nan
    upper_p_change_array = np.zeros(len(wind_direction_bins))* np.nan
    counts_p_change_array = np.zeros(len(wind_direction_bins))* np.nan
    



    for i, wind_direction_bin in enumerate(wind_direction_bins):

        wind_dir_mask_baseline = (wind_direction_array_baseline >= wind_direction_bin - wind_direction_bin_radius) \
            & (wind_direction_array_baseline < wind_direction_bin + wind_direction_bin_radius)

        wind_dir_mask_controlled = (wind_direction_array_controlled >= wind_direction_bin - wind_direction_bin_radius) \
            & (wind_direction_array_controlled < wind_direction_bin + wind_direction_bin_radius)


        reference_power_baseline_wd = reference_power_baseline[wind_dir_mask_baseline]
        test_power_baseline_wd = test_power_baseline[wind_dir_mask_baseline]
        wind_speed_array_baseline_wd = wind_speed_array_baseline[wind_dir_mask_baseline]
        
        reference_power_controlled_wd = reference_power_controlled[wind_dir_mask_controlled]
        test_power_controlled_wd = test_power_controlled[wind_dir_mask_controlled]
        wind_speed_array_controlled_wd = wind_speed_array_controlled[wind_dir_mask_controlled]

        if (len(reference_power_baseline_wd)==0) or (len(reference_power_controlled_wd)==0):
            continue

        # Convert wind speed to integers
        wind_speed_array_baseline_wd = wind_speed_array_baseline_wd.astype(int)
        wind_speed_array_controlled_wd = wind_speed_array_controlled_wd.astype(int)

        # compute the energy ratio
        ratio_array_base[i], ratio_array_con[i], diff_array[i],p_change_array[i]   = energy_ratio(reference_power_baseline_wd,test_power_baseline_wd,wind_speed_array_baseline_wd,
                                           reference_power_controlled_wd, test_power_controlled_wd, wind_speed_array_controlled_wd     )


        # Get the bounds through boot strapping
        # determine the number of bootstrap iterations if not given
        if n_boostrap is None:
            n_boostrap = _calculate_bootstrap_iterations(len(reference_power_baseline_wd))

        ratio_base_bs = np.zeros(n_boostrap)
        ratio_con_bs = np.zeros(n_boostrap)
        diff_bs = np.zeros(n_boostrap)
        p_change_bs = np.zeros(n_boostrap)
        for i_bs in range(n_boostrap):

            # random resampling w/ replacement
            ind_bs = np.random.randint(len(reference_power_baseline_wd), size=len(reference_power_baseline_wd))
            reference_power_binned_baseline = reference_power_baseline_wd[ind_bs]
            test_power_binned_baseline = test_power_baseline_wd[ind_bs]
            wind_speed_binned_baseline = wind_speed_array_baseline_wd[ind_bs]

            ind_bs = np.random.randint(len(reference_power_controlled_wd), size=len(reference_power_controlled_wd))
            reference_power_binned_controlled = reference_power_controlled_wd[ind_bs]
            test_power_binned_controlled = test_power_controlled_wd[ind_bs]
            wind_speed_binned_controlled = wind_speed_array_controlled_wd[ind_bs]
        
            # compute the energy ratio
            ratio_base_bs[i_bs], ratio_con_bs[i_bs], diff_bs[i_bs],p_change_bs[i_bs]   = energy_ratio(reference_power_binned_baseline,test_power_binned_baseline,wind_speed_binned_baseline,
                                           reference_power_binned_controlled, test_power_binned_controlled, wind_speed_binned_controlled     )


        # Get the confidence bounds
        percentiles = _get_confidence_bounds(confidence)

        lower_ratio_array_base[i], upper_ratio_array_base[i] = _calculate_lower_and_upper_bound(ratio_base_bs, percentiles, central_estimate=ratio_array_base[i],method='simple_percentile')
        lower_ratio_array_con[i], upper_ratio_array_con[i] = _calculate_lower_and_upper_bound(ratio_con_bs, percentiles, central_estimate=ratio_array_con[i],method='simple_percentile')
        lower_diff_array[i], upper_diff_array[i] = _calculate_lower_and_upper_bound(diff_bs, percentiles, central_estimate=diff_array[i],method='simple_percentile')
        lower_p_change_array[i], upper_p_change_array[i] = _calculate_lower_and_upper_bound(p_change_bs, percentiles, central_estimate=p_change_array[i],method='simple_percentile')


    return ratio_array_base,lower_ratio_array_base, upper_ratio_array_base, ratio_array_con, lower_ratio_array_con, upper_ratio_array_con, diff_array, lower_diff_array, upper_diff_array, p_change_array, lower_p_change_array, upper_p_change_array
       
def plot_energy_ratio(reference_power_baseline,
                            test_power_baseline,
                            wind_speed_array_baseline,
                            wind_direction_array_baseline,
                            reference_power_controlled,
                            test_power_controlled,
                            wind_speed_array_controlled,
                            wind_direction_array_controlled,
                            wind_direction_bins,
                            confidence=95,
                            n_boostrap=None,
                            wind_direction_bin_p_overlap=None,
                            axarr=None, 
                            base_color='b',
                            con_color='g',
                            label='_nolegend_', 
                            y_lim=None,
                            # indicate_all_ratios=False,
                            plot_simple=False,
                            plot_ratio_scatter=False,
                            marker_scale=1.,
                            alt=False):

    if axarr is None:
        fig, axarr = plt.subplots(3,1,sharex=True)


    ratio_array_base,lower_ratio_array_base, upper_ratio_array_base, ratio_array_con, lower_ratio_array_con, upper_ratio_array_con,  diff_array, lower_diff_array, upper_diff_array, p_change_array, lower_p_change_array, upper_p_change_array = calculate_balanced_energy_ratio(reference_power_baseline,
                            test_power_baseline,
                            wind_speed_array_baseline,
                            wind_direction_array_baseline,
                            reference_power_controlled,
                            test_power_controlled,
                            wind_speed_array_controlled,
                            wind_direction_array_controlled,
                            wind_direction_bins,
                            confidence=95,
                            n_boostrap=None,
                            wind_direction_bin_p_overlap=None,
                            )



    if plot_simple:
        # ax.plot(wind_direction_bins, ratio_array_base, label=label, color=color, ls='--')
        # ax.plot(wind_direction_bins, ratio_array_con, label=label, color='g', ls='--')
        # plt.show()
        ax = axarr[0]
        ax.plot(wind_direction_bins, ratio_array_base, label='Baseline', color=base_color,ls='--',marker='x')
        # ax.fill_between(wind_direction_bins,lower_ratio_array_base,upper_ratio_array_base,alpha=0.3,color=base_color,label='_nolegend_')
        ax.plot(wind_direction_bins, ratio_array_con, label='Controlled', color=con_color,ls='--',marker='x')
        # ax.fill_between(wind_direction_bins,lower_ratio_array_con,upper_ratio_array_con,alpha=0.3,color=con_color,label='_nolegend_')
        ax.axhline(1,color='k')
        ax.set_ylabel('Energy Ratio (-)')
        ax.grid(True)
        # ax.scatter(wind_direction_bins, ratio_array, label='_nolegend_', edgecolors=color, s=counts_array/marker_scale,facecolors='none',linewidth=2)
        # ax.scatter(wind_direction_bins, ratio_array, label='_nolegend_', color=color,marker='^')
        # ax = axarr[1]
        # ax.plot(wind_direction_bins, diff_array, label='Difference', color=con_color,ls='-',marker='.')
        # ax.fill_between(wind_direction_bins,lower_diff_array,upper_diff_array,alpha=0.3,color=con_color,label='_nolegend_')
        # ax.axhline(0,color='k')
        # ax.set_ylabel('Difference (-)')
        # ax.grid(True)
        ax = axarr[1]
        ax.plot(wind_direction_bins, p_change_array, label='Percent Change', color=con_color,ls='--',marker='x')
        # ax.fill_between(wind_direction_bins,lower_p_change_array,upper_p_change_array,alpha=0.3,color=con_color,label='_nolegend_')
        ax.axhline(0,color='k')
        ax.set_ylabel('Percent Change (%)')
        ax.grid(True)
    else:

        ax = axarr[0]
        ax.plot(wind_direction_bins, ratio_array_base, label='Baseline', color=base_color,ls='-',marker='.')
        ax.fill_between(wind_direction_bins,lower_ratio_array_base,upper_ratio_array_base,alpha=0.3,color=base_color,label='_nolegend_')
        ax.plot(wind_direction_bins, ratio_array_con, label='Controlled', color=con_color,ls='-',marker='.')
        ax.fill_between(wind_direction_bins,lower_ratio_array_con,upper_ratio_array_con,alpha=0.3,color=con_color,label='_nolegend_')
        ax.axhline(1,color='k')
        ax.set_ylabel('Energy Ratio (-)')
        ax.grid(True)
        # ax.scatter(wind_direction_bins, ratio_array, label='_nolegend_', edgecolors=color, s=counts_array/marker_scale,facecolors='none',linewidth=2)
        # ax.scatter(wind_direction_bins, ratio_array, label='_nolegend_', color=color,marker='^')
        # ax = axarr[1]
        # ax.plot(wind_direction_bins, diff_array, label='Difference', color=con_color,ls='-',marker='.')
        # ax.fill_between(wind_direction_bins,lower_diff_array,upper_diff_array,alpha=0.3,color=con_color,label='_nolegend_')
        # ax.axhline(0,color='k')
        # ax.set_ylabel('Difference (-)')
        # ax.grid(True)
        ax = axarr[1]
        ax.plot(wind_direction_bins, p_change_array, label='Percent Change', color=con_color,ls='-',marker='.')
        ax.fill_between(wind_direction_bins,lower_p_change_array,upper_p_change_array,alpha=0.3,color=con_color,label='_nolegend_')
        ax.axhline(0,color='k')
        ax.set_ylabel('Percent Change (%)')
        ax.grid(True)
        # plt.show()
    #     ax.set_ylim(y_lim)

    # return ratio_array, lower_array, upper_array, counts_array
