# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats

def _convert_to_numpy_array(series):
    if hasattr(series, 'values'):
        return series.values
    elif isinstance(series, np.ndarray):
        return series


# Define ci function
def ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


def calculate_balanced_wake_loss(reference_power_baseline,
                                    test_power_baseline,
                                    wind_speed_array_baseline,
                                    wind_direction_array_baseline,
                                    reference_power_controlled,
                                    test_power_controlled,
                                    wind_speed_array_controlled,
                                    wind_direction_array_controlled
                                    ):
    """
    Calculate balanced wake loss

    Args:
        reference_power_baseline (np.array): Array of power of 
            reference turbine in baseline conditions.
        test_power_baseline (np.array): Array of power of test turbine 
            in baseline conditions.
        wind_speed_array_baseline (np.array): Array of wind speeds in 
            baseline conditions.
        wind_direction_array_baseline (np.array): Array of wind 
            directions in baseline case.
        reference_power_controlled (np.array): Array of power of 
            reference turbine in controlled conditions.
        test_power_controlled (np.array): Array of power of test 
            turbine in controlled conditions.
        wind_speed_array_controlled (np.array): Array of wind speeds in 
            controlled conditions.
        wind_direction_array_controlled (np.array): Array of wind 
            directions in controlled case.
        

    Returns:
        dataframe

    """

    # Ensure that input arrays are np.ndarray
    reference_power_baseline = _convert_to_numpy_array(
        reference_power_baseline)
    test_power_baseline = _convert_to_numpy_array(test_power_baseline)
    wind_speed_array_baseline = _convert_to_numpy_array(
        wind_speed_array_baseline)
    wind_direction_array_baseline = _convert_to_numpy_array(
        wind_direction_array_baseline)

    reference_power_controlled = _convert_to_numpy_array(
        reference_power_controlled)
    test_power_controlled = _convert_to_numpy_array(test_power_controlled)
    wind_speed_array_controlled = _convert_to_numpy_array(
        wind_speed_array_controlled)
    wind_direction_array_controlled = _convert_to_numpy_array(
        wind_direction_array_controlled)

    # Construct data frame
    df_base = pd.DataFrame({'ref_power':reference_power_baseline,
                            'test_power':test_power_baseline,
                            'ws':wind_speed_array_baseline,
                            'wd':wind_direction_array_baseline})
    df_base['con'] = 'base'
    df_con = pd.DataFrame({'ref_power':reference_power_controlled,
                            'test_power':test_power_controlled,
                            'ws':wind_speed_array_controlled,
                            'wd':wind_direction_array_controlled})
    df_con['con'] = 'con'
    df = df_base.append(df_con)

    # Quantize wind speed and wind direction
    df['ws'] = df.ws.round().astype(int)
    df['wd'] = df.wd.round().astype(int)

    # Look at energy loss
    df['energy_loss'] = df.ref_power - df.test_power

    # Aggragate the losses
    df = df[['ws','wd','con','energy_loss','ref_power']]
    df['count_val']=1
    df_group = df.groupby(['ws','wd','con']).agg([np.mean,np.std,np.sum,ci])
    df_group.columns = ['%s_%s' % c for c in df_group.columns]
    df_group = df_group.unstack()
    df_group.columns = ['%s_%s' % c for c in df_group.columns]

    # Select down a little
    df_group = df_group[df_group.count_val_sum_base >1]
    df_group = df_group[df_group.count_val_sum_con >1]

    # Remove the multi-index and assign back to df
    df = df_group.reset_index()

    # Compute the weighted energy loss
    df['points_per_bin'] = df.count_val_sum_base + df.count_val_sum_con
    df['wt_loss_base'] = df['energy_loss_mean_base'] * df['points_per_bin']
    df['wt_loss_con'] = df['energy_loss_mean_con'] * df['points_per_bin']
    df['ref_en'] = (df['ref_power_mean_base'] + df['ref_power_mean_con'])/2.0 * df['points_per_bin']

    # Compute sums across ws now
    df_sum = df.groupby('wd').sum().reset_index()
    return df_sum


def plot_balanced_wake_loss(reference_power_baseline,
                                    test_power_baseline,
                                    wind_speed_array_baseline,
                                    wind_direction_array_baseline,
                                    reference_power_controlled,
                                    test_power_controlled,
                                    wind_speed_array_controlled,
                                    wind_direction_array_controlled,
                                    axarr
                                    ):
    """
    Calculate balanced wake loss

    Args:
        reference_power_baseline (np.array): Array of power of 
            reference turbine in baseline conditions.
        test_power_baseline (np.array): Array of power of test turbine 
            in baseline conditions.
        wind_speed_array_baseline (np.array): Array of wind speeds in 
            baseline conditions.
        wind_direction_array_baseline (np.array): Array of wind 
            directions in baseline case.
        reference_power_controlled (np.array): Array of power of 
            reference turbine in controlled conditions.
        test_power_controlled (np.array): Array of power of test 
            turbine in controlled conditions.
        wind_speed_array_controlled (np.array): Array of wind speeds in 
            controlled conditions.
        wind_direction_array_controlled (np.array): Array of wind 
            directions in controlled case.
        

    Returns:
        dataframe

    """

    # Get the result frame
    df_sum = calculate_balanced_wake_loss(reference_power_baseline,
                                    test_power_baseline,
                                    wind_speed_array_baseline,
                                    wind_direction_array_baseline,
                                    reference_power_controlled,
                                    test_power_controlled,
                                    wind_speed_array_controlled,
                                    wind_direction_array_controlled
                                    )

    # Plot the energy wake loss
    ax = axarr[0]

    ax.plot(df_sum.wd,df_sum.wt_loss_base/df_sum.ref_en,'b',label='Baseline')
    ax.plot(df_sum.wd,df_sum.wt_loss_con/df_sum.ref_en,'g',label='Controlled')


    # to percent
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    ax.set_ylabel('Wake Loss (%)')
    ax.set_xlabel('Wind Direction (Deg)')
    ax.grid()
    ax.legend()

    # Look percent change
    ax = axarr[1]
    ax.plot(df_sum.wd,(df_sum.wt_loss_base-df_sum.wt_loss_con)/df_sum.ref_en,'g',label='Percent Reduction')



    # to percent
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

    ax.set_ylabel('Difference in wake loss (%)')
    ax.set_xlabel('Wind Direction (Deg)')
    ax.grid()
    ax.legend()

def overall_wake_loss(reference_power_baseline,
                                    test_power_baseline,
                                    wind_speed_array_baseline,
                                    wind_direction_array_baseline,
                                    reference_power_controlled,
                                    test_power_controlled,
                                    wind_speed_array_controlled,
                                    wind_direction_array_controlled
                                    ):
    # Get the result frame
    df_sum = calculate_balanced_wake_loss(reference_power_baseline,
                                    test_power_baseline,
                                    wind_speed_array_baseline,
                                    wind_direction_array_baseline,
                                    reference_power_controlled,
                                    test_power_controlled,
                                    wind_speed_array_controlled,
                                    wind_direction_array_controlled
                                    )

    print('===OVERALL RESULTS===')
    print('Baseline Energy Loss:\t%d%%' % (100*(df_sum['wt_loss_base'].sum() / df_sum['ref_en'].sum())))
    print('Controlled Energy Loss:\t%d%%' % (100*(df_sum['wt_loss_con'].sum() / df_sum['ref_en'].sum())))
    print('Reduction Energy Loss:\t%d%%' % (100*(df_sum['energy_loss_mean_base'].sum() - df_sum['energy_loss_mean_con'].sum())/df_sum['energy_loss_mean_base'].sum()))
    # print('%.1f%%' % (100*(df_sum['energy_loss%s_mean_base' % suffix].sum() - df_sum['energy_loss%s_mean_con' % suffix].sum())/df_sum['energy_loss%s_mean_base' % suffix].sum()))