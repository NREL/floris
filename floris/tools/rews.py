import numpy as np
from ..utilities import wrap_180, wrap_360


def log_law_interpolate(z_test,z_ref,v_ref,roughness=0.03):
    return v_ref * np.log(z_test/roughness) / np.log(z_ref/roughness)

def determine_rews_weights(R, HH, heights_in):

    # Determine rotor area
    Area = np.pi * R**2

    # Remove any heights not in range of the rotor
    num_heights_in = len(heights_in)
    heights = [h for h in heights_in if ((h >= HH - R) and (h <= HH + R))]
    num_heights = len(heights)

    # Determine the zone interfaces
    zone_boundaries = np.zeros(num_heights+1)
    zone_boundaries[0] = HH - R
    zone_boundaries[-1] = HH + R
    for i in range(1,num_heights):
        zone_boundaries[i] = (heights[i] - heights[i-1]) / 2.0 + heights[i-1]
    zone_interfaces = zone_boundaries[1:-1]
    
    # Next find the central angles for each interace
    h = zone_interfaces - HH
    alpha = np.arcsin(h / R)
    C = np.pi - 2* alpha
    A = ((R**2)/2) * (C - np.sin(C))
    A = [np.pi*R**2] + [a for a in A]
    for i in range(num_heights-1):
        A[i] = A[i] - A[i+1]
    weights = A

    # normalize
    weights = weights / np.sum(weights)

    # Now re-pad weights to include heights that were initally cropped
    weight_dict = dict(zip(heights,weights))
    weights_return = [weight_dict.get(h,0.0) for h in heights_in]


    return weights_return

def rews_from_df(df, columns_in,weights, rews_name, circular=False):

    # Ensure numpy array
    weights = np.array(weights)

    # Get the data
    data_matrix = df[columns_in].values

    if not circular:
        df[rews_name] = compute_rews(data_matrix,weights)
    else:
        cos_vals = compute_rews(np.cos(np.deg2rad(data_matrix)), weights)
        sin_vals = compute_rews(np.sin(np.deg2rad(data_matrix)), weights)
        df[rews_name] = wrap_360(np.rad2deg(np.arctan2(sin_vals,cos_vals)))

    return df

def compute_rews(data_matrix, weights):
    return np.sum(data_matrix * weights, axis=1)