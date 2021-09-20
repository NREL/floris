import numpy as np

# Wrapper to include normalization of the inputs and cost function
def _norm(val, x1, x2):
    return (val - x1) / (x2 - x1)


def _unnorm(val, x1, x2):
    return np.array(val) * (x2 - x1) + x1


def _cost_norm(
    fi,
    yaw_angles_norm,
    yaw_norm_lb=0.0,
    yaw_norm_ub=1.0,
    farm_power_ub=1.0,
    turbine_weights=None,
    include_unc=False,
    unc_pmfs=None,
    unc_options=None,
):
    # Undo normalization in yaw_angles
    yaw_angles = _unnorm(yaw_angles_norm, yaw_norm_lb, yaw_norm_ub)

    # Initialize turbine_weights array
    if turbine_weights is None:
        turbine_weights = np.ones_like(fi.layout_x)

    # Calculate cost
    fi.calculate_wake(yaw_angles=yaw_angles)
    turbine_powers = fi.get_turbine_power(
        include_unc=include_unc,
        unc_pmfs=unc_pmfs,
        unc_options=unc_options,
    )
    
    # Normalize cost
    J = -1.0 * np.dot(turbine_weights, turbine_powers)
    J_norm = J / farm_power_ub

    return J_norm


def _cost_norm_selective_turbines(
    fi,
    yaw_angles_subset_norm,
    turbs_to_opt,
    yaw_angles_norm_template,
    yaw_norm_lb=0.0,
    yaw_norm_ub=1.0,
    farm_power_ub=1.0,
    turbine_weights=None,
    include_unc=False,
    unc_pmfs=None,
    unc_options=None,
):
    # Combine template yaw angles and subset
    yaw_angles_norm = np.array(yaw_angles_norm_template, dtype=float, copy=True)
    yaw_angles_norm[turbs_to_opt] = yaw_angles_subset_norm

    return _cost_norm(
        fi=fi,
        yaw_angles_norm=yaw_angles_norm,
        yaw_norm_lb=yaw_norm_lb,
        yaw_norm_ub=yaw_norm_ub,
        farm_power_ub=farm_power_ub,
        turbine_weights=turbine_weights,
        include_unc=include_unc,
        unc_pmfs=unc_pmfs,
        unc_options=unc_options,
    )