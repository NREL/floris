import numpy as np
from floris.utilities import rotate_coordinates_rel_west

# Geometric wake steering yaw angle selection for fast coupled
# layout and yaw optimization. For details, see
# https://wes.copernicus.org/preprints/wes-2023-1/

def _process_layout(
    turbine_x,
    turbine_y,
    rotor_diameter,
    spread=0.1
):
    """
    returns the distance from each turbine to the nearest downstream waked turbine
    normalized by the rotor diameter. Right now "waked" is determind by a Jensen-like
    wake spread, but this could/should be modified to be the same as the trapezoid rule
    used to determine the yaw angles.

    turbine_x: turbine x coords (rotated)
    turbine_y: turbine y coords (rotated)
    rotor_diameter: turbine rotor diameter (float)
    spread=0.1: Jensen alpha wake spread value
    """
    nturbs = len(turbine_x)

    # Intialize storage
    dx = np.zeros(nturbs) + 1E10
    dy = np.zeros(nturbs)

    # for waking_index in range(nturbs):
    #     for waked_index in range(nturbs):
    #         if turbine_x[waked_index] > turbine_x[waking_index]:
    #             r = spread*(turbine_x[waked_index]-turbine_x[waking_index]) + rotor_diameter/2.0
    #             if abs(turbine_y[waked_index]-turbine_y[waking_index]) < (r+rotor_diameter/2.0):
    #                 if (turbine_x[waked_index] - turbine_x[waking_index]) < dx[waking_index]:
    #                     dx[waking_index] = turbine_x[waked_index] - turbine_x[waking_index]
    #                     dy[waking_index] = turbine_y[waked_index]- turbine_y[waking_index]
    #     if dx[waking_index] == 1E10:
    #         dx[waking_index] = 0.0

    # Compute distances
    x_dists = turbine_x.reshape(-1,1).T - turbine_x.reshape(-1,1)
    y_dists = turbine_y.reshape(-1,1).T - turbine_y.reshape(-1,1)

    # Check within Jensen model spread
    # Also remove diagonal elements
    in_Jensen_wake = (abs(y_dists) < spread * x_dists + rotor_diameter)
    x_dists[~in_Jensen_wake] = np.inf
    x_dists = x_dists + np.diag(np.inf*np.ones(len(turbine_x)))

    # Get minimums (and arguments to select the correct y values also)
    dx = x_dists.min(axis=1)
    dy = y_dists[range(len(turbine_x)), x_dists.argmin(axis=1)]

    # Handle last turbine downstream
    furthest_ds_turb_idx = np.where(dx == np.inf)[0]
    dx[furthest_ds_turb_idx] = 0.
    dy[furthest_ds_turb_idx] = 0.

    return dx/rotor_diameter, dy/rotor_diameter


def _get_yaw_angles(
    x,
    y,
    left_x,
    top_left_y,
    right_x,
    top_right_y,
    top_left_yaw,
    top_right_yaw,
    bottom_left_yaw,
    bottom_right_yaw
):
    """
    _______2,5___________________________4,6
    |.......................................
    |......1,7...........................3,8
    |.......................................
    ________________________________________

    x and y: dx and dy to the nearest downstream turbine in rotor diameteters with
        turbines rotated so wind is coming left to right
    left_x: where we start the trapezoid. Should be left as 0.
    top_left_y: trapezoid top left coord
    right_x: where to stop the trapezoid downstream.
        Max coord after which the upstream turbine won't yaw.
    top_right_y: trapezoid top right coord
    top_left_yaw: yaw angle associated with top left point
    top_right_yaw: yaw angle associated with top right point
    bottom_left_yaw: yaw angle associated with bottom left point
    bottom_right_yaw: yaw angle associated with bottom right point
    """

    if x <= 0:
        return 0.0
    else:
        dx = (x-left_x)/(right_x-left_x)
        edge_y = top_left_y + (top_right_y-top_left_y)*dx
        if abs(y) > edge_y:
            return 0.0
        else:
            top_yaw = top_left_yaw + (top_right_yaw-top_left_yaw)*dx
            bottom_yaw = bottom_left_yaw + (bottom_right_yaw-bottom_left_yaw)*dx
            yaw = bottom_yaw + (top_yaw-bottom_yaw)*abs(y)/edge_y
            if y < 0:
                return -yaw
            else:
                return yaw

def geometric_yaw(
    turbine_x,
    turbine_y,
    wind_direction,
    rotor_diameter,
    left_x=0.0,
    top_left_y=1.0,
    right_x=25.0,
    top_right_y=1.0,
    top_left_yaw=30.0,
    top_right_yaw=0.0,
    bottom_left_yaw=30.0,
    bottom_right_yaw=0.0
):
    """
    turbine_x: unrotated x turbine coords
    turbine_y: unrotated y turbine coords
    wind_direction: float, degrees
    rotor_diameter: float
    left_x: where we start the trapezoid. Should be left as 0.
    top_left_y: trapezoid top left coord
    right_x: where to stop the trapezoid downstream.
        Max coord after which the upstream turbine won't yaw.
    top_right_y: trapezoid top right coord
    top_left_yaw: yaw angle associated with top left point
    top_right_yaw: yaw angle associated with top right point
    bottom_left_yaw: yaw angle associated with bottom left point
    bottom_right_yaw: yaw angle associated with bottom right point
    """

    nturbs = len(turbine_x)
    turbine_coordinates_array = np.zeros((nturbs,3))
    turbine_coordinates_array[:,0] = turbine_x[:]
    turbine_coordinates_array[:,1] = turbine_y[:]

    rotated_x, rotated_y, _, _, _ = rotate_coordinates_rel_west(
        np.array([wind_direction]),
        turbine_coordinates_array
    )
    processed_x, processed_y = _process_layout(rotated_x[0][0],rotated_y[0][0],rotor_diameter)
    yaw_array = np.zeros(nturbs)
    for i in range(nturbs-1):
        yaw_array[i] = _get_yaw_angles(
            processed_x[i],
            processed_y[i],
            left_x,
            top_left_y,
            right_x,
            top_right_y,
            top_left_yaw,
            top_right_yaw,
            bottom_left_yaw,
            bottom_right_yaw
        )

    return yaw_array


if __name__=="__main__":

    # check that it is returning values
    turbine_x = np.array([0,700,1400,2100])
    turbine_y = np.array([0,50,0,-100])
    wind_direction = 270
    rotor_diameter = 100

    yaw_angles = geometric_yaw(turbine_x, turbine_y, wind_direction, rotor_diameter)

    print("yaw angles: ", yaw_angles)
