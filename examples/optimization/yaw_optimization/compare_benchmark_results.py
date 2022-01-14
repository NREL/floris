# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def unpack_yaw_angles(df):
    yaw_angles = df["yaw_angles_opt"].apply(lambda x: x.replace("[", "").replace("]", ""))
    yaw_angles = [y.split(" ") for y in yaw_angles]
    yaw_angles = [[yw for yw in y if ((not yw == " ") and (not yw == ""))] for y in yaw_angles]  # Remove empty spaces
    yaw_angles = [[float(yw) for yw in y] for y in yaw_angles]  # Convert values to floats
    yaw_angles = np.vstack(yaw_angles)
    return yaw_angles


if __name__ == "__main__":
    # Specify current directory
    root_path = os.path.dirname(os.path.abspath(__file__))
    label_v2 = "v2.4 (develop, serial refine)"
    label_v3 = "v3.0 (serial refine)"
    df_v2 = pd.read_csv(os.path.join(root_path, "df_opt_N3_v24develop_serialrefine.csv"))
    df_v3 = pd.read_csv(os.path.join(root_path, "df_opt_N3_v30_serialrefine.csv"))

    fig, ax = plt.subplots(nrows=3)
    ax[0].bar(df_v2["wind_direction"] - 1.0, df_v2["farm_power_baseline"], width=2.0, label=label_v2)
    ax[0].bar(df_v3["wind_direction"] + 1.0, df_v3["farm_power_baseline"], width=2.0, label=label_v3)
    ax[0].set_ylabel("Baseline power production (W)")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].bar(df_v2["wind_direction"] - 1.0, df_v2["farm_power_opt"], width=2.0, label=label_v2)
    ax[1].bar(df_v3["wind_direction"] + 1.0, df_v3["farm_power_opt"], width=2.0, label=label_v3)
    ax[1].set_ylabel("Optimized power production (W)")
    ax[1].legend()
    ax[1].grid(True)

    ax[2].bar(df_v2["wind_direction"] - 1.0, 100 * (df_v2["farm_power_opt"] / df_v2["farm_power_baseline"] - 1.0), width=2.0, label=label_v2)
    ax[2].bar(df_v3["wind_direction"] + 1.0, 100 * (df_v3["farm_power_opt"] / df_v3["farm_power_baseline"] - 1.0), width=2.0, label=label_v3)
    ax[2].set_ylabel("Uplift (%)")
    ax[2].legend()
    ax[2].grid(True)

    yaw_angles_v2 = unpack_yaw_angles(df_v2)
    yaw_angles_v3 = unpack_yaw_angles(df_v3)
    for ti in range(yaw_angles_v2.shape[1]):
        fig, ax = plt.subplots()
        ax.plot(df_v2["wind_direction"], yaw_angles_v2[:, ti], label=label_v2)
        ax.plot(df_v3["wind_direction"], yaw_angles_v3[:, ti], '--o', markersize=3, label=label_v3)
        ax.grid(True)
        ax.legend()
        ax.set_ylabel("Optimal yaw offset (deg)")
        ax.set_title("Turbine {:d}".format(ti))

    plt.show()
