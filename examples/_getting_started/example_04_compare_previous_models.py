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
 

import matplotlib.pyplot as plt
import floris.tools as wfct

# Initialize the FLORIS interface for 4 seperate models
fi_jensen = wfct.floris_interface.FlorisInterface("../other_jsons/jensen.json")
fi_mz = wfct.floris_interface.FlorisInterface("../other_jsons/multizone.json")
fi_gauss = wfct.floris_interface.FlorisInterface("../other_jsons/input_legacy.json")
fi_gch = wfct.floris_interface.FlorisInterface("../example_input.json")

fig, axarr = plt.subplots(2,4,figsize=(16,4))

for idx, (fi,name) in enumerate(zip([fi_jensen,fi_mz,fi_gauss,fi_gch],['Jensen','Multizone','Gaussian','GCH'])):

    fi.calculate_wake()

    # Aligned
    ax = axarr[0, idx]
    hor_plane = fi.get_hor_plane()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    ax.set_title(name)
    axarr[0,0].set_ylabel('Aligned')

    # Yawed
    fi.calculate_wake(yaw_angles=[25])
    ax = axarr[1, idx]
    hor_plane = fi.get_hor_plane()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    axarr[1,0].set_ylabel('yawed')



plt.show()
