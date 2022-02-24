# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

from datetime import datetime
from re import M
import matplotlib.pyplot as plt
import mpld3
import pandas as pd

# Example plot
# fig = plt.figure()
# plt.plot([3,1,4,1,5], 'ks-', mec='w', mew=5, ms=20)
# print(mpld3.fig_to_html(fig, figid='2'))


COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


### Construct the data

columns = ["commit_hash", "commit_hash_8char", "date", "jensen", "gauss", "gch", "cc", "code_coverage", "tooltip_label"]
data = [
    ("df25a9cfacd3d652361d2bd37f568af00acb2631", "df25a9cf", datetime(2021,12, 29), 1.2691, 1.2584, 1.6432,   None, 0.4344, "df25a9cf"),
    ("b797390a43298a815f3ff57955cfdc71ecf3e866", "b797390a", datetime(2022, 1,  3), 0.6867, 1.2354, 1.8026,   None, 0.2993, "b797390a"),
    ("01a02d5f91b2f4a863eebe88a618974b0749d1c4", "01a02d5f", datetime(2022, 1,  4), 0.4227, 0.9118, 1.4492,   None, 0.3022, "01a02d5f"),
    ("dd847210082035d43b0273ae63a76a53cb8d2e12", "dd847210", datetime(2022, 1,  6), 0.4081, 0.9049, 1.5270,   None, 0.3627, "dd847210"),
    ("33779269e98cc882a5f066c462d8ec1eadf37a1a", "33779269", datetime(2022, 1, 10), 0.4147, 0.9126, 1.5391,   None, 0.3690, "33779269"),
    ("12890e029a7155b074b9b325d320d1798338e287", "12890e02", datetime(2022, 1, 11), 0.4152, 0.9070, 1.5128,   None, 0.3682, "12890e02"),
    ("66dafc08bd620d96deda7d526b0e4bfc3b086650", "66dafc08", datetime(2022, 1, 12), 0.4204, 0.9005, 1.5031,   None, 0.3709, "66dafc08"),
    ("a325819b3b03b84bd76ad455e3f9b4600744ba14", "a325819b", datetime(2022, 1, 13), 0.4250, 0.9043, 1.5006,   None, 0.3709, "a325819b"),
    ("8a2c1a610295c007f0222ce737723c341189811d", "8a2c1a61", datetime(2022, 1, 14), 0.4258, 0.9197, 1.5082,   None, 0.3708, "8a2c1a61"),
    ("c6bc79b0cfbc8ce5d6da0d33b68028157d2e93c0", "c6bc79b0", datetime(2022, 1, 14), 0.4270, 0.8828, 1.4818,   None, 0.3701, "c6bc79b0"),
    ("03e1f461c152e4f221fe92c834f2787680cf5772", "03e1f461", datetime(2022, 1, 18), 0.4621, 0.9151, 1.5274, 2.0719, 0.3673, "PR #56"),
    ("9e96d6c412b64fe76a57e7de8af3b00c21d18348", "9e96d6c4", datetime(2022, 1, 19), 0.4659, 0.9056, 1.5061, 2.0561, 0.3825, "v3.0rc1"),
    ("2a98428f9c6fb9bb4302ae09809441bf3e7162b0", "2a98428f", datetime(2022, 2, 11), 0.2996, 0.9091, 1.5168, 2.0349, 0.3824, "PR #317"),
]

df = pd.DataFrame(data=data, columns=columns)


### Timing plot

labels = list(df["tooltip_label"])

fig, ax = plt.subplots()

points = ax.plot(df["date"], df["jensen"], color=COLORS[0], marker='o', label='Jensen / Jimenez')
tooltip = mpld3.plugins.PointLabelTooltip(points[0], labels) # voffset=10, hoffset=10, css=css)
mpld3.plugins.connect(fig, tooltip)

points = ax.plot(df["date"], df["gauss"], color=COLORS[1], marker='o', label='Gauss')
tooltip = mpld3.plugins.PointLabelTooltip(points[0], labels) # voffset=10, hoffset=10, css=css)
mpld3.plugins.connect(fig, tooltip)

points = ax.plot(df["date"], df["gch"], color=COLORS[2], marker='o', label='GCH')
tooltip = mpld3.plugins.PointLabelTooltip(points[0], labels) # voffset=10, hoffset=10, css=css)
mpld3.plugins.connect(fig, tooltip)

points = ax.plot(df["date"], df["cc"], color=COLORS[3], marker='o', label='Cumulative-Curl')
tooltip = mpld3.plugins.PointLabelTooltip(points[0], labels) # voffset=10, hoffset=10, css=css)
mpld3.plugins.connect(fig, tooltip)

ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
ax.set_ylim(0.0, 2.5)
ax.set_xlabel("Commit date")
ax.set_ylabel("Time to solution (s)")
ax.set_title("5x5 Wind Farm Timing Test", size=20)

with open('timing.html', 'w') as f:
    plot_html = mpld3.fig_to_html(fig, figid="timing")
    f.write(plot_html)

# mpld3.show()
# plt.show()


### Code coverage plot

labels = list(df["commit_hash_8char"])

fig, ax = plt.subplots()

points = ax.plot(df["date"], df["code_coverage"], color=COLORS[0], marker='o')
tooltip = mpld3.plugins.PointLabelTooltip(points[0], labels) # voffset=10, hoffset=10, css=css)
mpld3.plugins.connect(fig, tooltip)

ax.grid(True, alpha=0.3)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("Commit date")
ax.set_ylabel("Test coverage as a percentage of Python code")
ax.set_title("Code Coverage", size=20)

with open('codecoverage.html', 'w') as f:
    plot_html = mpld3.fig_to_html(fig, figid="coverage")
    f.write(plot_html)

# mpld3.show()
# plt.show()