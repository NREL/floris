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

columns = ["commit_hash", "commit_hash_8char", "date", "jensen", "gauss", "gch", "cc"]
data = [
    ("df25a9cfacd3d652361d2bd37f568af00acb2631", "df25a9cf", datetime(2021,12, 29), 1.269101, 1.258412, 1.643206, None),
    ("b797390a43298a815f3ff57955cfdc71ecf3e866", "b797390a", datetime(2022, 1,  3), 0.686720, 1.235425, 1.802601, None),
    # ("418d8c3396c8785ea3ea56a317c5dcbea2f88fd6", "418d8c33", datetime(2022, 1,  4), 0.704307, 1.259922, 1.842875, None),
    ("01a02d5f91b2f4a863eebe88a618974b0749d1c4", "01a02d5f", datetime(2022, 1,  4), 0.433525, 0.906505, 1.517391, None),
    ("dd847210082035d43b0273ae63a76a53cb8d2e12", "dd847210", datetime(2022, 1,  6), 0.446517, 0.926989, 1.495375, None),
    ("33779269e98cc882a5f066c462d8ec1eadf37a1a", "33779269", datetime(2022, 1, 10), 0.422206, 0.910346, 1.504330, None),
    ("12890e029a7155b074b9b325d320d1798338e287", "12890e02", datetime(2022, 1, 11), 0.433348, 0.902654, 1.572844, None),
    ("66dafc08bd620d96deda7d526b0e4bfc3b086650", "66dafc08", datetime(2022, 1, 12), 0.433827, 0.906263, 1.541356, None),
    ("a325819b3b03b84bd76ad455e3f9b4600744ba14", "a325819b", datetime(2022, 1, 13), 0.440752, 0.924481, 1.487576, None),
    ("8a2c1a610295c007f0222ce737723c341189811d", "8a2c1a61", datetime(2022, 1, 14), 0.449687, 0.909984, 1.540740, None),
    ("c6bc79b0cfbc8ce5d6da0d33b68028157d2e93c0", "c6bc79b0", datetime(2022, 1, 14), 0.437149, 0.878757, 1.576347, None),
    ("03e1f461c152e4f221fe92c834f2787680cf5772", "03e1f461", datetime(2022, 1, 18), 0.489827, 0.930640, 1.547447, 1.398450),
    ("9e96d6c412b64fe76a57e7de8af3b00c21d18348", "9e96d6c4", datetime(2022, 1, 19), 0.529105, 0.957416, 1.550028, 1.340964)
]

df = pd.DataFrame(data=data, columns=columns)

fig = plt.figure()
plt.plot(df["date"], df["jensen"], color=COLORS[0], marker='o', label='Jensen / Jimenez')
plt.plot(df["date"], df["gauss"], color=COLORS[1], marker='o', label='Gauss')
plt.plot(df["date"], df["gch"], color=COLORS[2], marker='o', label='GCH')
plt.plot(df["date"], df["cc"], color=COLORS[3], marker='o', label='Cumulative-Curl')

plt.title("5x5 Wind Farm Timing Test")
plt.xlabel("Commit date")
plt.ylabel("Time to solution (s)")
plt.ylim(0.0, 2.0)
plt.grid()
plt.legend(loc="lower left")
# plt.show()
print(mpld3.fig_to_html(fig, figid="timing"))





# <div id="1"></div>
# <script>
# function mpld3_load_lib(url, callback){
#   var s = document.createElement('script');
#   s.src = url;
#   s.async = true;
#   s.onreadystatechange = s.onload = callback;
#   s.onerror = function(){console.warn("failed to load library " + url);};
#   document.getElementsByTagName("head")[0].appendChild(s);
# }

# if(typeof(mpld3) !== "undefined" && mpld3._mpld3IsLoaded){
#    !function(mpld3){
       
#        mpld3.draw_figure("1", {"width": 640.0, "height": 480.0, "axes": [{"bbox": [0.125, 0.10999999999999999, 0.775, 0.77], "xlim": [18988.95, 19012.05], "ylim": [0.37986125000000004, 1.31144575], "xdomain": [[2021, 11, 27, 22, 48, 0, 0.0], [2022, 0, 20, 1, 12, 0, 0.0]], "ydomain": [0.37986125000000004, 1.31144575], "xscale": "date", "yscale": "linear", "axes": [{"position": "bottom", "nticks": 6, "tickvalues": null, "tickformat_formatter": "", "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": false}, "visible": true}, {"position": "left", "nticks": 7, "tickvalues": null, "tickformat_formatter": "", "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": false}, "visible": true}], "axesbg": "#FFFFFF", "axesbgalpha": null, "zoomable": true, "id": "el683574752862896", "lines": [{"data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "id": "el683574832514160", "color": "#0000FF", "linewidth": 1.5, "dasharray": "none", "alpha": 1, "zorder": 2, "drawstyle": "default"}], "paths": [], "markers": [{"data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "id": "el683574832514160pts", "facecolor": "#0000FF", "edgecolor": "#0000FF", "edgewidth": 1.0, "alpha": 1, "zorder": 2, "markerpath": [[[0.0, 3.0], [0.7956093000000001, 3.0], [1.5587396123545605, 2.683901074764725], [2.121320343559643, 2.121320343559643], [2.683901074764725, 1.5587396123545605], [3.0, 0.7956093000000001], [3.0, 0.0], [3.0, -0.7956093000000001], [2.683901074764725, -1.5587396123545605], [2.121320343559643, -2.121320343559643], [1.5587396123545605, -2.683901074764725], [0.7956093000000001, -3.0], [0.0, -3.0], [-0.7956093000000001, -3.0], [-1.5587396123545605, -2.683901074764725], [-2.121320343559643, -2.121320343559643], [-2.683901074764725, -1.5587396123545605], [-3.0, -0.7956093000000001], [-3.0, 0.0], [-3.0, 0.7956093000000001], [-2.683901074764725, 1.5587396123545605], [-2.121320343559643, 2.121320343559643], [-1.5587396123545605, 2.683901074764725], [-0.7956093000000001, 3.0], [0.0, 3.0]], ["M", "C", "C", "C", "C", "C", "C", "C", "C", "Z"]]}], "texts": [], "collections": [], "images": [], "sharex": [], "sharey": []}], "data": {"data01": [[18990.0, 1.269101], [18995.0, 0.68672], [18996.0, 0.433525], [18998.0, 0.446517], [19002.0, 0.422206], [19003.0, 0.433348], [19004.0, 0.433827], [19005.0, 0.440752], [19006.0, 0.449687], [19006.0, 0.437149], [19010.0, 0.489827], [19011.0, 0.529105]]}, "id": "el683574752662640", "plugins": [{"type": "reset"}, {"type": "zoom", "button": true, "enabled": false}, {"type": "boxzoom", "button": true, "enabled": false}]});
#    }(mpld3);
# }else if(typeof define === "function" && define.amd){
#    require.config({paths: {d3: "https://d3js.org/d3.v5"}});
#    require(["d3"], function(d3){
#       window.d3 = d3;
#       mpld3_load_lib("https://mpld3.github.io/js/mpld3.v0.5.7.js", function(){
         
#          mpld3.draw_figure("1", {"width": 640.0, "height": 480.0, "axes": [{"bbox": [0.125, 0.10999999999999999, 0.775, 0.77], "xlim": [18988.95, 19012.05], "ylim": [0.37986125000000004, 1.31144575], "xdomain": [[2021, 11, 27, 22, 48, 0, 0.0], [2022, 0, 20, 1, 12, 0, 0.0]], "ydomain": [0.37986125000000004, 1.31144575], "xscale": "date", "yscale": "linear", "axes": [{"position": "bottom", "nticks": 6, "tickvalues": null, "tickformat_formatter": "", "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": false}, "visible": true}, {"position": "left", "nticks": 7, "tickvalues": null, "tickformat_formatter": "", "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": false}, "visible": true}], "axesbg": "#FFFFFF", "axesbgalpha": null, "zoomable": true, "id": "el683574752862896", "lines": [{"data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "id": "el683574832514160", "color": "#0000FF", "linewidth": 1.5, "dasharray": "none", "alpha": 1, "zorder": 2, "drawstyle": "default"}], "paths": [], "markers": [{"data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "id": "el683574832514160pts", "facecolor": "#0000FF", "edgecolor": "#0000FF", "edgewidth": 1.0, "alpha": 1, "zorder": 2, "markerpath": [[[0.0, 3.0], [0.7956093000000001, 3.0], [1.5587396123545605, 2.683901074764725], [2.121320343559643, 2.121320343559643], [2.683901074764725, 1.5587396123545605], [3.0, 0.7956093000000001], [3.0, 0.0], [3.0, -0.7956093000000001], [2.683901074764725, -1.5587396123545605], [2.121320343559643, -2.121320343559643], [1.5587396123545605, -2.683901074764725], [0.7956093000000001, -3.0], [0.0, -3.0], [-0.7956093000000001, -3.0], [-1.5587396123545605, -2.683901074764725], [-2.121320343559643, -2.121320343559643], [-2.683901074764725, -1.5587396123545605], [-3.0, -0.7956093000000001], [-3.0, 0.0], [-3.0, 0.7956093000000001], [-2.683901074764725, 1.5587396123545605], [-2.121320343559643, 2.121320343559643], [-1.5587396123545605, 2.683901074764725], [-0.7956093000000001, 3.0], [0.0, 3.0]], ["M", "C", "C", "C", "C", "C", "C", "C", "C", "Z"]]}], "texts": [], "collections": [], "images": [], "sharex": [], "sharey": []}], "data": {"data01": [[18990.0, 1.269101], [18995.0, 0.68672], [18996.0, 0.433525], [18998.0, 0.446517], [19002.0, 0.422206], [19003.0, 0.433348], [19004.0, 0.433827], [19005.0, 0.440752], [19006.0, 0.449687], [19006.0, 0.437149], [19010.0, 0.489827], [19011.0, 0.529105]]}, "id": "el683574752662640", "plugins": [{"type": "reset"}, {"type": "zoom", "button": true, "enabled": false}, {"type": "boxzoom", "button": true, "enabled": false}]});
#       });
#     });
# }else{
#     mpld3_load_lib("https://d3js.org/d3.v5.js", function(){
#          mpld3_load_lib("https://mpld3.github.io/js/mpld3.v0.5.7.js", function(){
                 
#                  mpld3.draw_figure("1", {"width": 640.0, "height": 480.0, "axes": [{"bbox": [0.125, 0.10999999999999999, 0.775, 0.77], "xlim": [18988.95, 19012.05], "ylim": [0.37986125000000004, 1.31144575], "xdomain": [[2021, 11, 27, 22, 48, 0, 0.0], [2022, 0, 20, 1, 12, 0, 0.0]], "ydomain": [0.37986125000000004, 1.31144575], "xscale": "date", "yscale": "linear", "axes": [{"position": "bottom", "nticks": 6, "tickvalues": null, "tickformat_formatter": "", "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": false}, "visible": true}, {"position": "left", "nticks": 7, "tickvalues": null, "tickformat_formatter": "", "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": false}, "visible": true}], "axesbg": "#FFFFFF", "axesbgalpha": null, "zoomable": true, "id": "el683574752862896", "lines": [{"data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "id": "el683574832514160", "color": "#0000FF", "linewidth": 1.5, "dasharray": "none", "alpha": 1, "zorder": 2, "drawstyle": "default"}], "paths": [], "markers": [{"data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "id": "el683574832514160pts", "facecolor": "#0000FF", "edgecolor": "#0000FF", "edgewidth": 1.0, "alpha": 1, "zorder": 2, "markerpath": [[[0.0, 3.0], [0.7956093000000001, 3.0], [1.5587396123545605, 2.683901074764725], [2.121320343559643, 2.121320343559643], [2.683901074764725, 1.5587396123545605], [3.0, 0.7956093000000001], [3.0, 0.0], [3.0, -0.7956093000000001], [2.683901074764725, -1.5587396123545605], [2.121320343559643, -2.121320343559643], [1.5587396123545605, -2.683901074764725], [0.7956093000000001, -3.0], [0.0, -3.0], [-0.7956093000000001, -3.0], [-1.5587396123545605, -2.683901074764725], [-2.121320343559643, -2.121320343559643], [-2.683901074764725, -1.5587396123545605], [-3.0, -0.7956093000000001], [-3.0, 0.0], [-3.0, 0.7956093000000001], [-2.683901074764725, 1.5587396123545605], [-2.121320343559643, 2.121320343559643], [-1.5587396123545605, 2.683901074764725], [-0.7956093000000001, 3.0], [0.0, 3.0]], ["M", "C", "C", "C", "C", "C", "C", "C", "C", "Z"]]}], "texts": [], "collections": [], "images": [], "sharex": [], "sharey": []}], "data": {"data01": [[18990.0, 1.269101], [18995.0, 0.68672], [18996.0, 0.433525], [18998.0, 0.446517], [19002.0, 0.422206], [19003.0, 0.433348], [19004.0, 0.433827], [19005.0, 0.440752], [19006.0, 0.449687], [19006.0, 0.437149], [19010.0, 0.489827], [19011.0, 0.529105]]}, "id": "el683574752662640", "plugins": [{"type": "reset"}, {"type": "zoom", "button": true, "enabled": false}, {"type": "boxzoom", "button": true, "enabled": false}]});
#             });
#          });
# }
# </script>