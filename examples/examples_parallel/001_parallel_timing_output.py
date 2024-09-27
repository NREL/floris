"""Example: View output from parallel_timing example.
"""

from time import perf_counter as timerpc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


max_workers_options = [2, 16, -1]
n_findex_options = [100, 1000, 10000]
n_turbines_options = [5, 10, 15] # Will be squared!
# Parallelization parameters

DEBUG = True

# First run max_workers tests
timing_data = []
for mw in max_workers_options:
    # Set up models
    n_turbs = 2 if DEBUG else 10 # Will be squared
    n_findex = 1000

    df = pd.read_csv(f"outputs/comptime_maxworkers{mw}_nturbs{n_turbs}_nfindex{n_findex}.csv")

    timing_data.append(df.time.values)

timing_data = np.array(timing_data).T

x = np.arange(len(max_workers_options))
width = 0.2
multiplier = 0

fig, ax = plt.subplots(1,1)

for dat, lab in zip(timing_data.tolist(), df.model.values):
    offset = width * multiplier
    rects = ax.bar(x + offset, dat, width, label=lab)
    ax.bar_label(rects, padding=3, fmt='%.1f')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Max. workers [-]')
ax.set_xticks(x + width, max_workers_options)
ax.set_ylabel('Time [s]')
ax.legend(loc='upper left', ncols=2)
ax.set_yscale('log')
fig.savefig("outputs/max_workers_timing.png", format='png', dpi=300)


# Similar now for n_turbs
timing_data = []
for nt in n_turbines_options:
    # Set up models
    n_findex = 10 if DEBUG else 1000
    max_workers = -1
    df = pd.read_csv(f"outputs/comptime_maxworkers{max_workers}_nturbs{nt}_nfindex{n_findex}.csv")
    timing_data.append(df.time.values)

timing_data = np.array(timing_data).T

x = np.arange(len(n_turbines_options))
width = 0.2
multiplier = 0

fig, ax = plt.subplots(1,1)

for dat, lab in zip(timing_data.tolist(), df.model.values):
    offset = width * multiplier
    rects = ax.bar(x + offset, dat, width, label=lab)
    ax.bar_label(rects, padding=3, fmt='%.1f')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('n_turbines [-]')
ax.set_xticks(x + width, np.array(n_turbines_options)**2)
ax.set_ylabel('Time [s]')
ax.legend(loc='upper left', ncols=2)
ax.set_yscale('log')
fig.savefig("outputs/n_turbines_timing.png", format='png', dpi=300)


# Similar now for n_findex
timing_data = []
for nf in n_findex_options:
    # Set up models
    n_turbs = 2 if DEBUG else 10 # Will be squared
    max_workers = -1
    df = pd.read_csv(f"outputs/comptime_maxworkers{max_workers}_nturbs{n_turbs}_nfindex{nf}.csv")
    timing_data.append(df.time.values)

timing_data = np.array(timing_data).T

x = np.arange(len(n_findex_options))
width = 0.2
multiplier = 0

fig, ax = plt.subplots(1,1)

for dat, lab in zip(timing_data.tolist(), df.model.values):
    offset = width * multiplier
    rects = ax.bar(x + offset, dat, width, label=lab)
    ax.bar_label(rects, padding=3, fmt='%.1f')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('n_findex [-]')
ax.set_xticks(x + width, n_findex_options)
ax.set_ylabel('Time [s]')
ax.legend(loc='upper left', ncols=2)
ax.set_yscale('log')
fig.savefig("outputs/n_findex_timing.png", format='png', dpi=300)


plt.show()




    # # Then run n_turbines tests
    # for nt in n_turbines_options:
    #     # Set up models
    #     n_findex = 10 if DEBUG else 1000
    #     max_workers = 16

    #     set_up_and_run_models(
    #         n_turbs=nt, n_findex=n_findex, max_workers=max_workers
    #     )

    # # Then run n_findex tests
    # for nf in n_findex_options:
    #     # Set up models
    #     n_turbs = 2 if DEBUG else 10 # Will be squared
    #     max_workers = 16

    #     set_up_and_run_models(
    #         n_turbs=n_turbs, n_findex=nf, max_workers=max_workers
    #     )
