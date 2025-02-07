window.BENCHMARK_DATA = {
  "lastUpdate": 1738948329485,
  "repoUrl": "https://github.com/NREL/floris",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
      {
        "commit": {
          "author": {
            "name": "NREL",
            "username": "NREL"
          },
          "committer": {
            "name": "NREL",
            "username": "NREL"
          },
          "id": "17d992f76fadb58a2e5382ec0b82c9c42607186f",
          "message": "Add automatic benchmarking",
          "timestamp": "2025-02-01T22:00:25Z",
          "url": "https://github.com/NREL/floris/pull/1062/commits/17d992f76fadb58a2e5382ec0b82c9c42607186f"
        },
        "date": 1738948326654,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/bench.py::test_timing_small_farm_set",
            "value": 46.88557351265387,
            "unit": "iter/sec",
            "range": "stddev: 0.00008554100597365544",
            "extra": "mean: 21.328522295458573 msec\nrounds: 44"
          },
          {
            "name": "benchmarks/bench.py::test_timing_small_farm_run",
            "value": 54.52107877300606,
            "unit": "iter/sec",
            "range": "stddev: 0.0001313022927268959",
            "extra": "mean: 18.341529964280717 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/bench.py::test_timing_large_farm_set",
            "value": 27.302069379462058,
            "unit": "iter/sec",
            "range": "stddev: 0.0007644331868898508",
            "extra": "mean: 36.62726023076656 msec\nrounds: 26"
          },
          {
            "name": "benchmarks/bench.py::test_timing_large_farm_run",
            "value": 0.2993398120074651,
            "unit": "iter/sec",
            "range": "stddev: 0.01807672081635735",
            "extra": "mean: 3.3406849336000164 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/bench.py::test_timing_het_set",
            "value": 30.879214604004865,
            "unit": "iter/sec",
            "range": "stddev: 0.0017301261136506484",
            "extra": "mean: 32.384243343750896 msec\nrounds: 32"
          },
          {
            "name": "benchmarks/bench.py::test_timing_het_run",
            "value": 111.76303333946917,
            "unit": "iter/sec",
            "range": "stddev: 0.00006485920285697893",
            "extra": "mean: 8.947502319148754 msec\nrounds: 94"
          }
        ]
      }
    ]
  }
}