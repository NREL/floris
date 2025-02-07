window.BENCHMARK_DATA = {
  "lastUpdate": 1738909642334,
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
          "id": "514cd8d723fe1bb63f8c44e7a02ebc8551d9a43c",
          "message": "Add automatic benchmarking",
          "timestamp": "2025-02-01T22:00:25Z",
          "url": "https://github.com/NREL/floris/pull/1062/commits/514cd8d723fe1bb63f8c44e7a02ebc8551d9a43c"
        },
        "date": 1738909635789,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/bench.py::test_timing_small_farm_set",
            "value": 46.44874041721559,
            "unit": "iter/sec",
            "range": "stddev: 0.00016432790517021385",
            "extra": "mean: 21.529109100004007 msec\nrounds: 40"
          },
          {
            "name": "benchmarks/bench.py::test_timing_small_farm_run",
            "value": 55.45801261387396,
            "unit": "iter/sec",
            "range": "stddev: 0.00014713356721090477",
            "extra": "mean: 18.03165949999855 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/bench.py::test_timing_large_farm_set",
            "value": 26.51384453789869,
            "unit": "iter/sec",
            "range": "stddev: 0.0008843348487630408",
            "extra": "mean: 37.71614480769122 msec\nrounds: 26"
          },
          {
            "name": "benchmarks/bench.py::test_timing_large_farm_run",
            "value": 0.2977259017109979,
            "unit": "iter/sec",
            "range": "stddev: 0.0217698014768123",
            "extra": "mean: 3.358794093000006 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/bench.py::test_timing_het_set",
            "value": 30.848326263381953,
            "unit": "iter/sec",
            "range": "stddev: 0.0003645965999327306",
            "extra": "mean: 32.41666959374179 msec\nrounds: 32"
          },
          {
            "name": "benchmarks/bench.py::test_timing_het_run",
            "value": 111.92171126905036,
            "unit": "iter/sec",
            "range": "stddev: 0.00020722351234771084",
            "extra": "mean: 8.93481692391286 msec\nrounds: 92"
          }
        ]
      }
    ]
  }
}