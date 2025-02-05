window.BENCHMARK_DATA = {
  "lastUpdate": 1738727516712,
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
          "id": "995bf40bc172ad8656a0a24703a26757aa4f4168",
          "message": "Add automatic benchmarking to FLORIS",
          "timestamp": "2025-02-01T22:00:25Z",
          "url": "https://github.com/NREL/floris/pull/1060/commits/995bf40bc172ad8656a0a24703a26757aa4f4168"
        },
        "date": 1738726534579,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/bench.py::test_benchmark_set",
            "value": 48.465853905748055,
            "unit": "iter/sec",
            "range": "stddev: 0.0010670573398481188",
            "extra": "mean: 20.633083282607757 msec\nrounds: 46"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_run",
            "value": 330.2513140553071,
            "unit": "iter/sec",
            "range": "stddev: 0.000038460724400155964",
            "extra": "mean: 3.027997035713627 msec\nrounds: 56"
          }
        ]
      },
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
          "id": "8a0b8b28164ce8f73bc4f814812dac0186ed3b4b",
          "message": "Add automatic benchmarking to FLORIS",
          "timestamp": "2025-02-01T22:00:25Z",
          "url": "https://github.com/NREL/floris/pull/1060/commits/8a0b8b28164ce8f73bc4f814812dac0186ed3b4b"
        },
        "date": 1738727511280,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/bench.py::test_benchmark_set",
            "value": 46.2567800402261,
            "unit": "iter/sec",
            "range": "stddev: 0.00009531646535682529",
            "extra": "mean: 21.618452454545565 msec\nrounds: 44"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_run",
            "value": 333.5992609475571,
            "unit": "iter/sec",
            "range": "stddev: 0.00008388117658023402",
            "extra": "mean: 2.9976085593223276 msec\nrounds: 59"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_100_turbine_run",
            "value": 0.3174028434431914,
            "unit": "iter/sec",
            "range": "stddev: 0.009491877896400965",
            "extra": "mean: 3.1505703891999928 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}