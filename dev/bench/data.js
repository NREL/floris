window.BENCHMARK_DATA = {
  "lastUpdate": 1738780628265,
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
          "id": "0d6b57fd75f1263e8bbee1de5258c745832dc9c5",
          "message": "Add automatic benchmarking to FLORIS",
          "timestamp": "2025-02-01T22:00:25Z",
          "url": "https://github.com/NREL/floris/pull/1060/commits/0d6b57fd75f1263e8bbee1de5258c745832dc9c5"
        },
        "date": 1738780621689,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/bench.py::test_benchmark_set",
            "value": 48.160173433241965,
            "unit": "iter/sec",
            "range": "stddev: 0.0013657808980077188",
            "extra": "mean: 20.764044826087822 msec\nrounds: 46"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_run",
            "value": 332.2464689955685,
            "unit": "iter/sec",
            "range": "stddev: 0.0000328453820786579",
            "extra": "mean: 3.0098137777751313 msec\nrounds: 63"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_100_turbine_run",
            "value": 0.3173304543741947,
            "unit": "iter/sec",
            "range": "stddev: 0.024689119422210946",
            "extra": "mean: 3.1512890938000053 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_het_set",
            "value": 31.48720442665588,
            "unit": "iter/sec",
            "range": "stddev: 0.0001732709547367211",
            "extra": "mean: 31.758932500004278 msec\nrounds: 32"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_het_run",
            "value": 111.04148678639793,
            "unit": "iter/sec",
            "range": "stddev: 0.0001743059962009105",
            "extra": "mean: 9.005643106379006 msec\nrounds: 94"
          }
        ]
      }
    ]
  }
}