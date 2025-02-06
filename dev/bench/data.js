window.BENCHMARK_DATA = {
  "lastUpdate": 1738883450007,
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
          "id": "0c9613586c7e5f0ab58242c125826679937ce1c6",
          "message": "Add automatic benchmarking",
          "timestamp": "2025-02-01T22:00:25Z",
          "url": "https://github.com/NREL/floris/pull/1062/commits/0c9613586c7e5f0ab58242c125826679937ce1c6"
        },
        "date": 1738880976961,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/bench.py::test_benchmark_set",
            "value": 47.372317450683376,
            "unit": "iter/sec",
            "range": "stddev: 0.00021533412133091642",
            "extra": "mean: 21.109374711107243 msec\nrounds: 45"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_run",
            "value": 329.5616283588456,
            "unit": "iter/sec",
            "range": "stddev: 0.0001551725977825971",
            "extra": "mean: 3.0343338360712995 msec\nrounds: 61"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_100_turbine_run",
            "value": 0.31052613800312806,
            "unit": "iter/sec",
            "range": "stddev: 0.03438378824276267",
            "extra": "mean: 3.220340826799986 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_het_set",
            "value": 29.291440064830784,
            "unit": "iter/sec",
            "range": "stddev: 0.0010023886514837395",
            "extra": "mean: 34.139666666667765 msec\nrounds: 30"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_het_run",
            "value": 106.2876656630853,
            "unit": "iter/sec",
            "range": "stddev: 0.00027168242880971444",
            "extra": "mean: 9.40842941428254 msec\nrounds: 70"
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
          "id": "5ed78a8b5349f43ef2dedca42096015686bdb065",
          "message": "Add automatic benchmarking",
          "timestamp": "2025-02-01T22:00:25Z",
          "url": "https://github.com/NREL/floris/pull/1062/commits/5ed78a8b5349f43ef2dedca42096015686bdb065"
        },
        "date": 1738883443743,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/bench.py::test_benchmark_set",
            "value": 47.845189095113746,
            "unit": "iter/sec",
            "range": "stddev: 0.00023564942400891207",
            "extra": "mean: 20.900742977774673 msec\nrounds: 45"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_run",
            "value": 331.09996792360465,
            "unit": "iter/sec",
            "range": "stddev: 0.000031817436354783656",
            "extra": "mean: 3.020235870970341 msec\nrounds: 62"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_100_turbine_run",
            "value": 0.30995236851362146,
            "unit": "iter/sec",
            "range": "stddev: 0.01481138982091962",
            "extra": "mean: 3.22630217279999 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_het_set",
            "value": 30.530469858498662,
            "unit": "iter/sec",
            "range": "stddev: 0.0012461595054808637",
            "extra": "mean: 32.75416345161925 msec\nrounds: 31"
          },
          {
            "name": "benchmarks/bench.py::test_benchmark_het_run",
            "value": 110.13023436342718,
            "unit": "iter/sec",
            "range": "stddev: 0.0002009771631237547",
            "extra": "mean: 9.08015864835104 msec\nrounds: 91"
          }
        ]
      }
    ]
  }
}