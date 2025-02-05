window.BENCHMARK_DATA = {
  "lastUpdate": 1738726540033,
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
      }
    ]
  }
}