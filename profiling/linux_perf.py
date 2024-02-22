
from contextlib import contextmanager
from os import getpid
from resource import getrusage, RUSAGE_SELF
from signal import SIGINT
from subprocess import Popen
from time import (
    perf_counter,
    sleep,
    time,
)


# Additional events described here:
# https://www.brendangregg.com/perf.html#SoftwareEvents
events = [
    "instructions",
    "cache-references",
    "cache-misses",
    "avx_insts.all",
]

@contextmanager
def perf():
    """
    Benchmark this process with Linux's perf util.

    Example usage:

        with perf():
            x = run_some_code()
            more_code(x)
            all_this_code_will_be_measured()
    """
    p = Popen([
        # Run perf stat
        "perf", "stat",
        # for the current Python process
        "-p", str(getpid()),
        # record the list of events mentioned above
        "-e", ",".join(events)
    ])

    # Ensure perf has started before running more
    # Python code. This will add ~0.1 to the elapsed
    # time reported by perf, so we also track elapsed
    # time separately.
    sleep(0.1)
    start = time()
    try:
        yield
    finally:
        print(f"Elapsed (seconds): {time() - start}")
        print("Peak memory (MiB):",
            int(getrusage(RUSAGE_SELF).ru_maxrss / 1024))
        p.send_signal(SIGINT)
