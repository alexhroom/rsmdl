import time

from rsmdl import calc_histogram, Data

files = [f"SIM0000000{n}.nxs" for n in range(1,4)]

stats = 1
n_filters = 0
for file in files:
    print("\nFile: ", file)
    data = Data(file)
    avg_calc_time = 0
    avg_run_time = 0
    for _ in range(0, stats):
        start_time = time.time()
        hist, n, calc_time = calc_histogram(data, n_filters)
        duration = time.time() - start_time
        avg_run_time += duration
        avg_calc_time += calc_time 
    avg_run_time /= stats
    avg_calc_time /= stats
    print("  Average run time: ", avg_run_time * 1e3, " ms",
          "\n  Average calc time: ", avg_calc_time, " ms",
          "\n  Number of events:", n)
