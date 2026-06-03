import time

from stats import calc_histogram

files = ["HIFI00206202.nxs"]

stats = 1 
for file in files:
    print("\nFile: ", file)
    avg_calc_time = 0
    avg_run_time = 0
    for _ in range(0, stats):
        start_time = time.time()
        hist, n, calc_time = calc_histogram(file)
        duration = time.time() - start_time
        avg_run_time += duration
        avg_calc_time += calc_time 
    avg_run_time /= stats
    avg_calc_time /= stats
    print("  Average run time: ", avg_run_time * 1e3, " ms",
          "\n  Average calc time: ", avg_calc_time, " ms",
          "\n  Number of events:", n)
