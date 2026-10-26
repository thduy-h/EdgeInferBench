# v0.1.1 standard batch-1 measurements

These are actual standard-mode measurements on the AMD Ryzen 7 4800H development
host. Each row used five fresh child processes, 20 warmups, 200 timed iterations,
FP32 NCHW input, seed 42, and four requested CPU threads. Existing v0.1.0 artifacts
were used as read-only inputs. The runs cover batch size 1 only; they are not a full
replacement for the v0.1.0 batch-size table.

The aggregate is the median of five trial summaries. Raw trials, P50/throughput
variability, bootstrap intervals, process IDs, RSS fields, and input checksums are in
the individual result JSON files.
