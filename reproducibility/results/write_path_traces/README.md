# Native write-trace provenance

`summary.csv` contains the Qwen3-30B coordinates used for the write-rate axis
in Figure 1 and the corresponding write-path table. The rate is source turns
divided by observed native per-memory-instance elapsed time.

The LoCoMo rows are a matched `conv-43` probe with cross-instance concurrency
one. The LongMemEval rows retain the source scope of each measurement:
MemForest is the mean of three representative traces, EverMemOS/Mem0/MemoryOS
are means over 500 benchmark-harness per-instance traces, and Zep Local is one
isolated marker. These rates must not be interpreted as sustained concurrent
benchmark throughput. `manifest.json` records the source hashes and scope.
