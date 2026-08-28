# Evaluated asynchronous write path

The R3-W3 online concurrency and snapshot description uses the asynchronous
`Forest` / `AsyncBPlusTree` implementation frozen in [`async_core/`](async_core/).
The write-path timing experiment uses the repository's batch `TreeBuilder`;
it measures dirty-tree deduplication and parallel refresh but does not generate
concurrent session writers. The two asynchronous source files are copied
byte-for-byte from
[`Concyclics/MemoryForest` commit `dbcbeef6050e49b08ee4de60540a789ac3229a0c`](https://github.com/Concyclics/MemoryForest/tree/dbcbeef6050e49b08ee4de60540a789ac3229a0c):

- [`async_core/bplustree.py`](async_core/bplustree.py) implements the per-tree
  asynchronous reader--writer lock, shared read operations, exclusive tree
  insertion/refresh, and read-locked state export.
- [`async_core/forest.py`](async_core/forest.py) groups writes by tree, schedules
  distinct-tree insert and refresh tasks concurrently, snapshots the dirty-tree
  set, refreshes root vectors, and publishes completed temporary snapshot
  directories by rename.

The repository's top-level `src/forest/UserForest` is the original submission
API wrapper and retains a coarser compatibility lock. It is not the online
asynchronous path used for the R3-W3 lock-granularity semantics.
