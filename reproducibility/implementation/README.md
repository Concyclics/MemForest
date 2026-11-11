# Asynchronous write path

[`async_core/`](async_core/) holds the asynchronous `Forest` / `AsyncBPlusTree`
source files, copied byte-for-byte from
[`Concyclics/MemoryForest` commit `dbcbeef6050e49b08ee4de60540a789ac3229a0c`](https://github.com/Concyclics/MemoryForest/tree/dbcbeef6050e49b08ee4de60540a789ac3229a0c):

- [`async_core/bplustree.py`](async_core/bplustree.py): per-tree asynchronous
  reader-writer lock, shared reads, exclusive insertion and refresh, and
  read-locked state export.
- [`async_core/forest.py`](async_core/forest.py): groups writes by tree, runs
  distinct-tree insert and refresh tasks concurrently, snapshots the dirty-tree
  set, refreshes root vectors, and publishes snapshot directories by rename.
