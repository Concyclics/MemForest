# MemForest

Code and experiment data for the ICDE 2027 second-round submission:

**MemForest: Write-Efficient Agent Memory with Decoupled Extraction and Hierarchical View Maintenance**

## Repository Layout

```text
.
├── benchmark/              # Per-question answers of the Qwen3-4B and Qwen3-30B runs
├── reproducibility/        # Baseline code, evaluation code, scripts, manifests, and result data
├── src/                    # MemForest implementation
│   ├── api/                # OpenAI-compatible chat and embedding clients
│   ├── build/              # Tree construction, routing, and indexing
│   ├── config/             # Configuration dataclasses and YAML files
│   ├── extraction/         # Chunking, extraction, deduplication, fact store
│   ├── forest/             # Multi-user MemForest coordinator
│   ├── logger/             # Per-call and per-step logging
│   ├── prompt/             # Prompt templates
│   ├── query/              # Recall, browse, rerank, and answer pipeline
│   └── utils/              # Shared dataclasses and utilities
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- FAISS
- OpenAI-compatible chat-completion endpoint
- OpenAI-compatible embedding endpoint

```bash
python -m pip install -r requirements.txt
```

## Configuration

Default configurations are in `src/config/`:

- `src/config/default.yaml`: Qwen3-30B setting
- `src/config/default_4b.yaml`: Qwen3-4B setting

Edit the `url`, `model_name`, and `key` fields, or override them through the
environment variables read by `src/config/config.py`.

## Quick Start

```python
from src.forest.memforest import MemForest
from src.config.config import load_default_config

config = load_default_config()  # or load_default_config("default_4b")
forest = MemForest("data/memforest", config=config)

turns = [
    {"role": "user", "content": "Bob moved from Boston to Davis in May 2023.",
     "timestamp": "2023-05-01T00:00:00Z"},
    {"role": "user", "content": "Bob moved from Davis to Miami in July 2024.",
     "timestamp": "2024-07-01T00:00:00Z"},
]

forest.register_user("alice")
forest.ingest_session("alice", "sess_001", turns)

result = forest.query("alice", "Where did Bob live before moving to Miami?")
for fact in result.top_facts:
    print(fact.text)

forest.save("alice")
```

Multi-user operations:

```python
forest.ingest_parallel([
    {"user_id": "alice", "session_id": "s1", "turns": turns_a},
    {"user_id": "bob", "session_id": "s1", "turns": turns_b},
])

forest.query_parallel([
    {"user_id": "alice", "question": "..."},
    {"user_id": "bob", "question": "..."},
])
```

## Runtime Data Layout

```text
<snapshot_dir>/<user_id>/
├── facts/                  # Canonical facts and their FAISS index
├── trees/                  # Serialized session/entity/scene trees
├── node_index/             # Tree-node embeddings and FAISS index
├── session_registry.json   # Source-session metadata
├── summary_cache.json      # Derived summary cache
├── cell_store.json         # Persisted cell records
└── metadata.json           # Snapshot metadata
```

## Experiment Data

Main-table per-question labels and summaries are in
[`reproducibility/results/public_judge_three_backbone/`](reproducibility/results/public_judge_three_backbone/).
See [`reproducibility/README.md`](reproducibility/README.md) for the other
directories. The API-free checker verifies row counts, hashes, and protocol
fields:

```bash
python reproducibility/scripts/verify_release.py
```

## License

See `LICENSE`.
