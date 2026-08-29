# MemForest

Supplementary artifact for the VLDB 2027 submission:

**MemForest: An Efficient Agent Memory System with Hierarchical Temporal Indexing**

This repository provides the reference implementation, revision PDFs,
configuration files, prompts, core logs, and result data used to support the
reported results.

## Paper and Supplement

- [Revised paper with public appendix](reproducibility/paper/MemForest_full_revision.pdf)
- [Revision response, revised paper, and public appendix](reproducibility/paper/MemForest_response_full_revision.pdf)

The length-limited submission PDF is uploaded through the conference system.
The linked public versions include the additional diagnostics referenced by the
response.

## Artifact Scope

This artifact is intended to support transparency and reproducibility of the paper results.

It includes:

- Source code for the MemForest memory substrate and retrieval pipeline.
- Configuration files for the 30B and 4B experimental settings.
- Prompt templates used for extraction, summarization, browsing, answering, and judging.
- Per-question labels and compact result tables for LongMemEval-S and LoCoMo.
- Core manifests, validation summaries, and measured write-path traces.
- A public appendix with additional experimental details.

Reviewers can either inspect the released benchmark outputs directly or run the system with their own OpenAI-compatible model endpoints.

## VLDB Revision Artifacts

The revision package is organized under [`reproducibility/`](reproducibility/).
It includes pinned baseline versions, timestamp and retrieval policies,
baseline reproduction methods, corrected Mem0 results, the Gemma cross-family
benchmark, the Zep Local full-system benchmark, judge-prompt sensitivity
results, and sanitized per-question data.

Start with:

- [`reproducibility/paper/MemForest_response_full_revision.pdf`](reproducibility/paper/MemForest_response_full_revision.pdf)
  for the combined response, revised paper, references, and complete public
  appendix;
- [`reproducibility/paper/MemForest_full_revision.pdf`](reproducibility/paper/MemForest_full_revision.pdf)
  for the revised paper, references, and complete public appendix without the
  response;
- [`reproducibility/BASELINES.md`](reproducibility/BASELINES.md) for baseline
  setup and reproduction boundaries;
- [`reproducibility/PROTOCOL.md`](reproducibility/PROTOCOL.md) for the shared
  evaluation protocol;
- [`reproducibility/RESULTS.md`](reproducibility/RESULTS.md) for updated result
  tables;
- [`reproducibility/implementation/`](reproducibility/implementation/) for the
  online asynchronous per-tree locking and snapshot source path;
- [`reproducibility/scripts/verify_release.py`](reproducibility/scripts/verify_release.py)
  for an offline completeness check.

The CSVs under [`benchmark/`](benchmark/) are retained as the original
submission snapshot. Revision results do not silently overwrite them.

## Overview

MemForest is a persistent memory system for long-context LLM agents. It converts dialogue sessions into canonical facts, organizes memory into scoped temporal trees, and retrieves evidence through tree-level recall followed by tree browsing.

The main design components are:

- **Canonical facts**: stable, temporally anchored write units extracted from dialogue.
- **MemTree**: a scoped temporal index whose leaves preserve time-local evidence and whose internal nodes summarize contiguous intervals.
- **Three complementary tree views**: session trees, entity trees, and scene trees.
- **Localized maintenance**: updates refresh only affected tree paths and derived artifacts.
- **Coarse-to-fine retrieval**: queries first recall relevant trees and then browse from interval summaries to leaf evidence.

## Repository Layout

```text
.
├── benchmark/              # Per-question outputs and judge results
├── reproducibility/        # Revision PDFs, protocols, baseline recipes, and results
├── src/                    # MemForest implementation
│   ├── api/                # OpenAI-compatible chat and embedding clients
│   ├── build/              # Tree construction, routing, and indexing
│   ├── config/             # Configuration dataclasses and YAML files
│   ├── extraction/         # Chunking, extraction, deduplication, fact store
│   ├── forest/             # Multi-user MemForest coordinator
│   ├── logger/             # Per-call and per-step logging
│   ├── prompt/             # Prompt templates
│   ├── query/              # Recall, planner, browser, reranker, answer pipeline
│   └── utils/              # Shared dataclasses and utilities
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- FAISS
- OpenAI-compatible chat-completion endpoint
- OpenAI-compatible embedding endpoint

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

The experiments in the paper use Qwen3 instruction models and Qwen3-Embedding-0.6B served through OpenAI-compatible APIs. Other compatible endpoints can be used by editing the configuration files.

## Configuration

Default configurations are in `src/config/`:

- `src/config/default.yaml`: Qwen3-30B setting
- `src/config/default_4b.yaml`: Qwen3-4B setting

Each configuration specifies:

- chat-completion endpoint
- embedding endpoint
- model name
- API key
- extraction settings
- tree summarization settings
- query and browse settings
- logging settings

To use a different model server, edit the `url`, `model_name`, and `key` fields in the YAML files or override them through the environment variables consumed by `src/config/config.py`.

## Quick Start

The repository exposes a Python API. There is no separate HTTP server required.

```python
from src.forest.memforest import MemForest
from src.config.config import load_default_config

config = load_default_config()  # or load_default_config("default_4b")
forest = MemForest("data/memforest", config=config)

turns = [
    {
        "role": "user",
        "content": "Bob moved from Boston to Davis in May 2023.",
        "timestamp": "2023-05-01T00:00:00Z"
    },
    {
        "role": "user",
        "content": "Bob moved from Davis to Miami in July 2024.",
        "timestamp": "2024-07-01T00:00:00Z"
    }
]

forest.register_user("alice")
forest.ingest_session("alice", "sess_001", turns)

result = forest.query("alice", "Where did Bob live before moving to Miami?")
for fact in result.top_facts:
    print(fact.text)

forest.save("alice")
```

Parallel multi-user operations are also supported:

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

Runtime artifacts are written under the `snapshot_dir` passed to `MemForest`, with one subdirectory per user:

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

The persistent state consists of canonical facts, scope assignments, tree structure, and source-session references. Summaries, embeddings, and retrieval index rows are derived artifacts that can be regenerated from the persistent state.

## Benchmark Outputs

The CSVs under [`benchmark/`](benchmark/) preserve the original-submission
snapshot. Revised three-backbone main-table results are under
[`reproducibility/results/public_judge_three_backbone/`](reproducibility/results/public_judge_three_backbone/):

- [`summary.csv`](reproducibility/results/public_judge_three_backbone/summary.csv)
  contains the aggregate benchmark and category values;
- [`per_question_labels.csv`](reproducibility/results/public_judge_three_backbone/per_question_labels.csv)
  contains the frozen per-question judge labels;
- [`input_manifest.json`](reproducibility/results/public_judge_three_backbone/input_manifest.json)
  and [`validation.json`](reproducibility/results/public_judge_three_backbone/validation.json)
  record the protocol and completeness checks.

Run the API-free release verifier to check row counts, hashes, protocol fields,
and the headline table values:

```bash
python reproducibility/scripts/verify_release.py
```

## Reproducing System Runs

A model-serving rerun requires:

- the original LongMemEval-S and LoCoMo benchmark data;
- OpenAI-compatible endpoints for the chat model and embedding model;
- sufficient GPU resources for serving the selected models;
- the configuration files in `src/config/`.

The main paper uses:

- Qwen3-30B-A3B-Instruct-2507 and Qwen3-4B-Instruct-2507 as answer/build models;
- Qwen3-Embedding-0.6B as the embedding model;
- DeepSeek-V4-Flash as the LLM judge;
- vLLM with FlashAttention for model serving.

Because benchmark reruns depend on model-serving infrastructure and external
benchmark data, the release also provides core logs, per-question labels,
manifests, and result tables for API-free verification.

## Module Overview

- `src/api/`: OpenAI-compatible wrappers for chat completion and embedding, with optional API-call logging.
- `src/build/`: tree construction, session/entity/scene routing, summary caching, tree primitives, tree storage, and FAISS node indexing.
- `src/config/`: YAML loading and typed configuration dataclasses.
- `src/extraction/`: chunking, extraction orchestration, embedding-based deduplication, and fact-store management.
- `src/forest/`: public MemForest coordinator, per-user forest state, and forest merge utilities.
- `src/logger/`: API-call latency/token logging and extraction-stage logging.
- `src/prompt/`: prompt builders for extraction, deduplication, tree summarization, answer generation, and judging.
- `src/query/`: forest recall, optional planner, tree browser, reranker, and answer pipeline.
- `src/utils/`: shared dataclasses, time helpers, and text utilities.

## Notes for Reviewers

The fastest way to verify the revised main results is to run
`python reproducibility/scripts/verify_release.py` and inspect
`reproducibility/results/public_judge_three_backbone/summary.csv`.

The fastest way to inspect the system implementation is to start from:

- `src/forest/memforest.py`: public API and multi-user coordination
- `src/extraction/pipeline.py`: parallel extraction pipeline
- `src/build/tree_builder.py`: MemTree construction and refresh
- `src/query/pipeline.py`: recall, browse, rerank, and answer workflow
- `src/forest/forest_merge.py`: migration and merge support

## Reproducibility Status

The artifact supports three levels of reproducibility:

1. **Direct result verification**: use the revision summaries, per-question
   labels, manifests, and offline verifier under `reproducibility/`.
2. **Pipeline inspection**: inspect the source code, prompts, configurations, and logging utilities.
3. **Model-serving rerun**: rerun released components with compatible model
   services and the original benchmark data.

The first level is lightweight and does not require GPUs. Model-serving reruns
require external compute and benchmark access.

## License

This artifact is released for academic review and research use. Please see `LICENSE` for details.

## Citation

If this work is accepted, please cite the camera-ready PVLDB version. A BibTeX entry will be added after publication.
