# MemForest

Supplementary artifact for the VLDB 2027 submission:

**MemForest: An Efficient Agent Memory System with Hierarchical Temporal Indexing**

This repository provides the reference implementation, supplementary appendix, configuration files, prompts, and benchmark outputs used to support the results reported in the paper.

## Paper and Supplement

- Main submission PDF: `paper/MemForest_VLDB_Final_12.pdf`
- Supplementary appendix: `paper/MemForest_Supplementary_Appendix.pdf`

The main paper follows the 12-page PVLDB research-track submission format. The supplementary appendix contains additional derivations, prompts, ablation details, extended result tables, and implementation notes.

## Artifact Scope

This artifact is intended to support transparency and reproducibility of the paper results.

It includes:

- Source code for the MemForest memory substrate and retrieval pipeline.
- Configuration files for the 30B and 4B experimental settings.
- Prompt templates used for extraction, summarization, browsing, answering, and judging.
- Per-question benchmark outputs for LongMemEval-S and LoCoMo.
- CSV files containing answer and judge results used to compute the reported pass@1 and pass@1--8 metrics.
- Supplementary appendix with additional experimental details.

Reviewers can either inspect the released benchmark outputs directly or run the system with their own OpenAI-compatible model endpoints.

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
├── paper/                  # Main submission PDF and supplementary appendix
├── benchmark/              # Per-question outputs and judge results
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
├── sqlite/        # Fact and turn store
├── index/         # FAISS indexes and sidecar metadata
├── trees/         # Serialized session/entity/scene trees
└── logs/          # Retrieval traces and per-step API logs
```

The persistent state consists of canonical facts, scope assignments, tree structure, and source-session references. Summaries, embeddings, and retrieval index rows are derived artifacts that can be regenerated from the persistent state.

## Benchmark Outputs

Per-question benchmark outputs are provided under `benchmark/`. Each row contains the question, gold answer, model answer, and LLM-judge verdicts for repeated runs.

| Benchmark | Model size | File |
|---|---|---|
| LoCoMo | Qwen3-30B | `benchmark/locomo_per_question_30b.csv` |
| LoCoMo | Qwen3-4B | `benchmark/locomo_per_question_4b.csv` |
| LongMemEval-S | Qwen3-30B | `benchmark/longmemeval_per_question_30b.csv` |
| LongMemEval-S | Qwen3-4B | `benchmark/longmemeval_per_question_4b.csv` |

Expected columns:

```text
method, model_size, question_type, qid, question, gold_answer,
answer_1, ..., answer_8,
judge_1, ..., judge_8
```

The `judge_*` columns are binary correctness labels produced by the LLM judge. The paper reports pass@1 as the main metric and reports pass@1--8 curves in the supplementary appendix.

## Reproducing Accuracy Tables

The reported accuracy numbers can be recomputed from the benchmark CSV files by aggregating the judge columns.

For pass@1:

```python
import pandas as pd

df = pd.read_csv("benchmark/longmemeval_per_question_30b.csv")
score = df.groupby("method")["judge_1"].mean() * 100
print(score.sort_values(ascending=False))
```

For pass@8:

```python
judge_cols = [f"judge_{i}" for i in range(1, 9)]
df["pass_at_8"] = df[judge_cols].max(axis=1)
score = df.groupby("method")["pass_at_8"].mean() * 100
print(score.sort_values(ascending=False))
```

For per-category accuracy:

```python
score = (
    df.groupby(["method", "question_type"])["judge_1"]
      .mean()
      .mul(100)
      .reset_index()
)
print(score)
```

## Reproducing System Runs

A full end-to-end rerun requires:

- the original LongMemEval-S and LoCoMo benchmark data;
- OpenAI-compatible endpoints for the chat model and embedding model;
- sufficient GPU resources for serving the selected models;
- the configuration files in `src/config/`.

The main paper uses:

- Qwen3-30B-A3B-Instruct-2507 and Qwen3-4B-Instruct-2507 as answer/build models;
- Qwen3-Embedding-0.6B as the embedding model;
- DeepSeek-V3.2 as the LLM judge;
- vLLM with FlashAttention for model serving.

Because full benchmark reruns depend on model-serving infrastructure and external benchmark licenses, this repository provides the per-question outputs used in the paper so that the reported tables can be independently checked from the released CSV files.

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

The fastest way to verify the main reported accuracy numbers is to inspect the CSV files in `benchmark/` and recompute pass@1 or pass@8 using the snippets above.

The fastest way to inspect the system implementation is to start from:

- `src/forest/memforest.py`: public API and multi-user coordination
- `src/extraction/pipeline.py`: parallel extraction pipeline
- `src/build/tree_builder.py`: MemTree construction and refresh
- `src/query/pipeline.py`: recall, browse, rerank, and answer workflow
- `src/forest/forest_merge.py`: migration and merge support

## Reproducibility Status

The artifact supports three levels of reproducibility:

1. **Direct result verification**: use the released CSV files in `benchmark/` to recompute the reported pass@1 and pass@8 numbers.
2. **Pipeline inspection**: inspect the source code, prompts, configurations, and logging utilities.
3. **Full rerun**: rerun the system with compatible model-serving infrastructure and the original benchmark data.

The first level is lightweight and does not require GPUs. The full rerun requires external model-serving resources.

## License

This artifact is released for academic review and research use. Please see `LICENSE` for details.

## Citation

If this work is accepted, please cite the camera-ready PVLDB version. A BibTeX entry will be added after publication.