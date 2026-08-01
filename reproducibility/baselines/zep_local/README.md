# Zep Local runner

The runner follows the public MemoryData Zep Local implementation while adding
dataset conversion, resumable per-conversation graph builds, instrumentation,
and completeness checks.

Prepare the pinned dependencies under the repository root:

```bash
mkdir -p external
git clone https://github.com/OpenDataBox/MemoryData.git external/MemoryData
git -C external/MemoryData checkout c63391c128e33eedb91115edf689f12acf4bbc63
git clone https://github.com/getzep/graphiti.git external/graphiti-0.24.1
git -C external/graphiti-0.24.1 checkout d2654003ffc11821bce73c493162a40181b23504
```

Start Neo4j 5.26.2 plus OpenAI-compatible generation and embedding endpoints,
then invoke `run_benchmark.py --help`. Required arguments identify the dataset,
model lane, run root, LLM endpoint, embedding endpoint, Neo4j URI, and
concurrency. The runner:

1. ingests ordered raw episodes with benchmark `reference_time`;
2. reuses a graph only when protocol hash and episode count match;
3. clears a graph when the persisted episode prefix and progress marker differ;
4. executes native Graphiti search and the MemoryData schema-aware answer path;
5. saves build, query, answer, call/token, and completeness summaries.

Use `summarize_run.py --run-root <path>` to aggregate completed lanes. Judge the
resulting frozen answer JSONL with the released strict prompt recorded beside the
released result summaries.

To reproduce the paper's native retrieval-budget table, run:

```bash
python reproducibility/scripts/summarize_zep_native_budget.py \
  --run-root <path> \
  --output reproducibility/results/zep_local/native_budget_summary.csv \
  --manifest reproducibility/results/zep_local/native_budget_manifest.json
```

The script tokenizes the exact serialized context and excludes the answer
instruction and question.
