# EverMemOS reproduction overlay

Upstream: `https://github.com/EverMind-AI/EverMemOS`
Commit: `539db77d5cc804c875246e34611fd266bf8c1e5d`

`upstream.patch` contains the complete local-evaluation overlay used by our
controlled EverMemOS runs. It adds OpenAI-compatible local model support,
bounded parallel execution, complete per-stage logging, benchmark timestamp
propagation, a global final retrieval budget, and resumable add/search/answer
stages. The patch also contains the shared EverMemOS-harness changes used by
the corrected Mem0 run; the standalone corrected Mem0 adapter is published in
[`../mem0/`](../mem0/).

Prepare a checkout:

```bash
git clone https://github.com/EverMind-AI/EverMemOS.git
git -C EverMemOS checkout 539db77d5cc804c875246e34611fd266bf8c1e5d
EVERMEMOS_ROOT=$PWD/EverMemOS \
  bash /path/to/MemForest/reproducibility/baselines/evermemos/scripts/prepare.sh
```

The preparation script applies the patch and installs `locomo_all.yaml` under
`evaluation/config/datasets/` and system YAML files under
`evaluation/config/systems/`. Then run:

```bash
EVERMEMOS_ROOT=$PWD/EverMemOS \
DATASET=longmemeval \
SYSTEM=evermemos_longmemeval_local \
OUTPUT_DIR=$PWD/runs/evermemos_qwen30_lme \
bash /path/to/MemForest/reproducibility/baselines/evermemos/scripts/run.sh
```

Available system configurations cover Qwen3-4B, Qwen3-30B-A3B, and Gemma 4
12B on both LongMemEval-S and LoCoMo. Set `EVAL_LLM_BASE_URL`,
`VECTORIZE_BASE_URL`, `VECTORIZE_MODEL`, and API-key environment variables for
the selected OpenAI-compatible services. All released configurations use
`response_top_k: 10`.
The preparation script is idempotent with respect to an already-applied patch:
it checks the patch first and only applies it when the clean pinned checkout
still needs the overlay.
