# Evaluation code

`unified_deepseek_judge.py` collects answers and applies the DeepSeek judge for
LongMemEval and LoCoMo, writing per-question labels and summaries.

```bash
export DEEPSEEK_API_KEY=...
python reproducibility/evaluation/unified_deepseek_judge.py \
  --root /path/to/completed/baseline/runs \
  --out-dir /path/to/judged-output \
  --judge-url https://api.deepseek.com/v1 \
  --judge-model deepseek-chat \
  --prompt-version appendix \
  --benchmarks longmemeval locomo \
  --workers 32
```

The prompt commits, hashes, and judge settings of the main-table labels are
recorded in `../results/public_judge_three_backbone/input_manifest.json`.
