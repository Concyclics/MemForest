# Shared evaluation code

`unified_deepseek_judge.py` is the exact answer collector and DeepSeek judge
used to produce the frozen revision labels. It separates answer generation
from judging, accepts an explicit result root, supports LongMemEval and LoCoMo,
and writes per-question labels plus aggregate summaries.

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

The judge key is read from `DEEPSEEK_API_KEY` unless `--judge-api-key` is
provided. Released judged records contain no credentials. The prompt version
used by each result family is recorded in its adjacent prompt manifest.
