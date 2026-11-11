# Experiment code and data

- [`baselines/`](baselines/): pinned upstream commits, code overlays, configs,
  and run scripts for the baselines.
- [`evaluation/`](evaluation/): shared answer collector and judge code.
- [`implementation/`](implementation/): asynchronous write-path source files.
- [`scripts/`](scripts/): build, analysis, probe, and verification scripts.
- [`manifests/`](manifests/): baseline versions, evaluation and runtime
  configurations, and judge-prompt manifests.
- [`results/`](results/): result data, per-question records, and manifests.
- [`SHA256SUMS`](SHA256SUMS): checksums for every file in this directory.

From the repository root:

```bash
python reproducibility/scripts/verify_release.py
sha256sum -c reproducibility/SHA256SUMS
```

API keys are read from environment variables. No credential, local absolute
path, or service URL is stored in the released data.
