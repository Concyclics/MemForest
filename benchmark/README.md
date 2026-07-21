# Submitted benchmark snapshot

The four CSV files in this directory are the per-question artifacts released
with the original submission. They are retained unchanged so that the original
paper tables remain auditable.

Revision results are under [`../reproducibility/results/`](../reproducibility/results/).
In particular, the revised Mem0 LongMemEval results replace the submitted Mem0
rows because the submitted adapter exposed ingestion-time timestamps in the
retrieved context. See
[`../reproducibility/BASELINES.md`](../reproducibility/BASELINES.md#mem0-corrected-rerun)
for the root-cause analysis and corrected protocol.

Do not combine the submitted and revised files into one table without using
the release status recorded in
[`../reproducibility/manifests/revision_release.json`](../reproducibility/manifests/revision_release.json).
