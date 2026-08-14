# Published results boundary

This directory may contain only small aggregate result tables and static report artifacts that comply with the dataset license.
Raw Criteo rows, prepared rows, row-level predictions, checkpoints, trained model weights, and ordinary run directories must remain outside Git.

Every published result must identify its protocol lock, code commit, environment hash, prepared-data manifest hash, seeds, compute accounting, uncertainty method, and limitations.
A result is not publication eligible when it was produced from a dirty-tree override, an incomplete run, an unmatched compute budget, or an unlocked final protocol.

HTML and PNG renderings under `results/published/` are ignored by default.
Small reviewed aggregate CSV, JSON, or Parquet tables may be committed only after the repository's publication audit passes.
