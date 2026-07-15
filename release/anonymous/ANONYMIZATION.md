# Double-Blind Release Policy

This artifact is generated from an explicit allowlist and starts with fresh Git
history. It does not contain the development repository's `.git` directory.

The export process applies the following controls:

- excludes development archives, superseded v1-v4 result documents, local
  outputs, review triage, and workstation notes;
- excludes source manifests that contain identifiable internal commit history;
- replaces remaining internal commit references with
  `ANONYMIZED_INTERNAL_COMMIT`;
- removes source remote, author, email, home-path, and precise host metadata;
- retains third-party ABIDES and official MAPPO commits for dependency
  reproducibility;
- recomputes a release-local SHA-256 manifest and row counts;
- requires the only Git author identity to be `Anonymous Authors
  <anonymous@invalid>`.

The omitted internal provenance can be released after the double-blind review
period. Scientific failures and limitations are not removed: IPPO remains
inconclusive, the low-dose reversal remains visible, the forced-deviation test
restricts the claim to high-spread coordination, and external mechanism
adaptations that failed frozen integrity gates remain descriptive only.
