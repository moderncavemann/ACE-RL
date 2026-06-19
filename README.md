# ACE-RL Revision Workspace

This repository root is now the clean workspace for the post-rejection
revision.

On this workstation, the original submission-era project has been archived in:

- `archive/v1_rejected_submission/`

That archive is intentionally ignored by git. Treat it as read-only local
evidence unless a file is deliberately migrated into `revision/`.

## Current Layout

- `archive/v1_rejected_submission/` - local-only v1 rejected-submission assets
  moved from the old repository root. This directory is ignored by git.
- `revision/` - clean working area for the revised paper/codebase.
- `REVISION_TRIAGE.md` - current evidence audit and rewrite checklist.
- `docs/` - lightweight notes for the revised project.

## Working Rule

Do not reuse v1 paper claims until they are re-audited against the relevant
experiment backend.

The main known issue is that the strong mock/calibrated result in v1 does not
carry over to the long Gymnasium LOB runs. The revision should either repair
that gap experimentally or change the paper claim.

## Next Steps

1. Add the rejection reviews to `REVISION_TRIAGE.md`.
2. Decide the revised core claim before porting code.
3. Migrate only the minimum needed implementation from the v1 archive.
4. Rebuild reproducibility from small tests first, then long runs.
