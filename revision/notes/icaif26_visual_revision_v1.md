# ICAIF 2026 Visual Revision Note

## Current PDF Audit

The reviewed `ICAIF26.pdf` is seven pages. It already contains the fresh-panel
main table, learner/ecology transfer table, and a two-panel dose/routing
sensitivity figure. The main unresolved presentation defect is structural:
both the Introduction and Section 3 refer to `Figure ??`, but no framework
figure is present.

The existing result page is numerically dense, but its tables use flat headers.
The result narrative is stronger than the visual hierarchy because the reader
must infer which columns establish execution, selectivity, economic cost, and
robust inference.

## Main-Text Replacement Set

Use exactly four core visual items:

1. `figure_framework.tex` as Figure 1, immediately after the opening paragraph
   of Section 3. It resolves both missing references and makes the information
   order explicit.
2. `table_main_results.tex` in Section 5.1. Replace the existing Table 1; do
   not include both versions.
3. `table_transfer_results.tex` in Section 5.2. Replace the existing Table 2.
4. `figure_sensitivity.tex` in Section 5.3. Replace the existing bitmap/PDF
   sensitivity wrapper with the vector PGFPlots version.

The grouped headers align each visual block with a claim:

- execution outcome: does state-aware timing improve low-risk execution?
- selectivity: does it avoid high-risk intervention?
- economic cost: is spread compression obtained through blanket pressure?
- paired evidence: is the result stable across independently trained policies?
- robust inference: does the transfer decision survive path resampling?

The current manuscript has one spare page. Adding the framework figure while
replacing, rather than duplicating, the existing result assets should keep the
paper near the eight-page ICAIF limit. Final page count still requires the
complete paper source; only the compiled PDF and section fragments were
available locally during this revision.

## Supplement Set

- `figure_paired_seeds.tex` exposes the two MAPPO and three IPPO reversals. It
  directly supports the `Supported` versus `Inconclusive` labels without
  implying seed-wise dominance.
- `table_behavioral_boundary.tex` compresses the forced-undercut evidence into
  one column. It can replace prose in the main text if space permits, otherwise
  it belongs in the supplement.

Do not add generic learning-rate, entropy, network-width, or loss-curve plots.
The dose and routing panels are the claim-relevant sensitivity analysis.
Brero/Banerjee details remain descriptive and should stay in the supplement
because their frozen admissibility gates did not pass.

## Compilation

The complete asset preview is:

`output/pdf/icaif26_assets/preview.pdf`

The snippets and required preamble are under:

`revision/paper/latex/icaif26_assets/`

The preview compiled with bundled Tectonic under an ACM two-column class and
was rendered page by page. No clipped labels, overlapping nodes, or table-width
violations remained after the second framework layout pass.
