# FAIMI Submission Requirements

Requirements for submitting to the **FAIMI workshop (Fairness of AI in Medical
Imaging)**, held in conjunction with MICCAI. These are the hard constraints our
`submission/` paper must satisfy.

> **Status (June 2026):** FAIMI 2026 has **not yet posted** its call for papers.
> The constraints below are taken from **FAIMI 2025** (MICCAI 2025, Daejeon) and
> cross-checked against **FAIMI 2024** — they have been identical for two years
> running, so they are a reliable planning baseline. **Re-verify against the
> FAIMI 2026 CfP once it is published** (watch <https://faimi-workshop.github.io/>).

## Hard constraints (the ones that bind us)

| Requirement | Value |
|---|---|
| **Page limit (content)** | **8 pages maximum** |
| **References** | **up to 2 additional pages** (references only — not counted in the 8) |
| **Template** | **Springer LNCS** (`llncs.cls`) — already what `main.tex` uses |
| **Anonymization** | **Required.** Double-blind. Strip author names, affiliations, emails, acknowledgements, and any self-identifying phrasing/links before submission |
| **Review** | Double-blind, MICCAI standards |
| **Platform** | Microsoft **CMT** |
| **Track** | Single track. No archival/non-archival or short-paper distinction |
| **Proceedings** | Published in the MICCAI workshop volumes of the Springer **LNCS** series |

## Supplementary material

- **Multimedia only** (videos: `avi`, `mp4`, `wmv`).
- **No PDFs** of extra results/proofs/analysis. (Exception: a PDF of cited
  unpublished work, itself anonymized.)
- Must **not** display proofs, analysis, additional results, or any
  identification markers.
- Single zipped file with a README describing contents.
- Reviewers are under no obligation to look at it — **the paper must be fully
  self-contained without it.**

## What this means for our paper

- We are **currently over budget**: the LNCS-formatted draft in `paper/` spans
  six fairly dense sections (dataset, methodology, experiments, discussion + rich
  tables/figures). Methodology alone is large. **Trimming to 8 content pages will
  be the main editorial constraint** — expect to cut/condense, likely starting
  with the methodology and discussion subsections.
- **Anonymization** means a submission-time pass to remove the author block in
  `main.tex` (names, `s225224`/`adipa`/`afhar` emails, DTU institute line) and
  any first-person references to our own prior work that would unblind us.
- The **HF model-release** TODO (`methodology.tex:270`) must not introduce a
  deanonymizing URL in the double-blind version.

## Important dates

FAIMI dates track the MICCAI workshop timeline and shift yearly. For reference,
**FAIMI 2025** ran:

| Milestone | FAIMI 2025 date |
|---|---|
| Full paper submission | June 30, 2025 |
| Acceptance notification | July 16, 2025 |
| Camera-ready | July 30, 2025 |
| Workshop | September 23, 2025 |

**FAIMI 2026 dates are TBA** — MICCAI 2026 itself is the anchor; check the CfP
when released.

## Related venue option

FAIMI also runs a **MELBA journal special issue** (Machine Learning for
Biomedical Imaging) in some years — a longer-form, archival alternative to the
8-page workshop paper. Worth keeping in mind if the 8-page limit proves too
tight for the full story, though the workshop is the primary target per the
supervisor discussion.

## Sources

- [FAIMI 2025 workshop (MICCAI 2025)](https://faimi-workshop.github.io/2025-miccai-workshop/)
- [FAIMI 2024 workshop (MICCAI 2024)](https://faimi-workshop.github.io/2024-miccai/)
- [FAIMI workshop home](https://faimi-workshop.github.io/)
- [MICCAI 2026 paper submission guidelines](https://conferences.miccai.org/2026/en/PAPER-SUBMISSION-GUIDELINES.html)
- [MELBA FAIMI special issue](https://www.melba-journal.org/blog/016-special-issue-faimi.html)
