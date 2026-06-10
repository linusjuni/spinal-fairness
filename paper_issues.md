# Paper Audit — Issues Found
**Date:** 2026-06-09  
**Scope:** Full paper — all 7 sections verified  
**Method:** Every section read; non-comment rendered content extracted with `grep -v '^%'`; every quantitative claim cross-checked against `outputs/eda/`, `outputs/fairness/`, `docs/`, and `docs/papers/spine.pdf`.

---

## Section-by-section rendering audit

The following shows exactly what each section currently renders in the compiled PDF (non-comment, non-blank lines only):

| Section file | Rendered output |
|---|---|
| `abstract.tex` | `\keywords{Fairness \and Medical segmentation \and Spine MRI \and Bias.}` + placeholder "Lorem Ipsum Dolor." |
| `introduction.tex` | `\section{Introduction}` — heading only |
| `dataset.tex` | Full prose — **the only section with verifiable claims** |
| `methodology.tex` | `\section{Methodology}` — heading only |
| `experiments.tex` | `\section{Experiments and Results}` — heading only |
| `discussion.tex` | `\section{Discussion}` — heading only |
| `conclusion.tex` | `\section{Conclusion}` — heading only |

Verified with: `grep -v '^%' paper/sections/<name>.tex | grep -v '^$'`

All content in the five non-dataset sections (excluding headings) is inside LaTeX comment lines (`% ...`) — planning notes that produce no PDF output. There are therefore **zero verifiable claims** in those sections.

The keywords (`Fairness`, `Medical segmentation`, `Spine MRI`, `Bias`) are factually appropriate for this paper.

The abstract placeholder ("Lorem Ipsum Dolor.") will need to be written; no factual audit possible yet.

**Additional rendered items in `main.tex`:**  
- Title: "Fairness in Cervical Spine MRI Segmentation" — accurate  
- Authors: Linus Juni, Aditya Parikh, Aasa Feragen — matches project record  
- Institute: Section for Visual Computing, DTU Compute, Technical University of Denmark, Kongens Lyngby — accurate  
- Email pattern: `{s225224, adipa, afhar}@dtu.dk` — matches known student ID; DTU email addresses not independently verifiable from data

---

## ERRORS

### E1 — Sex vertebral-body percentage gap is stated as +24%, should be +23%

**Location:** `paper/sections/dataset.tex`  
- Table 2, row "Sex (Male)", column "Gap" (vertebral body): `+24%`  
- Figure caption for `fig:volumes`: `"men are ~24% larger in both structures (r_rb = 0.56 vertebral body, 0.58 disc)"`

**What the actual data says:**  
From `outputs/eda/full/segmentation_volumes/20260421_112655/stats.json`:
- Female VB median: **47,332.28 mm³**
- Male VB median: **58,106.42 mm³**
- Gap: (58,106 − 47,332) / 47,332 = **22.76%** → rounds to **+23%**, not +24%

The disc gap is correctly stated as +24%:
- Female disc median: 10,806.20 mm³, Male: 13,403.08 mm³ → (13,403 − 10,806) / 10,806 = **24.03%** ✓

**Fix:**
- Table 2: change VB gap for Sex from `+24\%` to `+23\%`
- Figure caption: change `"~24\% larger in both structures"` to something like `"~23\% larger in vertebral bodies and ~24\% larger in discs"`

---

### E2 — Race vertebral-body percentage gap is internally inconsistent: +11% vs +12% depending on denominator

**Location:** `paper/sections/dataset.tex`  
- Table 2, row "Race (White)", column "Gap" (vertebral body): `+11%`  
- Table caption says "Gap is the difference in median volume between the larger group … and the smaller one"

**What the actual data says:**  
From `outputs/eda/full/segmentation_volumes/20260421_112655/stats.json`:
- White VB median: **53,633.07 mm³**
- Black VB median: **47,860.33 mm³**
- (White − Black) / **Black** = 12.06% → **+12%** (consistent with how sex and age gaps are computed: using the smaller as denominator)
- (White − Black) / **White** = 10.77% → **+11%** (this is what the paper seems to use for race, but not for the other rows)

The sex and age gaps use the smaller group as denominator:
- Sex disc: (13,403 − 10,806) / 10,806 = 24.0% ✓ (smaller = Female)
- Age VB: (56,253 − 46,899) / 46,899 = 19.9% ✓ (smaller = <40)

Using the same formula (smaller-group denominator) for race VB gives **+12%**, not +11%.

**Fix:** Recompute all gaps consistently as (larger − smaller) / smaller × 100. This gives:
| Row | Structure | Correct gap |
|-----|-----------|------------|
| Sex (Male) | VB | +23% |
| Sex (Male) | Disc | +24% |
| Race (White) | VB | +12% |
| Race (White) | Disc | ~2% (still negligible, keep "~0%") |
| Age (≥60 vs <40) | VB | +20% ✓ |
| Age (≥60 vs <40) | Disc | +7% ✓ |

---

## VERIFIED CORRECT

All of the following numbers were checked against `outputs/` data files and/or `docs/papers/spine.pdf`:

### Cohort sizes
| Paper claim | Verified value | Status |
|---|---|---|
| $N_0 = 1{,}254$ exams | `n0_exams: 1254` | ✓ |
| $N = 1{,}142$ analysis cohort | `1142 = 448+694` (splits.md) | ✓ |
| "1,255 … from 1,232 patients" (Zhou et al.) | spine.pdf p.1 | ✓ |
| "1,231 patients" (working set) | `n_patients: 1231` | ✓ |
| "a handful of patients contributed more than one exam" | 1254−1231 = 23 multi-exam patients | ✓ |

### Demographic counts and percentages
| Paper claim | Actual | Status |
|---|---|---|
| Female 683 (54.5%) | 683 / 54.5% | ✓ |
| Male 571 (45.5%) | 571 / 45.5% | ✓ |
| White 809 (64.5%) | 809 / 64.5% | ✓ |
| Black 349 (27.8%) | 349 / 27.8% | ✓ |
| Other/unknown 96 (7.7%) | 33+28+25+9+1 = 96 | ✓ |
| Footnote: Asian 25, AIAN 9, NHPI 1, Other 28, NR 33 | exact match | ✓ |
| Non-Hispanic 1,162 (92.7%) | exact | ✓ |
| Hispanic 59 (4.7%) | exact | ✓ |
| Not reported 33 (2.6%) | exact | ✓ |
| Age mean±SD: $54.6 \pm 16.3$ | 54.596 ± 16.309 | ✓ |
| Age median 56 | 56.0 | ✓ |
| Age range 18–89 | min=18, max=89 (13 missing >89) | ✓ |
| 13 exams with missing age (confirmed >89) | n_missing=13 | ✓ |
| Age bins: <40 241 (19.2%), 40–60 486 (38.8%), ≥60 527 (42.0%) | exact | ✓ |
| Siemens 788 (62.8%) | exact | ✓ |
| GE 466 (37.2%) | exact | ✓ |
| 1.5 T 746 (59.5%), 3.0 T 508 (40.5%) | exact | ✓ |

### MRI volume statistics
| Paper claim | Actual | Status |
|---|---|---|
| In-plane resolution ~0.53 mm | spacing_x mean 0.5296 mm | ✓ |
| Slice thickness ~4 mm | spacing_z mean 3.9554 mm | ✓ |
| Slices 12–25 per scan, mean 15.2 ± 1.3 | min=12, max=25, mean=15.21, std=1.32 | ✓ |

### Effect sizes (Table 2 and figure caption)
| Paper claim | Actual | Status |
|---|---|---|
| Sex VB: $r_{rb} = 0.56$, large | 0.5588 | ✓ |
| Sex Disc: $r_{rb} = 0.58$, large | 0.5797 | ✓ |
| Race VB: $r_{rb} = 0.29$, small | 0.2924 | ✓ |
| Race Disc: negligible, not significant | r_rb=0.025, p_adj=0.50 | ✓ |
| Age VB: $\varepsilon^2 = 0.12$, medium | 0.1192 | ✓ |
| Age Disc: $\varepsilon^2 = 0.01$, negligible | 0.0098 | ✓ |

### Scanner sanity check numbers
| Paper claim | Actual | Status |
|---|---|---|
| GE VB voxel median 69,689 | 69,689.0 | ✓ (exact) |
| Siemens VB voxel median 32,401 | 32,401.0 | ✓ (exact) |
| Voxel ratio ~2.15× | 69689/32401 = 2.151 | ✓ |
| VB mm³ medians 49,850 vs 52,956 | 49,850.41 vs 52,955.94 | ✓ |
| VB mm³ gap ~6% | 6.23% | ✓ |
| Disc mm³ difference not significant | p_adj=0.369 | ✓ |

### Component counts
| Paper claim | Actual | Status |
|---|---|---|
| VB components: median 9 in <40, 7 in 60+ | 9.0, 7.0 | ✓ |
| Disc piece-count stays flat across age | <40=9.0, 40–60=9.0, 60+=9.0 | ✓ |
| VB component age effect: medium | ε²=0.119 | ✓ |

### Confounder checks
| Paper claim | Actual | Status |
|---|---|---|
| White patients older than Black: median 58 vs 54 | White=58.0, Black=54.0 (n_a=800, n_b=346) | ✓ |
| "all Cramér's V ≈ 0.05–0.07, none significant" (sex×race, race×mfr, race×FS) | 0.0564, 0.0658, 0.0708; all p_adj>0.39 | ✓ (approximate range OK; 0.071 rounds to 0.07) |
| Race independent of manufacturer and field strength | V=0.066 (p=0.49), V=0.071 (p=0.39) | ✓ |

### Source paper (spine.pdf) cross-checks
| Paper claim | spine.pdf | Status |
|---|---|---|
| "1,255 sagittal T2-weighted exams from 1,232 patients at Duke University" | p.1 abstract | ✓ |
| "About 40% of the exams come with segmentations drawn by expert radiologists" | 491/1,255 = 39.1% (spine.pdf p.2) | ✓ |
| Silver labels "never checked by a human" (as segmentations) | "not undergone manual review … weakly labeled" (spine.pdf p.5) | ✓ |
| Gold standard DSC (memory only, not yet in paper): VB 0.929, disc 0.904, macro 0.916 | spine.pdf Table 2, Ensemble column | ✓ (will need citing when Methodology is written) |

---

## NOTES (not errors, but worth reviewing)

**N1 — Table 2 caption does not specify percentage denominator**  
The caption says "Gap is the difference in median volume between the larger group (named in parentheses) and the smaller one." It does not say this is a percentage, nor which denominator is used. With E2 fixed (using smaller as denominator throughout), the caption should clarify: "expressed as (larger − smaller) / smaller × 100%".

**N2 — "disc volume barely moves" vs "+7%" in table**  
The text says "disc volume barely moves" (age) and the figure caption says "essentially flat (ε² = 0.01, negligible)". The table shows "+7%". These are not contradictory (negligible effect size = practically flat), but a reader could be confused by "+7%" alongside "barely moves." Consider noting the ε² = 0.01 is what makes it "negligible" despite a modest-looking percentage.

**N3 — "never checked by a human" nuance**  
The source paper states: "The mid-sagittal slice of all unannotated exams was reviewed for quality and consistency by one of the six experts." This is an image-quality check, not a segmentation review. The silver segmentation labels were not reviewed. The paper's wording is accurate but could be tightened: "the segmentation labels were generated automatically and never reviewed by a human."

**N4 — Post-doc annotator omission**  
The source paper's annotation team includes "one post-doctoral researcher without medical training" who drafted annotations (subsequently reviewed by radiologists). Our current text implies annotations are purely from radiologists. When the introduction / dataset section elaborates on the gold standard, this detail from the source paper should be included for completeness.

**N5 — Abstract is placeholder ("Lorem Ipsum Dolor")**  
The abstract section currently contains only a placeholder. It needs to be written.

**N6 — Introduction, Experiments, Discussion, Conclusion are all TODO**  
These sections contain only planning comments. No factual claims to verify yet.

---

## SUMMARY

| Category | Count |
|---|---|
| Factual errors requiring correction | **2** (E1, E2) |
| Notes / clarifications recommended | 6 (N1–N6) |
| Claims verified correct | ~40 |

The two errors are both in Table 2 (`tab:volumes`) in `paper/sections/dataset.tex`:
1. Sex VB gap: `+24\%` → `+23\%`  
2. Race VB gap: `+11\%` → `+12\%` (using consistent (larger−smaller)/smaller formula)
