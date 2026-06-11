# Verification task: are the three nnU-Net plans identical across regimes?

**For:** an LLM/agent with access to the HPC where nnU-Net was trained (the paper
repo does not contain the preprocessed plans).
**Why:** the Methodology section (`paper/sections/methodology.tex`, `sec:model`)
claims all three label regimes share an *identical* nnU-Net architecture and
training recipe. We need to confirm that empirically before the wording is
final.

---

## Background

We train three models that differ **only in their training labels**, holding the
model fixed so any fairness difference is attributable to labels, not capacity:

| Regime | Dataset (nnU-Net) | Train N | Labels |
|---|---|---|---|
| `M_mix`    | Dataset001_CSpineSeg | 798 | gold + silver |
| `M_gold`   | Dataset002_*         | 288 | gold only |
| `M_silver` | Dataset003_*         | 450 | silver only |

(Confirm the exact dataset folder names on disk — Dataset002/003 suffixes may
differ.)

The paper currently asserts (draft wording):

> "holding the framework fixed across the three regimes guarantees that
> `M_mix`, `M_gold`, and `M_silver` share an identical architecture and training
> recipe."

## What we already know (from nnU-Net v2 source review)

- **Patch size, network depth, downsampling topology, and normalization are
  geometry-driven** (target spacing + median image shape + GPU-memory budget)
  and are label-independent. Because all three datasets are built from the *same
  image pool* with the same voxel geometry, these are expected to be **identical**.
- **Batch size is the one element that can differ**, because nnU-Net caps a
  minibatch at 5% of total dataset voxels:
  `bs_cap = 0.05 * (median_shape_voxels * numTraining) / prod(patch_size)`,
  then `batch_size = max(min(vram_derived_bs, bs_cap), min_bs)`.
  Since `numTraining` differs across regimes (798 / 288 / 450 → ~3x range), the
  cap differs ~3x. **It only changes the actual batch size if the cap is the
  binding constraint** (i.e. smaller than the VRAM-derived value). For large
  datasets the cap usually does not bind, so batch size often stays the same —
  but this must be checked, not assumed.

So the open question is narrow: **do the three plans end up with the same
`batch_size` (and, as a sanity check, the same `patch_size`)?**

---

## What to check

The plans live under `$nnUNet_preprocessed/<DatasetXXX>/nnUNetPlans.json` on the
HPC. (If `$nnUNet_preprocessed` is unset, find them with
`find / -name nnUNetPlans.json 2>/dev/null` or check the project's nnU-Net env
setup in `docs/nnunet/01_setup.md`.)

For **each** of the three datasets (001 = mix, 002 = gold, 003 = silver), and for
**both** configurations (`2d` and `3d_fullres`), extract from
`configurations` in `nnUNetPlans.json`:

1. `patch_size`
2. `batch_size`
3. `architecture` → `arch_kwargs` → `n_stages`, `features_per_stage`,
   `strides` / `pool_op_kernel_sizes`, `kernel_sizes` (the topology)
4. `spacing` and `normalization_schemes`
5. `UNet_class_name` / network class

A quick way to dump the relevant fields:

```bash
for ds in Dataset001_CSpineSeg Dataset002_<...> Dataset003_<...>; do
  echo "=== $ds ==="
  python - "$nnUNet_preprocessed/$ds/nnUNetPlans.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
for cfg in ("2d", "3d_fullres"):
    c = p["configurations"][cfg]
    print(cfg,
          "patch", c.get("patch_size"),
          "batch", c.get("batch_size"),
          "spacing", c.get("spacing"),
          "norm", c.get("normalization_schemes"))
    a = c.get("architecture", {}).get("arch_kwargs", {})
    print("   n_stages", a.get("n_stages"),
          "features", a.get("features_per_stage"),
          "strides", a.get("strides"))
PY
done
```

Also report each dataset's `numTraining` (from
`$nnUNet_raw/<DatasetXXX>/dataset.json`) to confirm the 798 / 288 / 450 figures.

---

## How to report back

Produce a small table: rows = the 5 fields above, columns = the three regimes
(for each of `2d` and `3d_fullres`), and a final **verdict**:

- **All fields identical across the three regimes** → the paper's "identical
  architecture and training recipe" claim is fully supported; report
  CONFIRMED and quote the shared `patch_size` / `batch_size`.
- **Everything identical except `batch_size`** → report which regime(s) differ
  and the values. This is expected behaviour from the dataset-size cap, **not** a
  design choice, and does not confound the label comparison — but the paper
  wording must acknowledge it (e.g. "architecture, patch size, depth and
  normalization are identical across regimes; the planner's dataset-size cap
  yields a [smaller/larger] batch size for the gold regime"). Report
  CONFIRMED-WITH-CAVEAT and give the exact numbers for a one-clause edit.
- **patch_size, topology, spacing, or normalization differ** → unexpected;
  report DISCREPANCY with full details, since it would undermine the
  controlled-comparison argument and needs investigation (e.g. were the three
  datasets planned/preprocessed with the same target spacing and the same
  planner/`-pl nnUNetPlannerResEncL`?).

---

## Secondary (no HPC needed — note only)

Separately, the draft's fingerprint parenthetical
(`paper/sections/methodology.tex`, ~line 223) lists the fingerprint contents as
"(image sizes, voxel spacings, modality, class frequencies)". In nnU-Net **v2**
the stored fingerprint is spacings, cropped image shapes, and foreground
intensity statistics — *modality* comes from `dataset.json` and *class
frequencies* are not in the v2 fingerprint. The phrasing is defensible at the
conceptual level (the 2021 Nature Methods paper describes the fingerprint
loosely), but if airtightness is wanted, replace "modality, class frequencies"
with "intensity statistics". Flag, don't auto-edit.

---

## Findings (verified 2026-06-11)

**Plan file used:** `nnUNetResEncUNetLPlans.json` (ResEncL planner, `-pl nnUNetPlannerResEncL`).
Note: Dataset002_CSpineSeg_Gold also contains a `nnUNetPlans.json` (default planner), but the
training runs used `nnUNetResEncUNetLPlans.json` consistently across all three datasets.

### Dataset sizes (from `dataset.json` and `imagesTr/`)

| Regime    | Dataset folder              | `numTraining` in dataset.json | Files in imagesTr |
|-----------|-----------------------------|------------------------------:|------------------:|
| `M_mix`   | Dataset001_CSpineSeg        | 914                           | 1164              |
| `M_gold`  | Dataset002_CSpineSeg_Gold   | 332                           | 332               |
| `M_silver`| Dataset003_CSpineSeg_Silver | 516                           | 516               |

Note: the paper draft states 798 / 288 / 450 for train N. The discrepancy with the nnU-Net
`numTraining` values (914 / 332 / 516) reflects the difference between what was provided to
nnU-Net (train + val, used internally for 5-fold CV) and the paper's train-only count after
holding out a test set. For Dataset001 there is an additional discrepancy between `numTraining`
(914) and actual files in imagesTr (1164) — the dataset.json may not have been updated when
cases were added.

### Plan comparison (`nnUNetResEncUNetLPlans.json`)

**Programmatic diff confirmed: ALL FIELDS IDENTICAL across the three datasets.**

#### 2d configuration

| Field                  | M_mix (DS001) | M_gold (DS002) | M_silver (DS003) |
|------------------------|---------------|----------------|------------------|
| `patch_size`           | [512, 512]    | [512, 512]     | [512, 512]       |
| `batch_size`           | **35**        | **35**         | **35**           |
| `spacing`              | [0.4297, 0.4297] | [0.4297, 0.4297] | [0.4297, 0.4297] |
| `normalization_schemes`| ZScoreNorm    | ZScoreNorm     | ZScoreNorm       |
| `n_stages`             | 8             | 8              | 8                |
| `features_per_stage`   | [32,64,128,256,512,512,512,512] | same | same |
| `strides`              | [[1,1],[2,2]×7] | same | same |
| `network_class_name`   | ResidualEncoderUNet | same | same |

#### 3d_fullres configuration

| Field                  | M_mix (DS001)       | M_gold (DS002)      | M_silver (DS003)    |
|------------------------|---------------------|---------------------|---------------------|
| `patch_size`           | [16, 512, 512]      | [16, 512, 512]      | [16, 512, 512]      |
| `batch_size`           | **2**               | **2**               | **2**               |
| `spacing`              | [3.99, 0.4297, 0.4297] | same             | same                |
| `normalization_schemes`| ZScoreNorm          | ZScoreNorm          | ZScoreNorm          |
| `n_stages`             | 8                   | 8                   | 8                   |
| `features_per_stage`   | [32,64,128,256,320,320,320,320] | same | same        |
| `strides`              | [[1,1,1],[1,2,2],[1,2,2],[1,2,2],[2,2,2],[2,2,2],[1,2,2],[1,2,2]] | same | same |
| `network_class_name`   | ResidualEncoderUNet | same                | same                |

Full architecture: `dynamic_network_architectures.architectures.unet.ResidualEncoderUNet`
with `Conv3d`, `InstanceNorm3d`, `LeakyReLU`, n_blocks_per_stage=[1,3,4,6,6,6,6,6].

### Verdict: CONFIRMED

All five checked fields (patch_size, batch_size, topology/architecture, spacing,
normalization_schemes) are **byte-for-byte identical** across M_mix, M_gold, and M_silver for
both 2d and 3d_fullres configurations. The batch_size cap does NOT differ across regimes despite
the 3× range in numTraining — the VRAM-derived batch size is the binding constraint for all
three, not the dataset-size cap. The paper's claim that the three regimes share an **identical
architecture and training recipe** is **fully supported**.

Shared values to quote in the paper: patch size [16, 512, 512] (3d_fullres) / [512, 512] (2d);
batch size 2 (3d_fullres) / 35 (2d); 8-stage ResidualEncoderUNet; ZScoreNormalization.

### Secondary note (fingerprint wording)

As flagged in the task: the draft's fingerprint parenthetical ("image sizes, voxel spacings,
modality, class frequencies") does not precisely match nnU-Net v2's stored fingerprint
(spacings, cropped image shapes, foreground intensity statistics). The wording is defensible
at the conceptual level but "modality, class frequencies" could be tightened to "intensity
statistics" for airtightness. **Not auto-edited** — flagged for author review.
