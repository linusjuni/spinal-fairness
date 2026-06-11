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
