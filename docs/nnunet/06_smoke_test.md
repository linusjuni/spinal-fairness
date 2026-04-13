# 06 — Smoke Test Log

A record of the first end-to-end pipeline run on DTU HPC (April 2026), including every
error encountered, its root cause, and the fix applied. Read this before running the
pipeline for the first time on a new machine.

---

## What Was Run

The full sequence from raw data to the start of `plan_and_preprocess`:

```bash
uv run -m src.data.splits.v3           # generate split_v3.tsv
uv run -m src.nnunet.prepare_dataset   # build Dataset001_CSpineSeg/
uv run nnUNetv2_plan_and_preprocess -d 1
```

`plan_and_preprocess` was killed early (smoke test only). `write_splits` and training
were not reached.

---

## Errors Encountered and Fixes

### 1. `nnUNetv2_plan_and_preprocess: command not found`

**Cause:** nnU-Net CLI entry points are installed inside the uv virtual environment and
are not on the system `PATH`.

**Fix:** Prefix all `nnUNetv2_*` commands with `uv run`:

```bash
uv run nnUNetv2_plan_and_preprocess -d 1
```

This applies to every nnU-Net command: `nnUNetv2_train`, `nnUNetv2_predict`,
`nnUNetv2_find_best_configuration`, `nnUNetv2_apply_postprocessing`,
`nnUNetv2_evaluate_folder`, etc.

---

### 2. `nnUNet_raw is not defined` / `RuntimeError: Could not find a dataset with the ID 1`

**Cause:** nnU-Net reads `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`
directly from shell environment variables. These were not exported in the session.
Our `settings.py` has them as defaults but they are not automatically available to
subprocesses spawned by `uv run`.

**Fix:** Add `UV_ENV_FILE=".env"` to `~/.bashrc` once:

```bash
export UV_ENV_FILE=".env"
```

Then put the three nnU-Net paths in the project's `.env` file (already done):

```
nnUNet_raw=/work3/s225224/nnunet/raw
nnUNet_preprocessed=/work3/s225224/nnunet/preprocessed
nnUNet_results=/work3/s225224/nnunet/results
```

`UV_ENV_FILE` causes every `uv run` invocation to inject the `.env` contents as real
environment variables into the spawned subprocess. No manual `export` needed in any
session or job script.

> **Note:** `[tool.uv] env-file = ".env"` in `pyproject.toml` is not yet supported
> (open issue [astral-sh/uv#15714](https://github.com/astral-sh/uv/issues/15714)).
> `UV_ENV_FILE` is the current workaround.

> **For LSF job scripts:** Add `export UV_ENV_FILE=".env"` after `source ~/.bashrc`.

---

### 3. `RuntimeError: Some images have errors` (direction/origin mismatch)

**Cause:** `--verify_dataset_integrity` raised a hard error on floating-point precision
differences between image and segmentation NIfTI headers. Example:

```
Direction images: (-0.0, ...)
Direction seg:    (1.03e-08, ...)   # difference is at machine epsilon (~1e-8)

Origin images: (46.49616, -111.18540, -72.62515)
Origin seg:    (46.49631, -111.18480, -72.62431)   # difference < 0.001 mm
```

These are not real misalignments. The segmentation masks were generated from these
exact images; the affines are identical in practice. The differences arise from
float32 precision loss when the NIfTI header was saved for the segmentation files.

**Fix:** Drop `--verify_dataset_integrity`:

```bash
# Wrong — crashes on benign float32 noise
uv run nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# Correct
uv run nnUNetv2_plan_and_preprocess -d 1
```

> **Caveat:** Without the flag, genuine misalignments (wrong patient, flipped axis)
> would not be caught at this step. The CSpineSeg data is clean — this is a known
> property of how the Silver Standard masks were generated.

---

### 4. `OSError: [Errno 122] Disk quota exceeded`

**Cause:** The `$nnUNet_preprocessed` directory did not exist. nnU-Net creates
subdirectories inside it but requires the root to already exist. When the root was
missing, nnU-Net's `shutil.copy` failed trying to write `dataset.json` into a
non-existent path.

**Fix:** Create the directories before running `plan_and_preprocess`:

```bash
mkdir -p /work3/s225224/nnunet/preprocessed /work3/s225224/nnunet/results
```

> **Note:** The error message (`Disk quota exceeded`) was misleading. The actual
> cause was a missing directory, not a quota limit. Quota at time of smoke test:
> 49.89 GiB used / 100 GiB hard limit, with ~50 GiB free — sufficient for
> preprocessing.

> **Quota check command on DTU HPC:** `getquota_work3.sh`
> (`df -h` and `quota -s` do not show per-user BeeGFS quota correctly.)

> **BeeGFS quirk:** `du -sh` reports ~27 GB but `getquota_work3.sh` shows ~50 GiB.
> BeeGFS charges per 1 M chunk, so many small files inflate reported usage relative
> to raw file size. Trust `getquota_work3.sh`.

---

### 5. `RuntimeError: Error while creating the NDArray` (blosc2 / quota)

**Cause:** Preprocessing crashed at case 331/1164 with a `blosc2` error:

```
RuntimeError: Error while creating the NDArray
```

This looks like a blosc2 bug (see `01_setup.md` known issues) but was actually
caused by **exceeding the work3 quota**. BeeGFS chunk overhead from ~1.5M small
files inflated usage to 128.5 GiB against the 100 GiB hard limit. `blosc2` does
not surface the underlying `ENOSPC` — it wraps it in a generic `RuntimeError`.

**Diagnosis:**

```bash
getquota_work3.sh   # showed 128.5 GiB / 100 GiB on storagepool 6
du -sh /work3/s225224/*/   # only ~26 GB real data — rest is chunk overhead
```

**Fix:** Quota increase to 300 GiB granted by support@cc.dtu.dk on 2026-04-13.
Delete partial preprocessing output and re-run:

```bash
rm -rf /work3/s225224/nnunet/preprocessed/Dataset001_CSpineSeg/nnUNet*
uv run nnUNetv2_plan_and_preprocess -d 1
```

---

## What `plan_and_preprocess` Produced (before kill)

Fingerprinting and experiment planning completed successfully. The planner output:

| Config | Status | Notes |
|---|---|---|
| `2d` | Planned | 512×512 patch, batch size 12, 8-stage PlainConvUNet |
| `3d_fullres` | Planned | 15×512×512 patch (highly anisotropic) |
| `3d_lowres` | **Dropped** | Image size difference to `3d_fullres` too small to warrant a separate low-res config. This is expected for our small volumes. |
| `3d_cascade_fullres` | Dropped | Requires `3d_lowres` |

2D config detail:
- Normalization: `ZScoreNormalization` (per-case z-score — correct for MRI)
- Architecture: `PlainConvUNet`, Conv2d, InstanceNorm2d
- Patch size: 512×512 (covers the full in-plane FOV)
- Batch size: 12

Preprocessing (the slow step — resampling and normalising 916 cases) was killed before
completion and needs to be re-run.

---

## Planner Warning: Old Default Planner

```
INFO: You are using the old nnU-Net default planner. We have updated our
recommendations. Please consider using those instead!
```

**This is not an error.** The default `ExperimentPlanner` targets 8 GB VRAM and works
correctly. The warning is informational.

**For the smoke test:** ignore it.

**For real training on A100:** switch to a ResEnc planner that utilises A100 VRAM:

| Planner | Target VRAM | Use on |
|---|---|---|
| `ExperimentPlanner` (default) | 8 GB | Smoke test / any GPU |
| `nnUNetPlannerResEncL` | 24 GB | A100 40 GB |
| `nnUNetPlannerResEncXL` | 40 GB | A100 80 GB |

```bash
# Real training — use ResEncL on A100 40 GB
uv run nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncL
nnUNetv2_train 1 2d 0 --npz -p nnUNetResEncUNetLPlans
```

See [03 — Training](03_training.md) for full details.

---

## Remaining Steps to Complete the Pipeline

In order:

1. **Clean partial output and re-run `plan_and_preprocess` with the ResEnc planner**
   (quota increased to 300 GiB on 2026-04-13; use `nnUNetPlannerResEncL` for A100 40 GB):
   ```bash
   rm -rf /work3/s225224/nnunet/preprocessed/Dataset001_CSpineSeg/
   uv run nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncL
   ```
   Expected duration: 15–45 minutes on a login node (CPU-bound, no GPU needed).
   Expected disk use: 20–40 GB additional on work3.

2. **Write custom stratified splits** (must be after step 1, before step 3):
   ```bash
   uv run -m src.nnunet.write_splits
   ```

3. **Submit training jobs** — one LSF job per fold per config. See [03 — Training](03_training.md).
   Pass `-p nnUNetResEncUNetLPlans` to match the planner used above:
   ```bash
   nnUNetv2_train 1 2d 0 --npz -p nnUNetResEncUNetLPlans
   ```
   Submit all 10 jobs (5 folds × 2 configs) to `gpua100`.

4. **Find best configuration** and run inference on test set. See [05 — Inference](05_inference.md).

---

## Caveats for Full Training Run

- **Quota:** 300 GiB granted on 2026-04-13 (was 128.5 / 100 GiB due to BeeGFS
  chunk overhead). Preprocessing adds ~20–40 GB, training results add ~10–30 GB more.
  Current limit is sufficient for the full training run.

- **Planner:** Use `nnUNetPlannerResEncL` (A100 40 GB) or `nnUNetPlannerResEncXL`
  (A100 80 GB) — not the default `ExperimentPlanner`. The default targets 8 GB VRAM and
  wastes A100 capacity. All folds must use the same planner; pass `-p nnUNetResEncUNetLPlans`
  to every `nnUNetv2_train` call.

- **`torch.compile` on first epoch:** GPU utilisation will be 0% for several minutes
  while the network compiles. Do not kill the job. Only start fold 1 after confirming
  fold 0 has non-zero GPU utilisation.

- **`write_splits` must run before any `nnUNetv2_train` call.** If it does not exist,
  nnU-Net auto-generates random (non-stratified) splits and logs a warning. Check that
  `$nnUNet_preprocessed/Dataset001_CSpineSeg/splits_final.json` exists before submitting.

- **`--npz` flag is required** if you intend to ensemble 2d + 3d_fullres afterwards.
  Retroactive fix: `nnUNetv2_train 1 2d 0 --val --npz` (re-runs validation only).

- **Test set is sacred.** Do not run `nnUNetv2_predict` on `imagesTs/` until training
  and model selection are fully complete.
