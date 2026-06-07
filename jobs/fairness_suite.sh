#!/bin/bash
#BSUB -J fairness_suite
#BSUB -q hpc
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -n 24
#BSUB -W 12:00
#BSUB -o jobs/logs/fairness_suite_%J.out
#BSUB -e jobs/logs/fairness_suite_%J.err
#
# Re-runs the fairness analysis suite under the binarized (rate-based) DPD/DIR
# definitions (Parikh et al. / four-fifths rule). Submit with:
#     bsub < jobs/fairness_suite.sh
#
# Per-case metric CSVs are UNCHANGED by the redefinition, so `evaluate` only runs
# when a CSV is missing (idempotent reuse) — the real work is the `analyze`
# passes (bootstrap + permutation). No GPU needed; predictions already exist.
#
# Each stage is GUARDED: it checks its required prediction/reference inputs and
# skips cleanly (without aborting the rest) if they are not present yet. So this
# is safe to submit at any time — it runs whatever is ready and defers the rest.
#
# Analyses produced (reports under outputs/fairness/<report-name>/):
#   1. fairness_global             Dataset001 vs all test labels        (228 cases)
#   2. fairness_biased_ruler       Dataset001 vs gold vs gen. silver    (76 cases)
#   3. fairness_bias_amplification Dataset001/2/3 vs gold labels        (76 cases)
#
# Each analyze pass also writes sensitivity_<ruler>.csv (DIR/DPD threshold sweep).

set -euo pipefail

export UV_ENV_FILE=".env"
mkdir -p jobs/logs outputs
uv sync
source .env

RAW="${nnUNet_raw}"
RES="${nnUNet_results}"
MAP="${RAW}/Dataset001_CSpineSeg/case_id_mapping.json"
METRICS="dice hd95 ndsc"
WORKERS=24

# Common input locations.
D1_PP="${RES}/Dataset001_CSpineSeg/predictions_test_pp"
D2_PP="${RES}/Dataset002_CSpineSeg_Gold/predictions_test_pp"
LABELS_TS="${RAW}/Dataset001_CSpineSeg/labelsTs"
LABELS_GOLD="${RAW}/Dataset001_CSpineSeg/labelsTs_gold"

# Dataset003 predictions on the 76 GOLD test images (NOT its own 138-case silver
# test set). Produced after Dataset003 finishes training; override if named
# differently.
DS3_GOLD_PREDS="${DS3_GOLD_PREDS:-${RES}/Dataset003_CSpineSeg_Silver/predictions_gold_test_pp}"

# Return 0 only if every path exists; otherwise list the missing ones and return 1.
require_paths () {
    local missing=0 p
    for p in "$@"; do
        if [[ ! -e "${p}" ]]; then
            echo "  missing input: ${p}"
            missing=1
        fi
    done
    return ${missing}
}

# Run evaluate only if the per-case CSV is missing (metrics are unchanged).
ensure_eval () {
    local out="$1" preds="$2" refs="$3"
    if [[ -f "${out}" ]]; then
        echo "  reuse existing ${out}"
    else
        echo "  evaluate -> ${out}"
        uv run -m src.fairness.evaluate \
            --predictions "${preds}" \
            --references  "${refs}" \
            --mapping     "${MAP}" \
            --output      "${out}" \
            --metrics ${METRICS} \
            --workers ${WORKERS}
    fi
}

echo "=== 1/3 Global fairness audit (Dataset001, 228 cases) ==="
if require_paths "${D1_PP}" "${LABELS_TS}"; then
    ensure_eval outputs/eval_global.csv "${D1_PP}" "${LABELS_TS}"
    uv run -m src.fairness.analyze \
        --evaluation-csvs outputs/eval_global.csv \
        --ruler-labels    global \
        --mapping         "${MAP}" \
        --report-name     fairness_global
else
    echo "  SKIP stage 1: inputs not ready (see above)."
fi

echo "=== 2/3 Biased ruler (gold vs generated silver, 76 cases) ==="
if require_paths "${D1_PP}" "${LABELS_GOLD}" "${D2_PP}"; then
    ensure_eval outputs/eval_ruler_gold.csv   "${D1_PP}" "${LABELS_GOLD}"
    ensure_eval outputs/eval_ruler_silver.csv "${D1_PP}" "${D2_PP}"
    # Labels MUST be `gold` and `silver` so analyze computes the DIR-widening (the
    # biased-ruler headline). Renaming the silver ruler disables that comparison.
    uv run -m src.fairness.analyze \
        --evaluation-csvs outputs/eval_ruler_gold.csv outputs/eval_ruler_silver.csv \
        --ruler-labels    gold silver \
        --mapping         "${MAP}" \
        --report-name     fairness_biased_ruler
else
    echo "  SKIP stage 2: inputs not ready (see above)."
fi

echo "=== 3/3 Bias amplification (mixed vs gold-trained vs silver-trained, 76 cases) ==="
if require_paths "${D1_PP}" "${D2_PP}" "${DS3_GOLD_PREDS}" "${LABELS_GOLD}"; then
    ensure_eval outputs/eval_ds1_on_gold.csv "${D1_PP}"          "${LABELS_GOLD}"
    ensure_eval outputs/eval_ds2_on_gold.csv "${D2_PP}"          "${LABELS_GOLD}"
    ensure_eval outputs/eval_ds3_on_gold.csv "${DS3_GOLD_PREDS}" "${LABELS_GOLD}"
    uv run -m src.fairness.analyze \
        --evaluation-csvs outputs/eval_ds1_on_gold.csv outputs/eval_ds2_on_gold.csv outputs/eval_ds3_on_gold.csv \
        --ruler-labels    mixed gold_trained silver_trained \
        --mapping         "${MAP}" \
        --report-name     fairness_bias_amplification
else
    echo "  SKIP stage 3: inputs not ready (see above)."
fi

echo "=== Done. Reports: outputs/fairness/{fairness_global,fairness_biased_ruler,fairness_bias_amplification}/ ==="
