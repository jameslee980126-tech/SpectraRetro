#!/usr/bin/env bash
set -euo pipefail

GDIFF_ROOT="${GDIFF_ROOT:-/root/rivermind-data/GDiffRetro-main}"
WORK="${WORK:-/root/baseline-work/RetroDiT-ch6_69pct}"
RUN="${RUN:-/root/rivermind-data/baselines/runs/retrodit_ch6_69pct_unknown_native/formal_8m_earlystop_100k_v2}"
FORMAL="${FORMAL:-${RUN}/inference_product_only/formal_top10_50step}"
POLL_SECONDS="${POLL_SECONDS:-120}"
SELECTED_CKPT="${RUN}/retrodit_ch6_unknown_formal_8m_earlystop_100k_v2/runs/2026-08-14_22-13-54/checkpoints/best_top5acc_model_step30000_acc0.9028.ckpt"

exec 9>"${RUN}/finalization.lock"
flock -n 9 || { echo "RetroDiT finalization queue is already active"; exit 0; }
exec > >(tee -a "${RUN}/finalization.log") 2>&1
printf '%s\n' "$$" > "${RUN}/finalization.pid"
trap 'rm -f "${RUN}/finalization.pid"' EXIT

echo "[$(date '+%F %T')] waiting for formal inference"
while true; do
  if [[ -s "${FORMAL}/status.json" ]]; then
    STATE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "${FORMAL}/status.json")"
    case "${STATE}" in
      completed) break ;;
      failed)
        echo "formal inference failed; finalization is not promoted"
        exit 2
        ;;
    esac
  fi
  sleep "${POLL_SECONDS}"
done

OUTPUT_DIR="$(find "${FORMAL}" -type f -name generated_metrics.json -printf '%h\n' | sort | tail -n 1)"
if [[ -z "${OUTPUT_DIR}" || ! -s "${OUTPUT_DIR}/generated_samples.json" ]]; then
  echo "completed status exists but generated outputs are missing"
  exit 3
fi

SOURCE_ARCHIVE="${RUN}/retrodit_official_source_used.tar.gz"
REPRO_ARCHIVE="${RUN}/retrodit_study_adapters_and_protocol.tar.gz"
tar --exclude='__pycache__' --exclude='*.pyc' -czf "${SOURCE_ARCHIVE}" \
  -C "${WORK}" \
  README.md LICENSE requirements.txt pyproject.toml configs docs scripts src \
  data/graph_vocab.json OFFICIAL_SOURCE_COMMIT OFFICIAL_SOURCE_ARCHIVE_SHA256
tar --exclude='__pycache__' --exclude='*.pyc' -czf "${REPRO_ARCHIVE}" \
  -C "${GDIFF_ROOT}" \
  baseline_experiments/PROTOCOL.md \
  baseline_experiments/MODEL_AUDIT.md \
  baseline_experiments/runs/retrodit_ch6_69pct.json \
  baseline_experiments/adapters/prepare_retrodit_ch6.py \
  baseline_experiments/adapters/audit_retrodit_processed.py \
  baseline_experiments/adapters/run_retrodit_with_compat.py \
  baseline_experiments/adapters/run_retrodit_eval_preserve_predictions.py \
  baseline_experiments/server/setup_retrodit_dependencies.sh \
  baseline_experiments/server/run_retrodit_ch6_unknown.sh \
  baseline_experiments/server/run_retrodit_ch6_unknown_inference.sh \
  baseline_experiments/server/finalize_retrodit_ch6_unknown.sh

python3 - "${RUN}" "${FORMAL}" "${OUTPUT_DIR}" "${SELECTED_CKPT}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

run, formal, output, checkpoint = map(Path, sys.argv[1:])
training = json.loads((run / "status.json").read_text())
inference = json.loads((formal / "status.json").read_text())
metrics = json.loads((output / "generated_metrics.json").read_text())
samples = json.loads((output / "generated_samples.json").read_text())
report = f"""# RetroDiT CH6 Unknown-class experiment report

- Finalized: {datetime.now().astimezone().isoformat(timespec='seconds')}
- Training state: {training.get('state')}
- Training start/end: {training.get('start_time')} / {training.get('end_time')}
- Early stop: global step 70,000 (logged epoch 22)
- Selected checkpoint: `{checkpoint}`
- Selection rule: highest sampled validation deduplicated Top-5
- Information regime: reaction class unknown
- Test ordering: product-only canonical identity ordering; no ground-truth reaction-center information
- Test diffusion steps: {inference.get('sampling_steps')}
- Reported Top-k: 1/3/5/10
- Natively processed test rows: {len(samples)} of 9,250 frozen rows
- Unsupported frozen rows: {9250 - len(samples)} (no credit in unified evaluation)

## Released native exact-match metrics

```json
{json.dumps(metrics, indent=2)}
```

## Leakage audit

The reaction-center prediction JSON shipped with the author repository belongs to the
original USPTO-50K split and has zero product overlap with this frozen CH6 test split.
Target-derived CH6 permutation files reveal the reaction center and were not used for
formal inference. Ten stochastic candidates were generated per product using only the
canonical product graph ordering.

## Artifact scope

The synchronized archive includes the released source snapshot used at runtime, study
adapters and launch scripts, every saved training checkpoint, training and inference
logs/configuration, raw ranked candidates, native metrics, and checksum manifests.
"""
(run / "ANALYSIS_REPORT.md").write_text(report, encoding="utf-8")
PY

MANIFEST_TMP="${RUN}/final_artifacts.sha256.tmp"
{
  find "${RUN}/retrodit_ch6_unknown_formal_8m_earlystop_100k_v2/runs/2026-08-14_22-13-54/checkpoints" \
    -maxdepth 1 -type f -name '*.ckpt' -print0
  printf '%s\0' \
    "${RUN}/status.json" \
    "${RUN}/train.log" \
    "${RUN}/launcher.log" \
    "${RUN}/compatibility.jsonl" \
    "${RUN}/product_only_canonical_permutations_test.json" \
    "${SOURCE_ARCHIVE}" \
    "${REPRO_ARCHIVE}" \
    "${RUN}/ANALYSIS_REPORT.md" \
    "${RUN}/retrodit_ch6_unknown_formal_8m_earlystop_100k_v2/runs/2026-08-14_22-13-54/.hydra/config.yaml" \
    "${RUN}/retrodit_ch6_unknown_formal_8m_earlystop_100k_v2/runs/2026-08-14_22-13-54/.hydra/hydra.yaml" \
    "${RUN}/retrodit_ch6_unknown_formal_8m_earlystop_100k_v2/runs/2026-08-14_22-13-54/.hydra/overrides.yaml" \
    "${FORMAL}/status.json" \
    "${FORMAL}/inference.log" \
    "${FORMAL}/launcher.log" \
    "${OUTPUT_DIR}/generated_samples.json" \
    "${OUTPUT_DIR}/generated_metrics.json" \
    "${OUTPUT_DIR}/config_tree.log" \
    "${OUTPUT_DIR}/.hydra/config.yaml" \
    "${OUTPUT_DIR}/.hydra/hydra.yaml" \
    "${OUTPUT_DIR}/.hydra/overrides.yaml"
} | sort -z | xargs -0 sha256sum -- > "${MANIFEST_TMP}"
mv "${MANIFEST_TMP}" "${RUN}/final_artifacts.sha256"
touch "${RUN}/finalization_complete.marker"
echo "[$(date '+%F %T')] RetroDiT finalization complete: $(wc -l < "${RUN}/final_artifacts.sha256") artifacts"
