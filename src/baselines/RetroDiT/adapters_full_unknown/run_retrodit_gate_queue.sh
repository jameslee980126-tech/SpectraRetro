#!/usr/bin/env bash
set -uo pipefail

project_root="/root/rivermind-data/retrodit_recovery/RetroDiT-latest"
python_bin="/root/rivermind-data/envs/retrodit-gen/bin/python"
checkpoint_root="/root/rivermind-data/retrodit_recovery/payload/checkpoints"
compat_root="/root/rivermind-data/retrodit_recovery/flash_compat"
recovery_root="/root/rivermind-data/retrodit_recovery"
data_root="${project_root}/data/ch6_69pct_gate200_50k"
permutation_json="${project_root}/rc_prediction/outputs/ch6_69pct_rc_seed42_export/predictions/test/retrodit_rc_ch6_seed42_strict_root_center_top7_root_top7_temp1p25_agg_split_edge.json"
queue_log="${recovery_root}/gate200_queue.log"
current_pid="${1:-}"

timestamp() {
    date '+%F %T %Z'
}

record() {
    printf '%s\t%s\n' "$(timestamp)" "$*" >> "${queue_log}"
}

valid_output() {
    local output_dir="$1"
    "${python_bin}" - "${output_dir}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
metrics_path = root / "generated_metrics.json"
samples_path = root / "generated_samples.json"
if not metrics_path.is_file() or not samples_path.is_file():
    raise SystemExit(1)
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
samples = json.loads(samples_path.read_text(encoding="utf-8"))
required = {
    "top_1_accuracy",
    "top_3_accuracy",
    "top_5_accuracy",
    "top_10_accuracy",
    "deduplicate_top_10_accuracy",
}
if not required.issubset(metrics) or len(samples) != 200:
    raise SystemExit(1)
PY
}

wait_for_current() {
    if [[ -z "${current_pid}" ]]; then
        return
    fi
    record "waiting_for_current_pid=${current_pid}"
    while kill -0 "${current_pid}" 2>/dev/null; do
        sleep 30
    done
    record "current_pid_finished=${current_pid}"
}

run_gate() {
    local step="$1"
    local attempt="$2"
    local output_dir="${project_root}/outputs/gate200_step${step}_rcpred_top10_s50_seed42_${attempt}"
    local run_log="${recovery_root}/gate200_step${step}_${attempt}.log"

    if valid_output "${output_dir}"; then
        record "step=${step} attempt=${attempt} already_complete output=${output_dir}"
        return 0
    fi

    record "step=${step} attempt=${attempt} started output=${output_dir}"
    (
        cd "${project_root}" || exit 1
        export PROJECT_ROOT="${project_root}"
        export PYTHONPATH="${compat_root}:${project_root}"
        "${python_bin}" -m src.test \
            experiment=retro_sampling \
            +model/sde=dfm \
            +model/net=graph_dit/8m \
            "ckpt_path=${checkpoint_root}/model_step${step}.ckpt" \
            "data.data_root=${data_root}" \
            data.batch_size=24 \
            data.num_workers=4 \
            data.graph_vocab=ch6_69pct_retrodit_unknown \
            data.forward_syn_model=null \
            data.n_dummy=30 \
            data.max_n_len=250 \
            data.shuffle_order=false \
            data.canonicalize=true \
            "data.external_perm_paths.test=${permutation_json}" \
            model.sample_store.forward_syn_cfg=null \
            'model.test_top_ks=[1,3,5,10]' \
            model.sde.test_sampling_steps=50 \
            trainer.accelerator=gpu \
            trainer.devices=1 \
            trainer.precision=bf16-mixed \
            extras.print_config=false \
            "hydra.run.dir=${output_dir}"
    ) > "${run_log}" 2>&1
    local exit_code=$?

    if [[ ${exit_code} -eq 0 ]] && valid_output "${output_dir}"; then
        record "step=${step} attempt=${attempt} completed output=${output_dir}"
        return 0
    fi
    record "step=${step} attempt=${attempt} failed exit_code=${exit_code} log=${run_log}"
    return 1
}

wait_for_current

current_output="${project_root}/outputs/gate200_step30000_rcpred_top10_s50_seed42_retry1"
if valid_output "${current_output}"; then
    record "step=30000 attempt=retry1 completed output=${current_output}"
elif ! run_gate 30000 retry2; then
    run_gate 30000 retry3 || exit 30
fi

for step in 50000 70000; do
    run_gate "${step}" auto1 || run_gate "${step}" auto2 || exit "${step:0:2}"
done

record "all_gate_runs_complete"
