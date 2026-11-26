#!/usr/bin/env bash
set -uo pipefail

project_root="/root/rivermind-data/retrodit_recovery/RetroDiT-latest"
python_bin="/root/rivermind-data/envs/retrodit-gen/bin/python"
checkpoint="/root/rivermind-data/retrodit_recovery/payload/checkpoints/model_step50000.ckpt"
compat_root="/root/rivermind-data/retrodit_recovery/flash_compat"
recovery_root="/root/rivermind-data/retrodit_recovery"
data_root="${project_root}/data/ch6_69pct_50k"
permutation_json="${project_root}/rc_prediction/outputs/ch6_69pct_rc_seed42_export/predictions/test/retrodit_rc_ch6_seed42_strict_root_center_top7_root_top7_temp1p25_agg_split_edge.json"
prepare_output="${project_root}/outputs/ch6_69pct_full_prepare"
inference_output="${project_root}/outputs/ch6_69pct_full_step50000_rcpred_top10_s50_seed42"
prepare_log="${recovery_root}/ch6_69pct_full_prepare.log"
inference_log="${recovery_root}/ch6_69pct_full_step50000.log"
queue_log="${recovery_root}/ch6_69pct_full_queue.log"
processed_root="${data_root}/processed/ndummy_30_maxlen_250_cano"

record() {
    printf '%s\t%s\n' "$(date '+%F %T %Z')" "$*" >> "${queue_log}"
}

export PROJECT_ROOT="${project_root}"
export PYTHONPATH="${compat_root}:${project_root}"

if [[ ! -s "${processed_root}/train.pt" || ! -s "${processed_root}/val.pt" || ! -s "${processed_root}/test.pt" ]]; then
    record "full_preprocessing_started data_root=${data_root}"
    (
        cd "${project_root}" || exit 1
        "${python_bin}" -m src.setup_dataset \
            experiment=retro \
            +model/sde=dfm \
            +model/net=graph_dit/8m \
            task_name=ch6_69pct_full_prepare \
            "data.data_root=${data_root}" \
            data.batch_size=24 \
            data.num_workers=4 \
            data.graph_vocab=ch6_69pct_retrodit_unknown \
            data.forward_syn_model=null \
            data.n_dummy=30 \
            data.max_n_len=250 \
            data.shuffle_order=false \
            data.canonicalize=true \
            data.perm_prob_sampler=null \
            '~logger' \
            extras.print_config=false \
            "hydra.run.dir=${prepare_output}"
    ) > "${prepare_log}" 2>&1
    prepare_exit=$?
    if [[ ${prepare_exit} -ne 0 || ! -s "${processed_root}/train.pt" || ! -s "${processed_root}/val.pt" || ! -s "${processed_root}/test.pt" ]]; then
        record "full_preprocessing_failed exit_code=${prepare_exit} log=${prepare_log}"
        exit 40
    fi
    record "full_preprocessing_completed processed_root=${processed_root}"
else
    record "full_preprocessing_already_complete processed_root=${processed_root}"
fi

if [[ -s "${inference_output}/generated_metrics.json" && -s "${inference_output}/generated_samples.json" ]]; then
    record "full_inference_already_complete output=${inference_output}"
    exit 0
fi

record "full_inference_started checkpoint=${checkpoint} output=${inference_output}"
(
    cd "${project_root}" || exit 1
    "${python_bin}" -m src.test \
        experiment=retro_sampling \
        +model/sde=dfm \
        +model/net=graph_dit/8m \
        "ckpt_path=${checkpoint}" \
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
        "hydra.run.dir=${inference_output}"
) > "${inference_log}" 2>&1
inference_exit=$?

if [[ ${inference_exit} -eq 0 && -s "${inference_output}/generated_metrics.json" && -s "${inference_output}/generated_samples.json" ]]; then
    record "full_inference_completed output=${inference_output}"
    exit 0
fi

record "full_inference_failed exit_code=${inference_exit} log=${inference_log}"
exit 50
