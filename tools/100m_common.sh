#!/usr/bin/env bash

source tools/dataset_path.sh

run_rotlex() {
  if [[ "$1" == python ]]; then shift; fi
  "${PYTHON_BIN:-python}" "$@"
}

run_heavy() (
  export OMP_NUM_THREADS="${PREPARATION_THREADS:-64}"
  export MKL_NUM_THREADS="${PREPARATION_THREADS:-64}"
  export OPENBLAS_NUM_THREADS="${PREPARATION_THREADS:-64}"
  run_rotlex "$@"
)

prepare_100m_experiment() {
  prepare_dataset_dir \
    "data/${DATASET_SLUG}" \
    "${DATASET_SOURCE_DIR}" \
    "${BASE_FILE}" \
    "${TRAIN_QUERY_FILE}" \
    "${TRAIN_GT_FILE}" \
    "${TEST_QUERY_FILE}" \
    "${TEST_GT_FILE}"
  run_heavy python -c "from ft_anns.datasets import preflight_dataset; print(preflight_dataset('${DATASET_NAME}', 'data/${DATASET_SLUG}'))"
  run_heavy python -m ft_anns prepare-workloads --config "${WORKLOAD_CONFIG}"
}

train_100m_ramps() {
  if [[ -f "${RUN_DIR}/best/metadata.json" ]]; then
    echo "RAMPS Best Checkpoint already exists: ${RUN_DIR}/best"
  elif [[ -f "${RUN_DIR}/last/training_state.pt" ]]; then
    run_heavy python -m ft_anns train --config "${TRAIN_CONFIG}" --resume "${RUN_DIR}/last"
  else
    run_heavy python -m ft_anns train --config "${TRAIN_CONFIG}"
  fi
}

evaluate_fixed_100m_sweep() {
  local widths
  local width
  local report_dir
  report_dir=${REPORT_DIR:-"reports/${DATASET_SLUG}"}
  mkdir -p "${report_dir}"
  widths=$(run_rotlex python -c "import json; [print(v) for v in json.load(open('${MATRIX_CONFIG}'))['fixed_bucket_counts']]")
  run_rotlex python -m ft_anns search-sweep \
    --checkpoint "${RUN_DIR}/best" \
    --partition "${PARTITION_DIR}" \
    --queries "data/${DATASET_SLUG}/${TEST_QUERY_FILE}" \
    --query-format "${TEST_QUERY_FORMAT}" \
    --fixed-topms ${widths//$'\n'/ } \
    --topk 100 \
    --batch-size 1000 \
    --faiss-threads 16 \
    --device cuda \
    --search-execution grouped \
    --output-dir "${RUN_DIR}" \
    --resume
  while read -r width; do
    run_rotlex python -m ft_anns evaluate \
      --config "${EVALUATION_CONFIG}" \
      --results "${RUN_DIR}/fixed_m${width}.npz" \
      --output "${report_dir}/ramps_fixed_m${width}.json"
  done <<< "${widths}"
}

report_100m_matrix() {
  run_rotlex python -m ft_anns report-matrix --config "${MATRIX_CONFIG}"
}

run_100m_matrix() {
  mkdir -p "${REPORT_DIR:-reports/${DATASET_SLUG}}" "${RUN_DIR}"
  prepare_100m_experiment
  train_100m_ramps
  evaluate_fixed_100m_sweep
  # Adaptive Query is intentionally deferred until its policy mass is
  # redesigned around raw Router logits for max-normalized RAMPS.
  report_100m_matrix
}
