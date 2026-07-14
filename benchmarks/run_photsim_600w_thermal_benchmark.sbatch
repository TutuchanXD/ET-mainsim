#!/usr/bin/env bash
#SBATCH --job-name=photsim_600w_therm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=36
#SBATCH --mem=160G
#SBATCH --time=00:20:00
#SBATCH --output=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%j.err

set -euo pipefail

# shellcheck source=/dev/null
source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-etbase-clu}"

ET_MAINSIM_ROOT="${ET_MAINSIM_ROOT:-/cluster/home/cxgao/ET/ET-mainsim}"
export ET_MAINSIM_ROOT
export PYTHONPATH="${ET_MAINSIM_ROOT}/src:${ET_MAINSIM_ROOT}:${PYTHONPATH:-}"
export ET_DATA_DIR="${ET_DATA_DIR:-/cluster/home/cxgao/ET/Photsim7-data}"
export ET_FOCALPLANE_ROOT="${ET_FOCALPLANE_ROOT:-/cluster/home/cxgao/ET/et_focalplane}"
: "${GAIA_CATALOG_DIR:?Set GAIA_CATALOG_DIR}"

CATALOG_CACHE="${CATALOG_CACHE:-/cluster/home/cxgao/sshfs-share/GRB/full_sim/main_rd_full_8900x9120_g17_sky22_subpix1_jipsf100_120x10s/cache/stars_main_rd_full_8900x9120_g17_sky22_subpix1_jipsf100_120x10s.npz}"
OUT="${OUT:-/cluster/home/cxgao/sshfs-share/benchmarks/photsim_600w_thermal_benchmark_${SLURM_JOB_ID}}"
test -f "${CATALOG_CACHE}"
mkdir -p "${OUT}" /cluster/home/cxgao/sshfs-share/slurm_logs
cd "${ET_MAINSIM_ROOT}"

QUERY=index,uuid,name,pstate,persistence_mode,clocks.current.sm,clocks.max.sm,clocks.current.memory,power.draw,power.limit,power.default_limit,power.min_limit,power.max_limit,temperature.gpu,temperature.memory,memory.used,memory.total,utilization.gpu,utilization.memory,clocks_throttle_reasons.active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.applications_clocks_setting,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.sync_boost
nvidia-smi --query-gpu="${QUERY}" --format=csv,noheader,nounits > "${OUT}/preflight_gpu.csv"
nvidia-smi -q -d POWER,CLOCK,PERFORMANCE,TEMPERATURE > "${OUT}/nvidia_smi_q_pre.txt"
if ! awk -F, 'BEGIN{ok=1} {gsub(/^ +| +$/, "", $10); if ($10 != "600.00") ok=0} END{exit ok ? 0 : 1}' "${OUT}/preflight_gpu.csv"; then
  echo "Expected every GPU power.limit to be 600.00 W; refusing benchmark." >&2
  exit 2
fi

pids=()
# shellcheck disable=SC2317  # Invoked indirectly by the trap below.
cleanup() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

rank=0
for gpu in 0 1 2; do
  for local_rank in 0 1 2 3; do
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      python -m benchmarks.run_full_frame_thermal_load \
        --output-root "${OUT}/runs" \
        --run-id "gpu${gpu}-rank${local_rank}" \
        --catalog-cache "${CATALOG_CACHE}" \
        --frames 1 \
        --mag-limit 17.0 \
        --max-stars 100000 \
        --jitter-models 100 \
        --seed "$((20260529 + rank))" \
        --gpu 0
    ) > "${OUT}/gpu${gpu}_rank${local_rank}.log" 2>&1 &
    pids+=("$!")
    rank=$((rank + 1))
  done
done

SAMPLE_QUERY=timestamp,index,uuid,pstate,clocks.current.sm,clocks.max.sm,clocks.current.memory,utilization.gpu,utilization.memory,power.draw,power.limit,power.default_limit,temperature.gpu,temperature.memory,memory.used,memory.total,clocks_throttle_reasons.active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.applications_clocks_setting,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.sync_boost
: > "${OUT}/nvidia_smi_samples.csv"
while true; do
  live=0
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      live=1
    fi
  done
  nvidia-smi --query-gpu="${SAMPLE_QUERY}" --format=csv,noheader,nounits >> "${OUT}/nvidia_smi_samples.csv"
  [[ "${live}" -eq 1 ]] || break
  sleep 1
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
trap - EXIT INT TERM
nvidia-smi --query-gpu="${QUERY}" --format=csv,noheader,nounits > "${OUT}/postflight_gpu.csv"
nvidia-smi -q -d POWER,CLOCK,PERFORMANCE,TEMPERATURE > "${OUT}/nvidia_smi_q_post.txt"
python benchmarks/evaluate_main_rd_benchmark.py "${OUT}/runs" --output-prefix "${OUT}/benchmark"
exit "${status}"
