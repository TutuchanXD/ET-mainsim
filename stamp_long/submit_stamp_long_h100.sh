#!/usr/bin/env bash
# Submit with:
#   sbatch --test-only /cluster/home/cxgao/ET/ET-mainsim/stamp_long/submit_stamp_long_h100.sh
#   sbatch /cluster/home/cxgao/ET/ET-mainsim/stamp_long/submit_stamp_long_h100.sh
#
# Runtime overrides:
#   STAGE=smoke|compute|io|physics|jitter_sensitivity
#   WORKERS_PER_GPU=10
#   MATRIX_PRESET=stamp_scale_v2
#   SCALE_GROUP=short_high_star|long_low_star
#   CASE_IDS=S1D11E300,S1D15E030
#   MAX_CASES=1
#   WRITE_MODE=sample|all|none
#   OUTPUT_FORMAT=npy|hdf5
#   DEVICE=cuda|auto|cpu
#   STAR_FLUX_MODE=random_et_mag|fixed
#   ET_MAG_MIN=12.5
#   ET_MAG_MAX=14.5
#   COSMIC_RAY_LIBRARY=/cluster/home/cxgao/ET/Photsim7-data/path/to/events.npz
#   PSF_BUNDLE_NAME=psf/et/241006/D280mm-focus
#   JITTER_VARIANTS=100x200,100x300,200x400,300x600
#   JITTER_CASES=J030S11,J300S15
#   JITTER_MODEL_SAMPLES=3
#   SAVE_JITTER_ARRAYS=1

#SBATCH --job-name=stamp_long_h100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=72
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/cxgao/sshfs-share/slurm_logs/%x-%j.err

set -euo pipefail

STAGE="${STAGE:-physics}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-10}"
GPUS="${GPUS:-0,1,2}"
SEED="${SEED:-20260617}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
if [[ -z "${OUTPUT_ROOT}" ]]; then
  case "${STAGE}" in
    jitter|jitter_sensitivity)
      OUTPUT_ROOT="/cluster/home/cxgao/sshfs-share/ET-mainsim/stamp_long_jitter_sensitivity"
      ;;
    *)
      OUTPUT_ROOT="/cluster/home/cxgao/sshfs-share/ET-mainsim/stamp_long"
      ;;
  esac
fi
CASE_IDS="${CASE_IDS:-}"
MATRIX_PRESET="${MATRIX_PRESET:-}"
SCALE_GROUP="${SCALE_GROUP:-}"
EXPOSURES="${EXPOSURES:-}"
STAMP_SIZES="${STAMP_SIZES:-}"
MAX_CASES="${MAX_CASES:-}"
WRITE_MODE="${WRITE_MODE:-}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-}"
if [[ -z "${OUTPUT_FORMAT}" ]]; then
  case "${STAGE}" in
    physics|io)
      OUTPUT_FORMAT="hdf5"
      ;;
    *)
      OUTPUT_FORMAT="npy"
      ;;
  esac
fi
SAMPLE_LIMIT="${SAMPLE_LIMIT:-1}"
DRY_RUN="${DRY_RUN:-0}"
STAR_FLUX_E_S="${STAR_FLUX_E_S:-100.0}"
STAR_FLUX_MODE="${STAR_FLUX_MODE:-random_et_mag}"
ET_MAG_MIN="${ET_MAG_MIN:-12.5}"
ET_MAG_MAX="${ET_MAG_MAX:-14.5}"
BACKGROUND_E_S_PIX="${BACKGROUND_E_S_PIX:-26.0}"
SCATTERED_LIGHT_E_S_PIX="${SCATTERED_LIGHT_E_S_PIX:-5.0}"
DARK_E_S_PIX="${DARK_E_S_PIX:-1.0}"
READ_NOISE_10S_E_PIX="${READ_NOISE_10S_E_PIX:-5.0}"
GAIN_E_PER_ADU="${GAIN_E_PER_ADU:-1.4}"
COSMIC_RAY_EVENT_RATE="${COSMIC_RAY_EVENT_RATE:-5.0}"
COSMIC_RAY_LIBRARY="${COSMIC_RAY_LIBRARY:-cosmic_ray/dark_test_10um/event_library_10um.npz}"
COSMIC_RAY_PEAK_ADU="${COSMIC_RAY_PEAK_ADU:-4000.0}"
PIXEL_SIZE_UM="${PIXEL_SIZE_UM:-10.0}"
PSF_SIGMA_PIX="${PSF_SIGMA_PIX:-1.25}"
PSF_BUNDLE_NAME="${PSF_BUNDLE_NAME:-psf/et/241006/D280mm-focus}"
PSF_FIELD_ID="${PSF_FIELD_ID:-6}"
PSF_SUBPIXELS="${PSF_SUBPIXELS:-7}"
PSD_MOTION_PATH="${PSD_MOTION_PATH:-pds/ET_psd3-2.pkl}"
DVA_MODEL_PATH="${DVA_MODEL_PATH:-DVA/et/ET_DVA_effect_models_slim_v231117.pkl}"
JITTER_PSF_MODELS="${JITTER_PSF_MODELS:-300}"
JITTER_FRAMES_PER_MODEL="${JITTER_FRAMES_PER_MODEL:-600}"
JITTER_VARIANTS="${JITTER_VARIANTS:-100x200,100x300,200x400,300x600}"
JITTER_CASES="${JITTER_CASES:-}"
JITTER_MODEL_SAMPLES="${JITTER_MODEL_SAMPLES:-3}"
SAVE_JITTER_ARRAYS="${SAVE_JITTER_ARRAYS:-1}"
DYNAMIC_EFFECTS="${DYNAMIC_EFFECTS:-1}"
PSD_MOTION="${PSD_MOTION:-1}"
DVA_DRIFT="${DVA_DRIFT:-1}"
THERMAL_DRIFT="${THERMAL_DRIFT:-1}"
MOMENTUM_DUMP="${MOMENTUM_DUMP:-1}"
PSF_BREATHING="${PSF_BREATHING:-1}"
PHOTSIM7_PSF="${PHOTSIM7_PSF:-1}"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
date
echo "stage=${STAGE}"
echo "output_root=${OUTPUT_ROOT}"
echo "matrix_preset=${MATRIX_PRESET:-unset}"
echo "scale_group=${SCALE_GROUP:-unset}"
echo "workers_per_gpu=${WORKERS_PER_GPU}"
echo "output_format=${OUTPUT_FORMAT}"
echo "star_flux_mode=${STAR_FLUX_MODE}"
echo "et_mag_range=${ET_MAG_MIN},${ET_MAG_MAX}"
echo "jitter_psf=${JITTER_PSF_MODELS}x${JITTER_FRAMES_PER_MODEL}"
if [[ "${STAGE}" == "jitter" || "${STAGE}" == "jitter_sensitivity" ]]; then
  echo "jitter_variants=${JITTER_VARIANTS}"
  echo "jitter_cases=${JITTER_CASES:-default}"
  echo "jitter_model_samples=${JITTER_MODEL_SAMPLES}"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true

source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate etbase-clu

export ET_ROOT="${ET_ROOT:-/cluster/home/cxgao/ET}"
export ET_MAINSIM_ROOT="${ET_MAINSIM_ROOT:-${ET_ROOT}/ET-mainsim}"
export PHOTSIM7_ROOT="${PHOTSIM7_ROOT:-${ET_ROOT}/Photsim7}"
export ET_DATA_DIR="${ET_DATA_DIR:-${ET_ROOT}/Photsim7-data}"
export PHOTSIM7_DATA_DIR="${PHOTSIM7_DATA_DIR:-${ET_DATA_DIR}}"
export RESULTS_ROOT="${OUTPUT_ROOT}"

cd "${ET_MAINSIM_ROOT}"

echo "python=$(command -v python)"
python - <<'PY'
import os
import subprocess
import sys

print("python_version=" + sys.version.replace("\n", " "))
for path in [os.environ["ET_MAINSIM_ROOT"], os.environ["PHOTSIM7_ROOT"]]:
    try:
        commit = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        commit = f"unavailable: {exc}"
    print(f"git_commit {path} {commit}")
try:
    import torch
    print(f"torch_version={torch.__version__}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
except Exception as exc:
    print(f"torch_error={exc}")
print(f"ET_DATA_DIR={os.environ.get('ET_DATA_DIR')}")
PY

case "${STAGE}" in
  smoke)
    RUNNER="stamp_long/run_stamp_long_smoke.py"
    ;;
  compute)
    RUNNER="stamp_long/run_stamp_long_compute_benchmark.py"
    ;;
  io)
    RUNNER="stamp_long/run_stamp_long_io_benchmark.py"
    ;;
  physics)
    RUNNER="stamp_long/run_stamp_long_physics_benchmark.py"
    ;;
  jitter|jitter_sensitivity)
    RUNNER="stamp_long/run_stamp_long_jitter_sensitivity.py"
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; expected smoke, compute, io, physics, or jitter_sensitivity" >&2
    exit 2
    ;;
esac

cmd=(
  python
  "${RUNNER}"
  --output-root "${OUTPUT_ROOT}"
  --workers-per-gpu "${WORKERS_PER_GPU}"
  --gpus "${GPUS}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --output-format "${OUTPUT_FORMAT}"
  --sample-limit "${SAMPLE_LIMIT}"
  --star-flux-e-s "${STAR_FLUX_E_S}"
  --star-flux-mode "${STAR_FLUX_MODE}"
  --et-mag-min "${ET_MAG_MIN}"
  --et-mag-max "${ET_MAG_MAX}"
  --background-e-s-pix "${BACKGROUND_E_S_PIX}"
  --scattered-light-e-s-pix "${SCATTERED_LIGHT_E_S_PIX}"
  --dark-e-s-pix "${DARK_E_S_PIX}"
  --read-noise-10s-e-pix "${READ_NOISE_10S_E_PIX}"
  --gain-e-per-adu "${GAIN_E_PER_ADU}"
  --cosmic-ray-event-rate "${COSMIC_RAY_EVENT_RATE}"
  --cosmic-ray-library "${COSMIC_RAY_LIBRARY}"
  --cosmic-ray-peak-adu "${COSMIC_RAY_PEAK_ADU}"
  --pixel-size-um "${PIXEL_SIZE_UM}"
  --psf-sigma-pix "${PSF_SIGMA_PIX}"
  --psf-bundle-name "${PSF_BUNDLE_NAME}"
  --psf-field-id "${PSF_FIELD_ID}"
  --psf-subpixels "${PSF_SUBPIXELS}"
  --psd-motion-path "${PSD_MOTION_PATH}"
  --dva-model-path "${DVA_MODEL_PATH}"
  --jitter-psf-models "${JITTER_PSF_MODELS}"
  --jitter-frames-per-model "${JITTER_FRAMES_PER_MODEL}"
)

if [[ -n "${CASE_IDS}" ]]; then
  cmd+=(--case-ids "${CASE_IDS}")
fi
if [[ -n "${MATRIX_PRESET}" ]]; then
  cmd+=(--matrix-preset "${MATRIX_PRESET}")
fi
if [[ -n "${SCALE_GROUP}" ]]; then
  cmd+=(--scale-group "${SCALE_GROUP}")
fi
if [[ -n "${EXPOSURES}" ]]; then
  cmd+=(--exposures "${EXPOSURES}")
fi
if [[ -n "${STAMP_SIZES}" ]]; then
  cmd+=(--stamp-sizes "${STAMP_SIZES}")
fi
if [[ -n "${MAX_CASES}" ]]; then
  cmd+=(--max-cases "${MAX_CASES}")
fi
if [[ -n "${WRITE_MODE}" ]]; then
  cmd+=(--write-mode "${WRITE_MODE}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi
if [[ "${STAGE}" == "jitter" || "${STAGE}" == "jitter_sensitivity" ]]; then
  cmd+=(--variants "${JITTER_VARIANTS}")
  cmd+=(--model-samples "${JITTER_MODEL_SAMPLES}")
  if [[ -n "${JITTER_CASES}" ]]; then
    cmd+=(--cases "${JITTER_CASES}")
  fi
  if [[ "${SAVE_JITTER_ARRAYS}" == "1" ]]; then
    cmd+=(--save-arrays)
  else
    cmd+=(--no-save-arrays)
  fi
fi
if [[ "${PHOTSIM7_PSF}" == "1" ]]; then
  cmd+=(--photsim7-psf)
else
  cmd+=(--no-photsim7-psf)
fi
if [[ "${DYNAMIC_EFFECTS}" == "1" ]]; then
  cmd+=(--dynamic-effects)
else
  cmd+=(--no-dynamic-effects)
fi
if [[ "${PSD_MOTION}" == "1" ]]; then
  cmd+=(--psd-motion)
else
  cmd+=(--no-psd-motion)
fi
if [[ "${DVA_DRIFT}" == "1" ]]; then
  cmd+=(--dva-drift)
else
  cmd+=(--no-dva-drift)
fi
if [[ "${THERMAL_DRIFT}" == "1" ]]; then
  cmd+=(--thermal-drift)
else
  cmd+=(--no-thermal-drift)
fi
if [[ "${MOMENTUM_DUMP}" == "1" ]]; then
  cmd+=(--momentum-dump)
else
  cmd+=(--no-momentum-dump)
fi
if [[ "${PSF_BREATHING}" == "1" ]]; then
  cmd+=(--psf-breathing)
else
  cmd+=(--no-psf-breathing)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

date
