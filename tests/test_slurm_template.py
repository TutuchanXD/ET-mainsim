from pathlib import Path


def test_full_frame_slurm_logs_are_account_portable() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "slurm" / "et_full_frame.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --output=%x-%j.out" in script
    assert "#SBATCH --error=%x-%j.err" in script
    assert "/cluster/home/cxgao/sshfs-share/slurm_logs" not in script
    assert 'export PYTHONPATH="${ET_MAINSIM_ROOT}/src:' in script
    assert 'if [[ -z "${ET_CATALOG_CACHE:-}" ]]' in script


def test_stamp_and_legacy_slurm_templates_use_maintained_cli() -> None:
    root = Path(__file__).resolve().parents[1] / "slurm"
    stamp = (root / "et_stamp.sbatch").read_text(encoding="utf-8")
    legacy = (root / "legacy_sim.sbatch").read_text(encoding="utf-8")

    assert "python -m et_mainsim run et-stamp" in stamp
    assert "--preset \"${PRESET:-production}\"" in stamp
    assert 'if [[ -n "${INPUT_TABLE:-}" ]]' in stamp
    assert 'args+=(--input-table "${INPUT_TABLE}")' in stamp
    assert 'if [[ -n "${VARIABILITY_TABLE:-}" ]]' in stamp
    assert 'args+=(--variability-table "${VARIABILITY_TABLE}")' in stamp
    assert '-z "${ET_CATALOG_CACHE:-}"' in stamp
    assert "python -m et_mainsim run legacy-sim" in legacy
    assert "--preset \"${PRESET:-full-effects-production}\"" in legacy
    assert 'args+=(--frames "${FRAMES}")' in legacy
    assert 'args+=(--stars-per-run "${STARS_PER_RUN}")' in legacy
    assert "main_rd_g18_parallel" not in stamp + legacy


def test_h100_validation_distinguishes_physical_and_visible_gpu_ids() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "slurm"
        / "validate_source_variability.sbatch"
    ).read_text(encoding="utf-8")

    assert 'visible_gpu="${CUDA_VISIBLE_DEVICES%%,*}"' in script
    assert 'physical_gpu="${SLURM_JOB_GPUS%%,*}"' in script
    assert '--id="${physical_gpu}"' in script
    assert '--gpus "${visible_gpu}"' in script
    assert '--focalplane-registry "${ET_FOCALPLANE_ROOT%/}/data"' in script
