from pathlib import Path


def test_full_frame_slurm_logs_are_account_portable() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "slurm" / "et_full_frame.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --output=%x-%j.out" in script
    assert "#SBATCH --error=%x-%j.err" in script
    assert "/cluster/home/cxgao/sshfs-share/slurm_logs" not in script
    assert 'export PYTHONPATH="${ET_MAINSIM_ROOT}/src:' in script


def test_stamp_and_legacy_slurm_templates_use_maintained_cli() -> None:
    root = Path(__file__).resolve().parents[1] / "slurm"
    stamp = (root / "et_stamp.sbatch").read_text(encoding="utf-8")
    legacy = (root / "legacy_sim.sbatch").read_text(encoding="utf-8")

    assert "python -m et_mainsim run et-stamp" in stamp
    assert "--preset \"${PRESET:-production}\"" in stamp
    assert 'if [[ -n "${INPUT_TABLE:-}" ]]' in stamp
    assert 'args+=(--input-table "${INPUT_TABLE}")' in stamp
    assert "python -m et_mainsim run legacy-sim" in legacy
    assert "--preset \"${PRESET:-full-effects-production}\"" in legacy
    assert 'args+=(--frames "${FRAMES}")' in legacy
    assert 'args+=(--stars-per-run "${STARS_PER_RUN}")' in legacy
    assert "main_rd_g18_parallel" not in stamp + legacy
