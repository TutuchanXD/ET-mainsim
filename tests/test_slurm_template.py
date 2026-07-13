from pathlib import Path


def test_full_frame_slurm_logs_are_account_portable() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "slurm" / "et_full_frame.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --output=%x-%j.out" in script
    assert "#SBATCH --error=%x-%j.err" in script
    assert "/cluster/home/cxgao/sshfs-share/slurm_logs" not in script
