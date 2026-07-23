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


def test_galaxy_standard_analysis_slurm_launcher_is_fail_closed() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "galaxy_standard_stamp_analysis_slurm.sh"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --time=12:00:00" in script
    assert "#SBATCH --gres=" not in script
    assert ': "${ET_STAMP_MANIFEST:?Set ET_STAMP_MANIFEST to production_manifest.json}"' in script
    assert ': "${ET_STAMP_ANALYSIS_SOURCE_ID:?Set ET_STAMP_ANALYSIS_SOURCE_ID to one source ID}"' in script
    assert 'python -m et_mainsim.standard_stamp_analysis' in script
    assert "--case injected" in script
    assert "--cadence-seconds 60" in script
    assert "--output-dir \"${OUTPUT_DIR}\"" in script
    assert "--overwrite" not in script


def test_galaxy_standard_analysis_array_launcher_is_serial_and_manifest_driven() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "galaxy_standard_stamp_analysis_array_slurm.sh"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --time=12:00:00" in script
    assert "#SBATCH --array=0-9%1" in script
    assert "#SBATCH --gres=" not in script
    assert 'ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?This launcher must be submitted as an array job}"' in script
    assert script.index("conda activate etbase-clu") < script.index(
        'python - "${ET_STAMP_MANIFEST}" "${ARRAY_INDEX}"'
    )
    assert "if len(targets) != 10:" in script
    assert "source_id = int(targets[array_index][\"source_id_int64\"])" in script
    assert 'python -m et_mainsim.standard_stamp_analysis' in script
    assert "--case injected" in script
    assert "--cadence-seconds 60" in script
    assert "--output-dir \"${OUTPUT_DIR}\"" in script
    assert "--overwrite" not in script


def test_galaxy_raw_coverage_analysis_launchers_are_qc_gated_and_immutable() -> None:
    root = Path(__file__).resolve().parents[1] / "scripts"
    strict = (root / "galaxy_raw_strict_analysis_array_slurm.sh").read_text(
        encoding="utf-8"
    )
    coverage = (root / "galaxy_raw_coverage_aware_analysis_array_slurm.sh").read_text(
        encoding="utf-8"
    )

    for script in (strict, coverage):
        assert "#SBATCH --partition=cpu" in script
        assert "#SBATCH --array=0-9%1" in script
        assert "#SBATCH --gres=" not in script
        assert (
            'ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?This launcher must be submitted as an array job}"'
            in script
        )
        assert "if len(targets) != 10:" in script
        assert 'source_id = int(targets[array_index]["source_id_int64"])' in script
        assert "injected_campaign_delivery_qc.json" in script
        assert 'qc.get("ready") is not True' in script
        assert "--overwrite" not in script

    assert "python -m et_mainsim.standard_stamp_analysis" in strict
    assert "--case injected" in strict
    assert "--cadence-seconds 10" in strict
    assert (
        'OUTPUT_DIR="${RUN_ROOT}/analysis/source_${SOURCE_ID}/injected/raw_10s_strict"'
        in strict
    )

    assert (
        ': "${ET_STAMP_COVERAGE_POLICY_JSON:?Set ET_STAMP_COVERAGE_POLICY_JSON'
        in coverage
    )
    assert "ET_STAMP_MINIMUM_COVERAGE_FRACTION" not in coverage
    assert "ET_STAMP_MINIMUM_ACCEPTED_BINS" not in coverage
    assert "ET_STAMP_BIN_ORIGIN_SECONDS" not in coverage
    assert "python -m et_mainsim.coverage_aware_stamp_analysis" in coverage
    assert (
        'REFERENCE_ANALYSIS_DIR="${RUN_ROOT}/analysis/source_${SOURCE_ID}/injected/raw_10s_strict"'
        in coverage
    )
    assert (
        'OUTPUT_DIR="${RUN_ROOT}/analysis/source_${SOURCE_ID}/injected/raw_10s_coverage_v2"'
        in coverage
    )
    assert '--campaign-qc "${ET_STAMP_CAMPAIGN_QC_JSON}"' in coverage
    assert '--coverage-policy "${ET_STAMP_COVERAGE_POLICY_JSON}"' in coverage


def test_galaxy_campaign_qc_slurm_launcher_is_a_fail_closed_analysis_gate() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "galaxy_campaign_delivery_qc_slurm.sh"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=16G" in script
    assert "#SBATCH --time=02:00:00" in script
    assert "#SBATCH --gres=" not in script
    assert (
        ': "${ET_STAMP_MANIFEST:?Set ET_STAMP_MANIFEST to production_manifest.json}"'
        in script
    )
    assert (
        ': "${ET_STAMP_CODE_ROOT:?Set ET_STAMP_CODE_ROOT to the deployed ET-mainsim checkout}"'
        in script
    )
    assert script.index("conda activate etbase-clu") < script.index(
        "scripts/audit_galaxy_campaign_delivery.py"
    )
    assert "--case injected" in script
    assert "--require-complete" in script
    assert (
        'OUTPUT_JSON="${RUN_ROOT}/quality_control/injected_campaign_delivery_qc.json"'
        in script
    )


def test_galaxy_raw_coverage_campaign_summary_launcher_is_policy_gated() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "galaxy_raw_coverage_campaign_summary_slurm.sh"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=16G" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "#SBATCH --array=" not in script
    assert "#SBATCH --gres=" not in script
    assert (
        ': "${ET_STAMP_COVERAGE_POLICY_JSON:?Set ET_STAMP_COVERAGE_POLICY_JSON'
        in script
    )
    assert "afterok:<coverage-array-job-id>" in script
    assert "python -m et_mainsim.raw_coverage_campaign_summary" in script
    assert '--campaign-qc "${ET_STAMP_CAMPAIGN_QC_JSON}"' in script
    assert '--coverage-policy "${ET_STAMP_COVERAGE_POLICY_JSON}"' in script
    assert (
        'OUTPUT_DIR="${RUN_ROOT}/analysis/campaign/injected/raw_10s_coverage_v2_summary"'
        in script
    )
    assert "--overwrite" not in script


def test_galaxy_staged_render_launcher_renders_locally_then_publishes_atomically() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "galaxy_independent_stamp_staged_slurm_array.sh"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --cpus-per-task=16" in script
    assert ': "${ET_STAMP_MANIFEST:?Set ET_STAMP_MANIFEST to production_manifest.json}"' in script
    assert ': "${ET_STAMP_CODE_ROOT:?Set ET_STAMP_CODE_ROOT to the deployed ET-mainsim checkout}"' in script
    assert 'LOCAL_CASE_ROOT="${JOB_SCRATCH}/cases/${ET_STAMP_CASE}"' in script
    assert '"${ET_STAMP_CODE_ROOT}/scripts/run_galaxy_independent_stamp_production.py" run-target' in script
    assert '--output-root "${LOCAL_CASE_ROOT}"' in script
    assert "python -m et_mainsim.staged_stamp_delivery publish" in script
    assert '--staged-case-root "${LOCAL_CASE_ROOT}"' in script
    assert '--formal-case-root "${RUN_ROOT}/cases/${ET_STAMP_CASE}"' in script
    assert 'rm -rf -- "${JOB_SCRATCH}"' in script
