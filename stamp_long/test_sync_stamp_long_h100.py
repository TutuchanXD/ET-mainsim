from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "sync_stamp_long_h100.sh"


def _copy_script_and_fake_commands(tmp_path: Path) -> tuple[Path, Path]:
    checkout = tmp_path / "ET"
    script_dir = checkout / "ET-mainsim" / "stamp_long"
    script_dir.mkdir(parents=True)
    copied_script = script_dir / SCRIPT.name
    shutil.copy2(SCRIPT, copied_script)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "commands.log"
    command = '#!/usr/bin/env bash\nprintf "%s\\n" "$0 $*" >> "$COMMAND_LOG"\n'
    for name in ("ssh", "rsync"):
        executable = bin_dir / name
        executable.write_text(command, encoding="utf-8")
        executable.chmod(0o755)
    return copied_script, log_path


def test_sync_script_requires_remote_host(tmp_path):
    script, log_path = _copy_script_and_fake_commands(tmp_path)
    env = os.environ.copy()
    env.pop("REMOTE", None)
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["COMMAND_LOG"] = str(log_path)

    result = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode != 0
    assert "REMOTE" in result.stderr
    assert not log_path.exists()


def test_sync_script_derives_local_et_root_from_checkout(tmp_path):
    script, log_path = _copy_script_and_fake_commands(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "REMOTE": "cluster.example",
            "PATH": f"{tmp_path / 'bin'}:{env['PATH']}",
            "COMMAND_LOG": str(log_path),
        }
    )
    env.pop("LOCAL_ET_ROOT", None)

    result = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    command_log = log_path.read_text(encoding="utf-8")
    assert f"{tmp_path}/ET/ET-mainsim/" in command_log
    assert f"{tmp_path}/ET/Photsim7/" in command_log
    assert f"{tmp_path}/ET/Photsim7-data/" in command_log
