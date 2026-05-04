import os
import subprocess
import sys


def test_desktop_app_smoke_mode_launches() -> None:
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
    }

    result = subprocess.run(
        [sys.executable, "-m", "yolo_trainer", "--smoke"],
        check=False,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "YOLO Trainer app smoke OK" in result.stdout
