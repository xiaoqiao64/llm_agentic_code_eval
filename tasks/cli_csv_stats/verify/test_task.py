import subprocess
import sys


def test_csv_stats_output():
    proc = subprocess.run(
        [sys.executable, "csv_stats.py", "data.csv"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "count=3 sum=60 avg=20.0"
