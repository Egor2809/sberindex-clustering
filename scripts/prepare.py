"""Rebuild the canonical panel without overwriting the published preparation reports."""
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbercluster.canonical import prepare
from sbercluster.data import PREPARED_FILES

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    inputs = list((root / "data/raw").glob("*.csv"))
    if len(inputs) > 1:
        raise SystemExit("Expected zero or one CSV reference")
    published = sorted(name for name in PREPARED_FILES if name.startswith("reports/") and (root / name).is_file())
    keep = root / "artifacts/preparation/.published"
    for name in published:
        (keep / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, keep / name)
    try:
        result = prepare(root)
    finally:
        rerun = root / "artifacts/preparation"
        for name in published:
            if (root / name).is_file():
                shutil.copyfile(root / name, rerun / Path(name).name)
            shutil.copyfile(keep / name, root / name)
        shutil.rmtree(keep)
    print(f"Prepared {result['complete_panel_territories']} territories x {result['n_months']} months. No fitting performed.")
