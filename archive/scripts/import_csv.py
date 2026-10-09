"""Optional original dashboard export import, never overwrite differing input."""
import argparse
from pathlib import Path
import zipfile
parser=argparse.ArgumentParser()
parser.add_argument("archive",type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]/"data/raw"
root.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(args.archive) as z:
    files=[i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(".csv")]
    if len(files)!=1: raise ValueError("Expected exactly one CSV")
    member=files[0]
    path=root/Path(member.filename).name
    data=z.read(member)
    if path.exists() and path.read_bytes()!=data: raise ValueError("Existing CSV differs")
    if not path.exists(): path.write_bytes(data)
print(path)
