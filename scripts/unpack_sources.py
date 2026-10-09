"""Extract only the four expected members of the hash-pinned official ZIP."""
from pathlib import Path
import hashlib
import zipfile
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"artifacts/sources/acquisition"
expected="a9f932ff4096a7df797d1547987937f34d3995ac445b4748177114488d12b010"
archive=SOURCE/"hackathonlicence.zip"
if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:
    raise ValueError("Official ZIP snapshot differs")
out=SOURCE/"hackathonlicence"
out.mkdir(parents=True,exist_ok=True)
allow={"consumption.parquet","market_access.parquet","connection.parquet","Данные_СберИндекс_лицензия.pdf"}
def archive_name(info):
    name=info.filename
    # This hash-pinned Windows ZIP uses CP866 without the UTF-8 flag.
    if not (info.flag_bits & 0x800):
        try:
            name=name.encode("cp437").decode("cp866")
        except UnicodeEncodeError:
            pass
    return Path(name).name

with zipfile.ZipFile(archive) as z:
    selected=[i for i in z.infolist() if not i.is_dir() and archive_name(i) in allow]
    if len(selected)!=len(allow) or {archive_name(i) for i in selected}!=allow:
        raise ValueError("Unexpected archive members")
    for i in selected:
        path=out/archive_name(i)
        content=z.read(i)
        if path.exists() and path.read_bytes()!=content:
            raise ValueError(f"Existing source differs: {path.name}")
        if not path.exists(): path.write_bytes(content)
print("Official source files verified")
