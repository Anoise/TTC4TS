"""Download original ETT files; preserve provenance and SHA256 checksums."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
for directory in ['workspace', 'logs', 'data']:
    (ROOT / directory).mkdir(parents=True, exist_ok=True)
BASE = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/'


def main():
    out = ROOT / 'data'
    out.mkdir(exist_ok=True)
    records = []
    for name in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        url = BASE + f'ETT-small/{name}.csv'
        path = out / f'{name}.csv'
        if not path.exists():
            with urllib.request.urlopen(url, timeout=90) as r:
                path.write_bytes(r.read())
        records.append(dict(dataset=name, source=url, bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            license='CC-BY-ND-4.0', modified=False))
        print(name, path.stat().st_size, flush=True)
    with urllib.request.urlopen(BASE + 'LICENSE', timeout=30) as r:
        (out / 'ETT_LICENSE.txt').write_bytes(r.read())
    (ROOT / 'workspace' / 'data_manifest.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
