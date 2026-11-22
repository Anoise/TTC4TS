"""Verify exact historical dataset bytes before attempting a GPU rerun."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttc4ts.results import checked_path


def verify(study, directory):
    if study == 'main':
        records = json.loads(checked_path('provenance/data_manifest.json', ROOT).read_text())
        records += json.loads(checked_path('provenance/uci_manifest.json', ROOT).read_text())
        expected = {x['dataset']: x.get('processed_sha256', x.get('sha256')) for x in records}
        runs = json.loads(checked_path('provenance/main_runs.json', ROOT).read_text())
        for record in runs:
            dataset, digest = record['dataset'], record['dataset_sha256']
            if expected.get(dataset) not in (None, digest):
                raise ValueError('Conflicting historical dataset checksums')
            expected[dataset] = digest
    else:
        name = 'lw_dataset_manifest.json' if study == 'lw_budget' else 'single_channel_dataset_manifest.json'
        records = json.loads(checked_path('provenance/'+name, ROOT).read_text())
        expected = {k: v['metadata']['sha256'] for k, v in records.items()}
    report = []
    for name, digest in expected.items():
        path = Path(directory)/(name+'.csv')
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        report.append(dict(dataset=name, status='matched' if actual == digest and digest else 'missing' if actual is None else 'mismatch',
                           expected_sha256=digest, actual_sha256=actual))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', choices=['main', 'lw_budget', 'single_channel'], required=True)
    p.add_argument('--data', type=Path, default=ROOT/'data')
    a = p.parse_args()
    report = verify(a.study, a.data)
    print(json.dumps(report, indent=2))
    if any(x['status'] != 'matched' for x in report):
        raise SystemExit('Dataset verification failed. Do not substitute another CSV version.')
