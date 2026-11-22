"""Validate distributable files and the included measured-result snapshots."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIRS = {'ttc4ts', 'scripts', 'tests', 'docs', 'examples', 'tools',
               'requirements', 'artifacts', 'paper', '.github', 'notebooks', 'experiments'}
PUBLIC_ROOT_FILES = {'README.md', 'pyproject.toml', 'LICENSE', 'NOTICE',
                    'THIRD_PARTY_NOTICES.md', 'CONTRIBUTING.md', 'MANIFEST.in',
                    '.gitignore', '.gitattributes'}
EXCLUDED_SUFFIXES = {'.pyc', '.pyo', '.aux', '.log', '.out', '.bbl', '.blg',
                     '.fls', '.fdb_latexmk', '.synctex.gz'}


def public_files():
    """Use Git exclusions in a checkout; a source archive uses the same allowlist."""
    candidates = None
    from_git = False
    try:
        top = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        if Path(top).resolve() == ROOT:
            raw = subprocess.check_output(['git', 'ls-files', '-z', '--cached',
                '--others', '--exclude-standard'], cwd=ROOT)
            candidates = {ROOT/name.decode('utf-8') for name in raw.split(b'\0') if name}
            from_git = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    if candidates is None:
        candidates = {p for p in ROOT.rglob('*') if p.is_file()}
    selected = []
    for p in sorted(candidates):
        rel = p.relative_to(ROOT)
        if rel.parts[0] not in PUBLIC_DIRS and rel.as_posix() not in PUBLIC_ROOT_FILES:
            # Unknown files in Git are errors, rather than silently packaged.
            if from_git or (len(rel.parts) == 1 and p.is_file()):
                raise ValueError(f'Unexpected public file: {rel}')
            continue
        if '__pycache__' in rel.parts or p.suffix in EXCLUDED_SUFFIXES:
            continue
        if rel.as_posix() == 'paper/results.facts.json' or any(x.startswith('render') for x in rel.parts[:-1]):
            continue
        if p.is_symlink():
            raise ValueError(f'Symlinks are not packaged: {rel}')
        if p.is_file():
            selected.append(p)
    return selected


def validate_artifacts():
    manifest = json.loads((ROOT/'artifacts/manifest.json').read_text())
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT/'artifacts').rglob('*')
              if p.is_file() and p.name not in {'README.md', 'manifest.json'}}
    if actual != set(manifest['files']):
        raise ValueError('Measured artifact file set does not match manifest')
    for name, record in manifest['files'].items():
        p = ROOT/name
        if not p.resolve().is_relative_to(ROOT/'artifacts'):
            raise ValueError('Artifact manifest path escapes artifacts/')
        data = p.read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError(f'Measured artifact checksum mismatch: {name}')
    sources = json.loads((ROOT/'artifacts/provenance/snapshot_sources.json').read_text())
    if set(sources) != actual-{'artifacts/provenance/snapshot_sources.json'}:
        raise ValueError('Artifact source mapping must cover each evidence file exactly once')

    models = ['chronos-bolt-small', 'chronos-bolt-base', 'timesfm-2.0-500m-pytorch',
              'timer-base-84m', 'sundial-base-128m', 'chronos-2']
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Appliances', 'BeijingWeather']
    expected = set(itertools.product(models, datasets, [96,192,336,720], [0,1,2]))
    def cell(row):
        return (row['model'], row['dataset'], int(row['horizon']), int(row['seed']))
    with gzip.open(ROOT/'artifacts/main/all_metrics.csv.gz', 'rt', newline='') as f:
        rows = list(csv.DictReader(f))
    seen = set()
    by_method = {}
    for row in rows:
        key = cell(row)
        if (key, row['method']) in seen:
            raise ValueError('Duplicate main method/cell')
        seen.add((key, row['method']))
        for metric in ['mse','mae','relative_mse','relative_mae']:
            if not math.isfinite(float(row[metric])) or float(row[metric]) <= 0:
                raise ValueError('Invalid measured metric')
        by_method.setdefault(row['method'], {})[key] = row
    for method, group in by_method.items():
        if set(group) != expected:
            raise ValueError(f'Incomplete main snapshot method: {method}')
    required = {'backbone','uniform','weighted','classifier','risk','ranking','greedy',
                'retrospective','validation_best','validation_selected','oracle'}
    if not required <= set(by_method):
        raise ValueError('Missing primary aggregation method')
    for row in rows:
        base = by_method['backbone'][cell(row)]
        for metric in ['mse','mae']:
            ratio = float(row[metric])/float(base[metric])
            if not math.isclose(ratio, float(row['relative_'+metric]), rel_tol=1e-10):
                raise ValueError('Relative metric is inconsistent with frozen backbone')
    selected = list(by_method['validation_selected'].values())
    gm = lambda values: math.exp(sum(math.log(v) for v in values)/len(values))
    summary = json.loads((ROOT/'artifacts/main/reports/analysis_summary.json').read_text())
    mse = gm([float(r['relative_mse']) for r in selected])
    if not math.isclose(mse, summary['selected_relative_mse'], rel_tol=1e-10):
        raise ValueError('Main summary does not match cell metrics')

    with (ROOT/'artifacts/adapter_samples/adapter_sample_results.csv').open(newline='') as f:
        sweep = list(csv.DictReader(f))
    samples = [5,10,15,20,25,30,35,40]
    expected_sweep = set(itertools.product(models[:3], ['ETTh1','ETTm1','Appliances'],
                                          [96], [0], samples))
    if {(*cell(r), int(r['num_samples'])) for r in sweep} != expected_sweep:
        raise ValueError('Adapter sweep coverage mismatch')
    if len({(*cell(r),r['num_samples'],r['method']) for r in sweep}) != len(sweep):
        raise ValueError('Duplicate sweep method/cell')
    with (ROOT/'artifacts/retrieval/retrieval_results.csv').open(newline='') as f:
        retrieval = list(csv.DictReader(f))
    if {cell(r) for r in retrieval} != expected:
        raise ValueError('Retrieval audit coverage mismatch')
    if len({(*cell(r),r['method']) for r in retrieval}) != len(retrieval):
        raise ValueError('Duplicate retrieval method/cell')
    return dict(main_cells=len(expected), adapter_cells=len(expected_sweep),
                retrieval_cells=len(expected), selected_relative_mse=mse,
                artifact_files=len(manifest['files']))


def validate_public_files():
    patterns = {
        'private key': r'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----',
        'access token': r'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,})\b',
        'literal credential': r'(?i)\b(?:password|passwd|api_key|access_token)\s*[:=]\s*[\"\x27][^\"\x27\s]{6,}[\"\x27]',
        'private machine path': r'(?:[A-Za-z]:[\\/](?:Users|Software)[\\/]|/ho' + r'me/[^/\s]+/)',
        'IP address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    }
    files = public_files()
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > 25*1024*1024:
            raise ValueError(f'Unexpectedly large public file: {rel}')
        if path.suffix in {'.pt','.pth','.ckpt','.safetensors','.npz','.arrow','.pem','.key'}:
            raise ValueError(f'Private/downloaded binary in source tree: {rel}')
        if path.suffix in {'.pdf','.png'}:
            continue
        data = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError:
            raise ValueError(f'Unexpected binary file: {rel}')
        if path.suffix == '.ipynb':
            # Scan human-readable sources/outputs, not base64 PNG bytes.
            notebook = json.loads(text)
            chunks = [json.dumps(notebook.get('metadata', {}))]
            for cell in notebook['cells']:
                chunks.append(''.join(cell['source']))
                for output in cell.get('outputs', []):
                    chunks.append(''.join(output.get('text', [])))
                    for kind, value in output.get('data', {}).items():
                        if kind.startswith('text/'):
                            chunks.append(''.join(value))
            text = '\n'.join(chunks)
        for name, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                line = text.count('\n', 0, match.start())+1
                raise ValueError(f'Possible {name}: {rel}:{line} (value omitted)')
    return dict(public_files=len(files), bytes=sum(p.stat().st_size for p in files))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(dict(files=validate_public_files(), artifacts=validate_artifacts(),
                         supplementary=validate_supplementary(), notebooks=validate_notebooks()), indent=2))


def validate_supplementary():
    """Independent coverage/ratio/summary checks over all released study grids."""
    import sys
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import pandas as pd
    from ttc4ts.results import load_snapshot, MODELS, METHODS, uncertainty_summary
    from ttc4ts.results import lw_budget_summary, summarize
    datasets = ['ETTh1','ETTh2','ETTm1','ETTm2','Appliances','BeijingWeather']
    def require(condition, message):
        if not condition: raise ValueError(message)
    def grid(frame, columns, dimensions, title):
        actual = set(frame[columns].itertuples(index=False, name=None))
        expected = set(itertools.product(*dimensions))
        require(not frame.duplicated(columns).any() and actual == expected, title+' grid mismatch')
    def ratios(frame, keys, rule, baseline):
        base = frame[frame[rule]==baseline][keys+['mse','mae']]
        merged = frame.merge(base, on=keys, suffixes=('', '_base'), validate='many_to_one')
        require(len(merged)==len(frame), 'Missing frozen reference')
        for metric in ['mse','mae']:
            require(np.isfinite(merged[metric]).all() and (merged[metric]>0).all(), 'Invalid loss')
            require(np.allclose(merged[metric]/merged[metric+'_base'], merged['relative_'+metric], rtol=1e-6), 'Relative error mismatch')
    sweep = load_snapshot('adapters', ROOT)
    grid(sweep, ['model','dataset','horizon','seed','num_samples','method'],
         [MODELS[:3], ['ETTh1','ETTm1','Appliances'],[96],[0],[5,10,15,20,25,30,35,40],
          ['risk_all','risk_input','risk_output','risk_joint','oracle_all','validation_best']], 'Adapter')
    retrieval = load_snapshot('retrieval', ROOT)
    grid(retrieval, ['model','dataset','horizon','seed','method'],
         [MODELS,datasets,[96,192,336,720],[0,1,2],['uniform_demo','risk_retrieval','validation_best_retrieval']], 'Retrieval')
    baselines = load_snapshot('baselines', ROOT)
    require(baselines.method.nunique()==17, 'Baseline method count')
    grid(baselines, ['model','dataset','horizon','seed','method'],
         [MODELS,datasets,[96,192,336,720],[0,1,2],baselines.method.unique()], 'Baselines')
    ratios(baselines, ['cell'], 'method', 'frozen')
    budgets = [1,2,4,8,16,80]
    lw = load_snapshot('lw_budget', ROOT)
    grid(lw, ['model','dataset','seed','split','rule','K'],
         [MODELS,['electricity','national_illness','exchange_rate'],[0,1,2],['validation','test'],['frozen','uniform','weighted'],budgets], 'Learned-weights budget')
    ratios(lw, ['cell','split','K'], 'rule', 'frozen')
    manifests = json.loads((ROOT/'artifacts/provenance/lw_dataset_manifest.json').read_text())
    for dataset, channel, context, horizon in [('electricity',321,512,96),('national_illness',7,104,24),('exchange_rate',8,512,96)]:
        rows = lw[lw.dataset==dataset]
        require(set(rows.channels)=={channel} and set(rows.context)=={context} and set(rows.horizon)=={horizon}, 'Dataset shape/protocol mismatch')
        require(set(rows.dataset_sha256)=={manifests[dataset]['metadata']['sha256']}, 'Dataset version mismatch')
    require(np.allclose(lw[lw.K==1].relative_mse, 1), 'K1 must be identity')
    old = load_snapshot('single_channel', ROOT)
    grid(old, ['model','dataset','seed','rule','K'],
         [MODELS,['traffic','electricity','exchange_rate'],[0,1,2],['frozen']+[m for m in METHODS if m!='backbone'],budgets], 'Single-channel budget')
    require(set(old.channels)=={1} and set(old.horizon)=={96}, 'Historical single-channel scope mismatch')
    ratios(old, ['model','dataset','seed','K'], 'rule', 'frozen')
    uq = load_snapshot('uncertainty', ROOT)
    require(uq.method.nunique()==13, 'Uncertainty method count')
    grid(uq, ['model','dataset','seed','method','alpha'],
         [MODELS,datasets,[0,1,2],uq.method.unique(),[.2,.1,.05]], 'Uncertainty')
    require(np.isfinite(uq[['coverage','width','wis','relative_wis']]).all().all(), 'Invalid UQ metric')
    require(uq.coverage.between(0,1).all() and (uq.width>=0).all(), 'Invalid interval metric bounds')
    facts = json.loads((ROOT/'artifacts/uncertainty/summary.json').read_text())['facts']
    for row in uncertainty_summary(ROOT).itertuples():
        require(math.isclose(row.coverage90,facts[row.method]['coverage90'],rel_tol=1e-9), 'UQ coverage summary mismatch')
        require(math.isclose(row.relative_wis,facts[row.method]['relative_wis'],rel_tol=1e-9), 'UQ WIS summary mismatch')
    example_manifest = json.loads((ROOT/'artifacts/examples/forecast_manifest.json').read_text())
    for entry in example_manifest['examples']:
        ex = pd.read_csv(ROOT/f"artifacts/examples/forecast_{entry['number']}.csv")
        for name in ['frozen','ttc4ts']:
            require(np.isclose(np.mean((ex[name]-ex.target)**2), entry[name+'_mse'], rtol=1e-6), 'Forecast excerpt MSE mismatch')
        interval = pd.read_csv(ROOT/f"artifacts/examples/interval_{entry['number']}.csv")
        require(len(interval)==96 and np.isfinite(interval).all().all(), 'Invalid interval excerpt')
        for name in interval:
            if name.endswith('_lower'):
                require((interval[name]<=interval[name[:-6]+'_upper']).all(), 'Reversed interval endpoints')
    return dict(baseline_records=len(baselines), lw_budget_records=len(lw), single_channel_records=len(old),
                uncertainty_records=len(uq), verified_forecast_examples=6)


def validate_notebooks():
    paths = sorted((ROOT/'notebooks').glob('*.ipynb'))
    if len(paths)!=7: raise ValueError('Expected seven executed notebooks')
    codes = figures = 0
    for path in paths:
        nb = json.loads(path.read_text(encoding='utf-8'))
        if nb['nbformat']!=4: raise ValueError('Unsupported notebook format')
        for cell in nb['cells']:
            if cell['cell_type']!='code': continue
            codes += 1
            if cell['execution_count'] is None: raise ValueError('Unexecuted notebook: '+path.name)
            for output in cell.get('outputs', []):
                if output['output_type']=='error': raise ValueError('Notebook error output: '+path.name)
                figures += int('image/png' in output.get('data', {}))
    if figures<7: raise ValueError('Expected embedded measured plots')
    return dict(notebooks=len(paths), executed_code_cells=codes, embedded_plots=figures)


if __name__ == '__main__':
    main()
