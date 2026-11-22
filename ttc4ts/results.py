"""Read and summarize the released measured snapshots without model downloads.

These utilities never fit models or invent missing records. Geometric means apply
to per-cell ratios, not to raw losses or confidence interval endpoints.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

MODELS = ['chronos-bolt-small', 'chronos-bolt-base', 'timesfm-2.0-500m-pytorch',
          'timer-base-84m', 'sundial-base-128m', 'chronos-2']
MODEL_LABELS = dict(zip(MODELS, ['Bolt-S', 'Bolt-B', 'TimesFM', 'Timer', 'Sundial', 'Chronos-2']))
METHODS = ['backbone', 'uniform', 'weighted', 'classifier', 'risk', 'ranking',
           'greedy', 'retrospective', 'validation_best', 'validation_selected', 'oracle']
METHOD_LABELS = dict(zip(METHODS, ['Frozen', 'Uniform', 'Learned weights', 'Classifier',
    'Risk regressor', 'Pairwise ranker', 'Greedy ensemble', 'Retrospective',
    'Validation-best candidate', 'Validation-selected TTC4TS', 'Oracle (unavailable)']))
SNAPSHOTS = {
    'main': 'main/all_metrics.csv.gz',
    'adapters': 'adapter_samples/adapter_sample_results.csv',
    'retrieval': 'retrieval/retrieval_results.csv',
    'baselines': 'baselines/all_metrics.csv',
    'lw_budget': 'lw_budget/budget_metrics.csv',
    'single_channel': 'single_channel_budget/budget_metrics.csv',
    'uncertainty': 'uncertainty/all_metrics.csv',
}


def find_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    for path in [start.resolve(), *start.resolve().parents]:
        if (path/'artifacts/manifest.json').is_file() and (path/'pyproject.toml').is_file():
            return path
    raise FileNotFoundError('Run from the TTC4TS repository or its notebooks directory.')


def checked_path(relative, root=None):
    root = find_root() if root is None else Path(root).resolve()
    path = (root/'artifacts'/relative).resolve()
    if not path.is_relative_to(root/'artifacts'):
        raise ValueError('Snapshot path must stay inside artifacts/')
    manifest = json.loads((root/'artifacts/manifest.json').read_text(encoding='utf-8'))
    record = manifest['files'][path.relative_to(root).as_posix()]
    data = path.read_bytes()
    if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError(f'Snapshot checksum mismatch: {relative}')
    return path


def load_snapshot(study, root=None):
    relative = SNAPSHOTS[study]
    root = find_root() if root is None else Path(root).resolve()
    # Larger CSVs can be losslessly compressed in source distributions.
    if not (root/'artifacts'/relative).exists() and (root/'artifacts'/(relative+'.gz')).exists():
        relative += '.gz'
    return pd.read_csv(checked_path(relative, root))


def geometric(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Geometric means require a nonempty finite positive sample.')
    return float(np.exp(np.log(values).mean()))


def summarize(frame, by='method'):
    """Equal-cell geometric relative errors; wins count cells, not significance."""
    return frame.groupby(by, sort=False).agg(
        relative_mse=('relative_mse', geometric),
        relative_mae=('relative_mae', geometric),
        wins=('relative_mse', lambda x: int((x < 1-1e-8).sum())),
        cells=('relative_mse', 'size')).reset_index()


def main_summary(root=None):
    data = load_snapshot('main', root)
    result = summarize(data[data.method.isin(METHODS)])
    return result.set_index('method').loc[METHODS].reset_index()


def lw_budget_summary(root=None):
    data = load_snapshot('lw_budget', root)
    data = data[(data.split == 'test') & (data.rule == 'weighted')]
    return summarize(data, ['dataset', 'K'])


def uncertainty_summary(root=None):
    data = load_snapshot('uncertainty', root)
    # WIS repeats across alpha rows; take one row per cell/method, not three.
    selected = data[np.isclose(data.alpha, .1)]
    if selected.duplicated(['cell', 'method']).any():
        raise ValueError('Duplicate 90% interval record')
    return selected.groupby('method').agg(coverage90=('coverage', 'mean'),
        width90=('width', 'mean'), relative_wis=('relative_wis', geometric),
        cells=('cell', 'size')).reset_index()
