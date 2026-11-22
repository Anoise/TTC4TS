"""Chronological origins: target blocks are disjoint across fitting splits."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_dataset(path):
    path = Path(path)
    frame = pd.read_csv(path)
    raw = frame.drop(columns=['date']).to_numpy(dtype=np.float32)
    if path.with_suffix('.json').exists():
        meta = json.loads(path.with_suffix('.json').read_text())
    else:
        factor = 4 if path.stem.startswith('ETTm') else 1
        meta = dict(train_end=8640*factor, val_end=11520*factor, test_end=14400*factor,
                    period=24*factor)
    if len(raw) < meta['test_end'] or not np.isfinite(raw).all():
        raise ValueError('Incomplete or nonfinite dataset')
    # Fit scaler only on the earliest adapter-fitting period.
    meta['adapter_end'] = int(meta['train_end'] * .6)
    mean = raw[:meta['adapter_end']].mean(0, dtype=np.float64)
    std = raw[:meta['adapter_end']].std(0, dtype=np.float64).clip(1e-6)
    meta.update(mean=mean.tolist(), std=std.tolist(), columns=list(frame.columns[1:]),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return ((raw - mean)/std).astype(np.float32), meta


def split_origins(meta, context, horizon, max_fit=192):
    def bounded(start, end, stride, limit=None):
        origins = np.arange(start, end-horizon+1, stride, dtype=np.int64)
        if not len(origins):
            raise ValueError(f'Empty split [{start},{end}) for horizon {horizon}')
        if limit and len(origins)>limit:
            origins = origins[np.linspace(0,len(origins)-1,limit).astype(int)]
        return origins
    return dict(
        adapter=bounded(context, meta['adapter_end'], max(8,horizon//8), max_fit),
        aggregator=bounded(meta['adapter_end'], meta['train_end'], max(8,horizon//4), max_fit),
        validation=bounded(meta['train_end'], meta['val_end'], horizon),
        test=bounded(meta['val_end'], meta['test_end'], horizon))


def windows(data, origins, context, horizon):
    return (np.stack([data[o-context:o] for o in origins]),
            np.stack([data[o:o+horizon] for o in origins]))
