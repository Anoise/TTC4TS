"""Fixed Learned-weights TTC4TS: equal fitting steps at each nested budget.

No validation-based aggregator selection and no retrospective candidate cache.
Each budget's full fitted state and every current candidate are saved for audit.
Core generators, shared-delta adapters and point decoders remain unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import types
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if '--model' in sys.argv and sys.argv[sys.argv.index('--model')+1] == 'chronos-2':
    sys.path.insert(0, str(ROOT/'.runtime/chronos2'))
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from lw_budget_data import load_lw_dataset, study_splits, nested_order, PROTOCOL, BUDGETS
from extra_backbones import load_extra
from ttc4ts.backbones import load_backbone, state_digest
from ttc4ts.data import windows
from ttc4ts.adapters import fit_adapters, AdapterSuite, ADAPTER_VERSION
from ttc4ts.generators import Generator, GENERATOR_VERSION, DEFAULT_RETRIEVAL_TOP_K
from ttc4ts.retrieval import RetrievalBank
from ttc4ts.aggregators import LearnedAggregator, AGGREGATOR_VERSION

VERSION = 'multivariate-fixed-lw-budget-v1'


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)


def source_hashes():
    paths = sorted((ROOT/'ttc4ts').glob('*.py')) + [Path(__file__), ROOT/'scripts/extra_backbones.py',
                                                                ROOT/'scripts/lw_budget_data.py']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def checkpointed_predict(self, x, horizon):
    """Exact recomputation through frozen decoder chunks; all channels retained."""
    b, t, c = x.shape
    pieces = []
    for part in x.transpose(1, 2).reshape(b*c, t).split(self.batch_size):
        if torch.is_grad_enabled() and part.requires_grad:
            pieces.append(checkpoint(self._predict_flat, part, horizon, use_reentrant=False))
        else:
            pieces.append(self._predict_flat(part, horizon))
    return torch.cat(pieces).reshape(b, c, horizon).transpose(1, 2)


def candidate_cache(generator, data, origins, a, out):
    path, progress = out/'candidates.npy', out/'candidate_progress.json'
    shape = (len(origins), 80, a.horizon, data.shape[1])
    done, costs = 0, []
    if progress.exists():
        saved = json.loads(progress.read_text())
        assert saved['origins'] == origins.tolist()
        done, costs = saved['completed'], saved['costs']
        cache = np.lib.format.open_memmap(path, mode='r+')
        assert cache.shape == shape and cache.dtype == np.float32
    else:
        cache = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
    for start in range(done, len(origins)):
        oo = origins[start:start+1]
        x = torch.tensor(windows(data, oo, a.context, a.horizon)[0], device=a.device)
        if a.device.startswith('cuda'): torch.cuda.synchronize()
        tick = time.perf_counter()
        pred, cost = generator.predict(x, torch.tensor(oo, device=a.device), a.horizon)
        if a.device.startswith('cuda'): torch.cuda.synchronize()
        cache[start] = pred[0].cpu().numpy()
        cache.flush()
        cost.update(seconds=time.perf_counter()-tick, origins=1)
        costs.append(cost)
        atomic_json(progress, dict(origins=origins.tolist(), completed=start+1, costs=costs))
        if start % 10 == 0:
            print(json.dumps(dict(stage='candidates', completed=start+1, total=len(origins))), flush=True)
    return cache, costs


def metrics(pred, target):
    e = pred.astype(np.float64)-target.astype(np.float64)
    return dict(mse=float(np.square(e).mean()), mae=float(np.abs(e).mean()),
                per_origin_mse=np.square(e).mean((1, 2)).tolist(),
                per_origin_mae=np.abs(e).mean((1, 2)).tolist())


def fit_budget(candidates, targets, ids, steps, seed, device):
    """Use the existing softmax-bias aggregator; features are unused in this mode."""
    torch.manual_seed(seed)
    c = torch.tensor(candidates[:, ids], device=device)
    y = torch.tensor(targets, device=device)
    features = torch.zeros(c.shape[0], len(ids), 8, device=device)
    agg = LearnedAggregator(len(ids), mode='weighted').to(device)
    logs = agg.fit(c, features, y, epochs=steps, seed=seed)
    agg = agg.cpu()
    return agg, logs


def evaluate(cache, origins, data, splits, names, a, out):
    index = {int(o): i for i, o in enumerate(origins)}
    sets, arrays, result, fitted, predictions = {}, {}, {}, {}, {}
    for split in ['aggregator', 'validation', 'test']:
        ids = np.array([index[int(o)] for o in splits[split]])
        target = np.stack([data[o:o+a.horizon] for o in splits[split]])
        sets[split] = (ids, target)
        arrays[split+'_origins'] = splits[split]
        arrays[split+'_indices'] = ids
        arrays[split+'_targets'] = target
    order = nested_order(names)
    for k in BUDGETS:
        ids = order[:k]
        fit_ids, fit_y = sets['aggregator']
        agg, logs = fit_budget(cache[fit_ids], fit_y, ids, a.aggregator_steps, a.seed, a.device)
        torch.save(dict(state_dict=agg.state_dict(), candidate_ids=ids, steps=a.aggregator_steps,
                        seed=a.seed, mode='weighted'), out/f'weighted_k{k}.pt')
        w = agg.bias.detach().softmax(-1).numpy()
        fitted[str(k)] = dict(candidate_ids=ids, weights=w.tolist(), loss=logs,
                             trainable_effective_parameters=k, steps=a.aggregator_steps)
        for split in ['validation', 'test']:
            row_ids, target = sets[split]
            # Match the core aggregation's float32 multiplication/sum.
            frozen, uniform, weighted = [], [], []
            for row in row_ids:
                c = torch.from_numpy(np.array(cache[row]))
                frozen.append(c[0].numpy())
                uniform.append(c[ids].mean(0).numpy())
                weighted.append((c[ids]*torch.from_numpy(w)[:, None, None]).sum(0).numpy())
            for rule, values in [('frozen', frozen), ('uniform', uniform), ('weighted', weighted)]:
                key = f'{split}/{rule}_{k}'
                predictions[key] = np.stack(values)
                result[key] = metrics(predictions[key], target)
        print(json.dumps(dict(stage='weighted', k=k, test_mse=result[f'test/weighted_{k}']['mse'])), flush=True)
    arrays.update(origins=origins, candidate_names=np.array(names), budget_order=np.array(order))
    np.savez_compressed(out/'evidence.npz', **arrays)
    np.savez_compressed(out/'predictions.npz', **predictions)
    return result, fitted


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', choices=list(PROTOCOL), required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--adapter-steps', type=int, default=32)
    p.add_argument('--aggregator-steps', type=int, default=160)
    p.add_argument('--fit-windows', type=int, default=96)
    p.add_argument('--limit-eval', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', default='results_lw_budget_v1')
    a = p.parse_args()
    a.context, a.horizon = PROTOCOL[a.dataset]
    if a.limit_eval and 'smoke' not in a.output: raise ValueError('Limited runs must be marked smoke')
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = ROOT/a.output/f'{a.model}_{a.dataset}_h{a.horizon}_s{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    data, meta = load_lw_dataset(ROOT/'data'/f'{a.dataset}.csv')
    hashes = source_hashes()
    if (out/'metrics.json').exists():
        old = json.loads((out/'metrics.json').read_text())
        assert old['code_sha256'] == hashes and old['config'] == vars(a) and old['dataset'] == meta
        print('ALREADY_COMPLETE', out.name, flush=True)
        return
    splits = study_splits(meta, a.context, a.horizon, a.fit_windows, a.limit_eval)
    model = (load_backbone(a.model, ROOT/'checkpoints', a.device) if
             a.model.startswith(('chronos-bolt', 'timesfm')) or a.model == 'naive'
             else load_extra(a.model, ROOT/'checkpoints', a.device))
    model.predict = types.MethodType(checkpointed_predict, model)
    before = state_digest(model)
    guard = dict(code_sha256=hashes, config=vars(a), backbone_sha256=before, dataset_sha256=meta['sha256'])
    if a.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    print(json.dumps(dict(stage='adapter', cell=out.name, channels=meta['channels'],
                         origins={k: len(v) for k, v in splits.items()})), flush=True)
    if (out/'training.json').exists():
        saved = json.loads((out/'training.json').read_text())
        assert saved['guard'] == guard, 'Stale adapter/cache cannot be resumed'
        adapters = AdapterSuite.from_checkpoint(torch.load(out/'adapters.pt', map_location='cpu', weights_only=False), a.device)
        logs, training_seconds = saved['logs'], saved['seconds']
    else:
        x, y = windows(data, splits['adapter'], a.context, a.horizon)
        start = time.time()
        adapters, logs = fit_adapters(model, torch.tensor(x, device=a.device), torch.tensor(y, device=a.device),
                                     steps=a.adapter_steps, seed=a.seed, batch_size=1)
        training_seconds = time.time()-start
        assert before == state_digest(model), 'Frozen parameters changed'
        torch.save(adapters.checkpoint(), out/'adapters.pt')
        atomic_json(out/'training.json', dict(guard=guard, logs=logs, seconds=training_seconds))
        del x, y
    bank = RetrievalBank(torch.tensor(data, device=a.device), meta['adapter_end'], a.horizon)
    generator = Generator(model, adapters, bank, a.seed)
    names = [s.name for s in generator.specs]
    origins = np.array(sorted({int(o) for split in ['aggregator', 'validation', 'test'] for o in splits[split]}))
    cache, costs = candidate_cache(generator, data, origins, a, out)
    assert before == state_digest(model), 'Frozen parameters changed during generation'
    peak = torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else 0
    del generator, bank, adapters, model
    if a.device.startswith('cuda'): torch.cuda.empty_cache()
    result, fitted = evaluate(cache, origins, data, splits, names, a, out)
    atomic_json(out/'metrics.json', dict(version=VERSION, config=vars(a), dataset=meta,
        code_sha256=hashes, backbone_sha256=before, backbone_unchanged=True,
        adapter_version=ADAPTER_VERSION, generator_version=GENERATOR_VERSION, aggregator_version=AGGREGATOR_VERSION,
        retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K, adapter_max_delta=.3, adapter_num_samples=20,
        adapter_batch_size=1, activation_checkpointing=True, adapter_training_seconds=training_seconds,
        adapter_training_loss=logs, candidate_costs=costs, peak_gpu_bytes=peak,
        elapsed_seconds=time.time()-tick, results=result, aggregator_fit=fitted,
        origins={k: v.tolist() for k, v in splits.items()},
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
            device=torch.cuda.get_device_name() if a.device.startswith('cuda') else 'cpu')))
    print(json.dumps(dict(stage='complete', cell=out.name, seconds=time.time()-tick)), flush=True)


if __name__ == '__main__': main()
