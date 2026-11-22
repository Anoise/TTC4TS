"""Risk-regressor sweep over the number of shared-Delta adapter samples.

The sweep deliberately excludes augmentations and retrievals.  For each
``num_samples`` value it fits the supplied shared-residual input/output
adapters, builds the identity plus input-only, output-only and joint adapter
pool, and trains the same observable-feature Risk regressor on the purged
aggregator split.  Validation is used only for diagnostics; test labels are
saved for metrics and the explicitly marked oracle.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from ttc4ts.adapters import ADAPTER_VERSION, DEFAULT_MAX_DELTA, fit_adapters
from ttc4ts.aggregators import LearnedAggregator, errors, features
from ttc4ts.backbones import load_backbone, state_digest
from ttc4ts.data import load_dataset, split_origins, windows
from ttc4ts.generators import Generator
from ttc4ts.retrieval import RetrievalBank


def source_digest():
    h=hashlib.sha256()
    for p in sorted((ROOT/'ttc4ts').glob('*.py'))+[Path(__file__)]:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def metric(pred,target):
    e=pred-target
    return dict(mse=float(e.square().mean()), mae=float(e.abs().mean()),
                per_origin_mse=e.square().mean((1,2)).tolist(),
                per_origin_mae=e.abs().mean((1,2)).tolist())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='chronos-bolt-small')
    p.add_argument('--dataset',default='ETTh1')
    p.add_argument('--horizon',type=int,default=96)
    p.add_argument('--context',type=int,default=512)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--num-samples',type=int,required=True)
    p.add_argument('--max-delta',type=float,default=DEFAULT_MAX_DELTA)
    p.add_argument('--adapter-steps',type=int,default=32)
    p.add_argument('--aggregator-steps',type=int,default=160)
    p.add_argument('--fit-windows',type=int,default=96)
    p.add_argument('--limit-eval',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',default='outputs/adapter_sample_sweep_d03')
    a=p.parse_args()
    if a.num_samples < 2:
        raise ValueError('num-samples must include zero and at least one nonzero Delta')
    torch.set_num_threads(2); torch.manual_seed(a.seed); np.random.seed(a.seed)
    out=ROOT/a.output/f'{a.model}_{a.dataset}_h{a.horizon}_s{a.seed}_k{a.num_samples}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():
        previous=json.loads((out/'metrics.json').read_text())
        if previous['source_sha256']!=source_digest() or previous['config']['max_delta']!=a.max_delta:
            raise RuntimeError('Refusing stale adapter sweep; use a new output directory')
        print('ALREADY_COMPLETE',out); return
    data,meta=load_dataset(ROOT/'data'/f'{a.dataset}.csv')
    meta['period']=meta.get('period',24)
    splits=split_origins(meta,a.context,a.horizon,a.fit_windows)
    splits['aggregator']=splits['aggregator'][splits['aggregator']>=meta['adapter_end']+a.horizon]
    if a.limit_eval:
        for key in ['aggregator','validation','test']:
            splits[key]=splits[key][:a.limit_eval]
    deltas=np.linspace(0.,a.max_delta,a.num_samples).astype(float).tolist()
    model=load_backbone(a.model,ROOT/'checkpoints',a.device)
    before=state_digest(model)
    tx,ty=windows(data,splits['adapter'],a.context,a.horizon)
    adapter_batch=1 if a.horizon>=720 or 'base' in a.model else 2
    started=time.time()
    adapters,adapter_logs=fit_adapters(model,torch.tensor(tx,device=a.device),
        torch.tensor(ty,device=a.device),steps=a.adapter_steps,seed=a.seed,
        batch_size=adapter_batch,max_delta=a.max_delta,num_samples=a.num_samples)
    if state_digest(model)!=before:
        raise AssertionError('Frozen backbone changed')
    bank=RetrievalBank(torch.tensor(data,device=a.device),meta['adapter_end'],a.horizon)
    generator=Generator(model,adapters,bank,a.seed)
    names=['identity']+[s.name for s in generator.specs if s.family=='adapter']
    all_origins=np.array(sorted({int(o-r*a.horizon) for key in ['aggregator','validation','test']
                                  for o in splits[key] for r in [0,1]}))
    predictions=[]
    costs=[]
    for start in range(0,len(all_origins),a.batch_size):
        oo=all_origins[start:start+a.batch_size]
        x=torch.tensor(windows(data,oo,a.context,a.horizon)[0],device=a.device)
        sync(a.device); tick=time.perf_counter()
        pred,cost=generator.predict(x,torch.tensor(oo,device=a.device),a.horizon,names)
        sync(a.device); cost.update(seconds=time.perf_counter()-tick)
        predictions.append(pred.cpu()); costs.append(cost)
    cache=torch.cat(predictions)
    index={int(o):i for i,o in enumerate(all_origins)}
    def make_set(key):
        oo=splits[key]; ids=[index[int(o)] for o in oo]
        c=cache[ids]
        x,y=windows(data,oo,a.context,a.horizon)
        past=cache[[index[int(o-a.horizon)] for o in oo]]
        past_y=torch.tensor(np.stack([data[o-a.horizon:o] for o in oo]))
        hist=errors(past,past_y)
        f=features(torch.tensor(x),c,hist)
        return c,f,torch.tensor(y),oo
    ac,af,ay,_=make_set('aggregator')
    vc,vf,vy,_=make_set('validation')
    tc,tf,ty,_=make_set('test')
    results={'frozen':metric(tc[:,0],ty)}
    deployed={'frozen':tc[:,0].numpy()}
    selections={}
    family_names={
        'input':['identity']+[n for n in names if n.startswith('input_')],
        'output':['identity']+[n for n in names if n.startswith('output_')],
        'joint':['identity']+[n for n in names if n.startswith('joint_')],
        'all':names,
    }
    name_to_index={n:i for i,n in enumerate(names)}
    val_metrics={}
    for family,selected in family_names.items():
        ids=[name_to_index[n] for n in selected]
        torch.manual_seed(a.seed)
        agg=LearnedAggregator(len(ids),'risk')
        agg.fit(ac[:,ids],af[:,ids],ay,epochs=a.aggregator_steps,seed=a.seed)
        with torch.no_grad():
            pv=agg.forecast(vc[:,ids],vf[:,ids]); pt=agg.forecast(tc[:,ids],tf[:,ids])
        results[f'risk_{family}']=metric(pt,ty)
        deployed[f'risk_{family}']=pt.numpy()
        torch.save(agg.state_dict(),out/f'risk_{family}.pt')
        val_metrics[family]=metric(pv,vy)
        selections[family]=dict(candidate_count=len(ids), candidates=selected,
                                 validation_mse=val_metrics[family]['mse'])
        oracle=errors(tc[:,ids],ty).min(-1).values
        results[f'oracle_{family}']=dict(mse=float(oracle.mean()),
            mae=float((tc[:,ids][torch.arange(len(tc)),errors(tc[:,ids],ty).argmin(-1)]-ty).abs().mean()))
    # Risk is the requested primary aggregation; keep the best individual
    # candidate to separate denser sampling from combination effects.
    best=int(errors(vc,vy).mean(0).argmin())
    results['validation_best']=metric(tc[:,best],ty)
    deployed['validation_best']=tc[:,best].numpy()
    selections['validation_best']=names[best]
    final=dict(config=vars(a),dataset=meta,adapter_version=ADAPTER_VERSION,
               deltas=deltas,candidate_names=names,source_sha256=source_digest(),
               backbone_sha256=before,backbone_unchanged=True,
               adapter_parameters=sum(x.numel() for x in adapters.parameters()),
               adapter_logs=adapter_logs,cache_costs=costs,results=results,
               selections=selections,validation=val_metrics,
               elapsed_seconds=time.time()-started)
    torch.save(adapters.checkpoint(),out/'adapters.pt')
    np.savez_compressed(out/'predictions.npz',test_targets=ty.numpy(),
        test_candidates=tc.numpy(),validation_targets=vy.numpy(),
        validation_candidates=vc.numpy(),candidate_names=np.array(names),
        aggregator_candidates=ac.numpy(),aggregator_features=af.numpy(),aggregator_targets=ay.numpy(),
        test_features=tf.numpy(),validation_features=vf.numpy(),**deployed)
    (out/'metrics.json').write_text(json.dumps(final,indent=2,allow_nan=False))
    print(json.dumps({k:results[k]['mse'] for k in results if k.startswith(('risk_','frozen','validation_best'))},indent=2))


if __name__=='__main__':
    main()
