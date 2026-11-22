"""Matched standalone baselines around saved Learned-weights TTC4TS results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from ttc4ts.baselines import BASELINE_VERSION, CHECKPOINT_STEPS, fit_plain, raw_demo_forecasts
from ttc4ts.adapters import AdapterSuite
from ttc4ts.aggregators import LearnedAggregator
from ttc4ts.backbones import load_backbone,state_digest
from ttc4ts.data import load_dataset,split_origins,windows
from ttc4ts.generators import Generator
from ttc4ts.retrieval import RetrievalBank


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def code_hashes():
    files=[Path(__file__),ROOT/'scripts/extra_backbones.py']
    files += [ROOT/'ttc4ts'/f'{n}.py' for n in ['baselines','adapters','aggregators','backbones','data','generators','retrieval']]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}


def metric(p,y):
    e=np.asarray(p,dtype=np.float64)-np.asarray(y,dtype=np.float64)
    return dict(mse=float(np.mean(e**2)),mae=float(np.mean(np.abs(e))),
                per_origin_mse=np.mean(e**2,axis=(1,2)).tolist(),
                per_origin_mae=np.mean(np.abs(e),axis=(1,2)).tolist())


def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()


def main():
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True)
    p.add_argument('--output',default='results_baselines_lw_v1')
    p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    if a.smoke and a.output=='results_baselines_lw_v1':raise ValueError('Separate smoke output required')
    out=ROOT/a.output/a.cell;out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results_revision_d03_demo10'/a.cell
    hashes=code_hashes()
    if (out/'metrics.json').exists():
        r=json.loads((out/'metrics.json').read_text())
        if r['code_sha256']!=hashes or r['smoke']!=a.smoke:raise RuntimeError('Stale completed baseline cell')
        print('ALREADY_COMPLETE',a.cell,flush=True);return
    start=time.time();original=json.loads((source/'metrics.json').read_text());cfg=original['config']
    h,seed,context=cfg['horizon'],cfg['seed'],cfg['context']
    if original['adapter_config']['max_delta']!=.3 or original['retrieval_top_k']!=10:raise ValueError('Wrong source protocol')
    data,meta=load_dataset(ROOT/'data'/f"{cfg['dataset']}.csv")
    assert meta==original['dataset']
    torch.manual_seed(seed);np.random.seed(seed)
    blocks={};origins={};xs={};ys={}
    with np.load(source/'evidence.npz',allow_pickle=False) as e:
        names=e['candidate_names'].tolist()
        for split in ['aggregator','validation','test']:
            oo=e[split+'_origins'];origins[split]=oo
            x,y=windows(data,oo,context,h)
            np.testing.assert_array_equal(y,e[split+'_targets'])
            xs[split],ys[split]=x,y
            blocks[split]=torch.from_numpy(e[split+'_candidates']).float()
    assert len(names)==80
    if a.smoke:
        for s in blocks:blocks[s]=blocks[s][:2];origins[s]=origins[s][:2];xs[s]=xs[s][:2];ys[s]=ys[s][:2]
    predictions={s:{} for s in ['validation','test']};fits={};choices={};costs={}
    for s in predictions:
        predictions[s]['frozen']=blocks[s][:,0].numpy()
        predictions[s]['ttc_uniform']=blocks[s].mean(1).numpy()
    weighted=torch.load(source/'weighted.pt',map_location='cpu',weights_only=True)
    w=weighted['bias'].softmax(-1)
    for s in predictions:predictions[s]['ttc_weighted']=(blocks[s]*w[None,:,None,None]).sum(1).numpy()
    with np.load(source/'predictions.npz') as old:
        np.testing.assert_allclose(predictions['test']['ttc_weighted'],old['weighted'][:len(ys['test'])],rtol=1e-6,atol=1e-6)
    subset_specs={
        'adapters_weighted':[i for i,n in enumerate(names) if i==0 or n.startswith(('input_','output_','joint_'))],
        'augmentations_weighted':[i for i,n in enumerate(names) if i==0 or (not n.startswith(('input_','output_','joint_','retrieval_')))],
        'retrieval_weighted':[i for i,n in enumerate(names) if i==0 or n.startswith('retrieval_demo')]}
    for method,ids in subset_specs.items():
        torch.manual_seed(seed)
        agg=LearnedAggregator(len(ids),'weighted')
        features=torch.zeros((len(blocks['aggregator']),len(ids),8))
        log=agg.fit(blocks['aggregator'][:,ids],features,torch.tensor(ys['aggregator']),
                    epochs=2 if a.smoke else 160,seed=seed)
        weights=agg.bias.detach().softmax(-1)
        for s in predictions:predictions[s][method]=(blocks[s][:,ids]*weights[None,:,None,None]).sum(1).numpy()
        torch.save(agg.state_dict(),out/f'{method}.pt');fits[method]=dict(ids=ids,weights=weights.tolist(),loss=log)
    demo_ids=[i for i,n in enumerate(names) if n.startswith('retrieval_demo')]
    adapter_ids=subset_specs['adapters_weighted']
    av=((blocks['validation'][:,adapter_ids]-torch.tensor(ys['validation'])[:,None])**2).mean((0,2,3))
    best_id=adapter_ids[int(av.argmin())];choices['shared_adapter_best']=names[best_id]
    for s in predictions:
        predictions[s]['retrieval_aligned']=blocks[s][:,demo_ids].mean(1).numpy()
        predictions[s]['retrieval_aligned_identity']=blocks[s][:,[0]+demo_ids].mean(1).numpy()
        predictions[s]['shared_adapter_best']=blocks[s][:,best_id].numpy()
        for mode in ['input','output','joint']:
            predictions[s]['shared_'+mode]=blocks[s][:,names.index(mode+'_0.3')].numpy()
    if cfg['model'] in ['timer-base-84m','sundial-base-128m','chronos-2']:
        if cfg['model']=='chronos-2':sys.path.insert(0,str(ROOT/'.runtime/chronos2'))
        from extra_backbones import load_extra
        model=load_extra(cfg['model'],ROOT/'checkpoints',a.device)
    else:model=load_backbone(cfg['model'],ROOT/'checkpoints',a.device)
    before=state_digest(model);assert before==original['backbone_sha256']
    bx=torch.tensor(xs['test'][:1],device=a.device)
    # Reproduce the original union-cache batch, including the query's position.
    # Different flattened GEMM batch shapes can change long-rollout rounding.
    with np.load(source/'evidence.npz',allow_pickle=False) as evidence:
        cache_origins=np.unique(np.concatenate([evidence[s+'_origins']-shift*h
            for s in ['aggregator','validation','test'] for shift in [0,1]]))
    index=int(np.searchsorted(cache_origins,origins['test'][0]))
    start=index//cfg['batch_size']*cfg['batch_size']
    check_x,_=windows(data,cache_origins[start:start+cfg['batch_size']],context,h)
    with torch.no_grad():fresh=model.predict(torch.tensor(check_x,device=a.device),h).cpu().numpy()
    fresh=fresh[index-start:index-start+1]
    np.testing.assert_allclose(fresh,predictions['test']['frozen'][:1],rtol=1e-4,atol=1e-5)
    splits=split_origins(meta,context,h,cfg['fit_windows'])
    training_origins=np.unique(np.concatenate([splits['adapter'],origins['aggregator']]))
    assert training_origins[-1]+h<=meta['train_end']
    if a.smoke:training_origins=training_origins[:4]
    trainx,trainy=windows(data,training_origins,context,h)
    tx=torch.tensor(trainx,device=a.device);ty=torch.tensor(trainy,device=a.device)
    batch=1 if h>=720 or 'base' in cfg['model'] else 2
    checkpoints=(1,2) if a.smoke else CHECKPOINT_STEPS
    adapters={}
    @torch.no_grad()
    def predict_adapter(adapter,split):
        result=[]
        for x in torch.tensor(xs[split],device=a.device).split(cfg['batch_size']):
            result.append(adapter(model,x).cpu())
        return torch.cat(result).numpy()
    for mode in ['input','output','joint']:
        print(json.dumps(dict(stage='fit',cell=a.cell,mode=mode)),flush=True)
        sync(a.device);begin=time.perf_counter()
        adapter,snapshots,log=fit_plain(model,tx,ty,mode,seed,batch,checkpoints=checkpoints)
        sync(a.device);seconds=time.perf_counter()-begin
        val={};losses={}
        for step,state in snapshots.items():
            adapter.load_state_dict(state);val[step]=predict_adapter(adapter,'validation')
            losses[step]=metric(val[step],ys['validation'])['mse']
        chosen=min(checkpoints,key=lambda k:losses[k]);adapter.load_state_dict(snapshots[chosen])
        method='plain_'+mode
        predictions['validation'][method]=val[chosen]
        sync(a.device);begin=time.perf_counter()
        predictions['test'][method]=predict_adapter(adapter,'test')
        sync(a.device);test_seconds=time.perf_counter()-begin
        torch.save(dict(config=adapter.config,state_dict=snapshots[chosen],step=chosen),out/f'{method}.pt')
        fits[method]=dict(**log,validation_mse=losses,selected_step=chosen,training_seconds=seconds,
                          test_seconds=test_seconds,parameters=sum(p.numel() for p in adapter.parameters()))
        adapters[mode]=adapter
        del snapshots
    options=['plain_input','plain_output','plain_joint']
    placement=min(options,key=lambda m:metric(predictions['validation'][m],ys['validation'])['mse'])
    choices['plain_placement']=placement
    for s in predictions:predictions[s]['plain_selected']=predictions[s][placement]
    bank=RetrievalBank(torch.tensor(data,device=a.device),meta['adapter_end'],h)
    reference=out.parent/(a.cell.rsplit('_s',1)[0]+'_s0')
    reuse=None
    if seed and (reference/'metrics.json').exists() and not a.smoke:
        prior=json.loads((reference/'metrics.json').read_text())
        if prior['code_sha256']!=hashes or prior['backbone_sha256']!=before:raise RuntimeError('Stale retrieval reuse')
        reuse=dict(cell=reference.name,sha256=sha(reference/'raw_demo.npz'))
    raw_arrays={}
    for s in predictions:
        if reuse:
            with np.load(reference/'raw_demo.npz') as raw:
                np.testing.assert_array_equal(raw[s+'_origins'],origins[s])
                np.testing.assert_array_equal(raw[s+'_targets'],ys[s])
                c=raw[s+'_candidates']
        else:
            print(json.dumps(dict(stage='raw_retrieval',cell=a.cell,split=s)),flush=True)
            result=[];sync(a.device);begin=time.perf_counter()
            for pos in range(0,len(origins[s]),cfg['batch_size']):
                x=torch.tensor(xs[s][pos:pos+cfg['batch_size']],device=a.device)
                oo=torch.tensor(origins[s][pos:pos+cfg['batch_size']],device=a.device)
                pred,info=raw_demo_forecasts(model,bank,x,oo)
                result.append(pred.cpu().numpy())
            sync(a.device);costs['raw_'+s]=dict(seconds=time.perf_counter()-begin,**info)
            c=np.concatenate(result)
        predictions[s]['retrieval_raw']=c.mean(1)
        raw_arrays.update({s+'_candidates':c,s+'_origins':origins[s],s+'_targets':ys[s]})
    if not reuse:np.savez_compressed(out/'raw_demo.npz',**raw_arrays)
    profile={}
    if cfg['dataset']=='ETTh1' and h==96 and seed==0 and not a.smoke:
        suite=AdapterSuite.from_checkpoint(torch.load(source/'adapters.pt',map_location='cpu',weights_only=True),a.device)
        gen=Generator(model,suite,bank,seed)
        oo=torch.tensor(origins['test'][:1],device=a.device)
        def generated(names=None,weights=None):
            c,info=gen.predict(bx,oo,h,names)
            y=c.mean(1) if weights is None else (c*weights.to(c)[None,:,None,None]).sum(1)
            return y,info
        calls={
            'frozen':lambda:(model.predict(bx,h),dict(backbone_calls=1,context_tokens=context)),
            'ttc_weighted':lambda:generated(weights=w),
            'retrieval_raw':lambda:raw_demo_forecasts(model,bank,bx,oo),
            'retrieval_aligned':lambda:generated([names[i] for i in demo_ids])}
        for mode in adapters:
            calls['plain_'+mode]=lambda mode=mode:(adapters[mode](model,bx),dict(backbone_calls=1,context_tokens=context))
        for label,call in calls.items():
            with torch.no_grad():
                call();sync(a.device);timings=[]
                if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
                for _ in range(3):
                    begin=time.perf_counter();_,info=call();sync(a.device);timings.append(time.perf_counter()-begin)
            profile[label]=dict(**info,median_seconds=float(np.median(timings)),repeats=timings,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else 0)
    assert state_digest(model)==before
    arrays={};results={};validation={}
    for s,values in predictions.items():
        for name,pred in values.items():
            if not np.isfinite(pred).all():raise FloatingPointError(name)
            arrays[s+'__'+name]=pred
        arrays[s+'_targets']=ys[s];arrays[s+'_origins']=origins[s]
    arrays['training_origins']=training_origins
    for name,pred in predictions['test'].items():results[name]=metric(pred,ys['test'])
    for name,pred in predictions['validation'].items():validation[name]=metric(pred,ys['validation'])['mse']
    np.savez_compressed(out/'predictions.npz',**arrays)
    result=dict(version=BASELINE_VERSION,cell=a.cell,config=cfg,smoke=a.smoke,code_sha256=hashes,
        source_sha256={n:sha(source/n) for n in ['metrics.json','evidence.npz','predictions.npz','weighted.pt']},
        backbone_sha256=before,backbone_unchanged=True,results=results,validation_mse=validation,
        fitted=fits,choices=choices,raw_retrieval_reuse=reuse,costs=costs,profile=profile,
        training_origins=training_origins.tolist(),elapsed_seconds=time.time()-start,
        dataset=meta,artifacts={p.name:sha(p) for p in sorted(out.glob('*.pt'))},
        predictions_sha256=sha(out/'predictions.npz'))
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(stage='complete',cell=a.cell,seconds=result['elapsed_seconds'],
                         mse={m:r['mse'] for m,r in results.items()})),flush=True)


if __name__=='__main__':main()
