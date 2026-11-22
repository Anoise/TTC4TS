"""One reproducible model/dataset/horizon/seed experiment, no test-time fitting."""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from ttc4ts.backbones import load_backbone,state_digest,SeasonalNaive
from ttc4ts.data import load_dataset,split_origins,windows
from ttc4ts.adapters import fit_adapters,ADAPTER_VERSION
from ttc4ts.generators import Generator,candidate_specs,GENERATOR_VERSION,DEFAULT_RETRIEVAL_TOP_K
from ttc4ts.retrieval import RetrievalBank
from ttc4ts.aggregators import LearnedAggregator,features,errors,retrospective_weights,greedy_weights,AGGREGATOR_VERSION


def metrics(pred,target):
    e=pred-target
    return dict(mse=float(e.square().mean()),mae=float(e.abs().mean()),
                per_origin_mse=e.square().mean((1,2)).tolist(),
                per_origin_mae=e.abs().mean((1,2)).tolist())


def source_digest():
    h=hashlib.sha256()
    for p in sorted((ROOT/'ttc4ts').glob('*.py'))+[Path(__file__)]:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def synchronize(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()


def fit_aggregator(mode,ac,af,ay,vc,vf,tc,tf,args,epochs,seed):
    """Fit trajectory classifier on the GPU; keep small feature scorers on CPU."""
    torch.manual_seed(seed)
    device=args.device if mode=='classifier' else 'cpu'
    agg=LearnedAggregator(ac.shape[1],mode,seq_len=ac.shape[2],
                          enc_dim=ac.shape[3],hidden_dim=512).to(device)
    log=agg.fit(ac.to(device),af.to(device),ay.to(device),epochs=epochs,seed=seed)
    def predict(c,f):
        with torch.no_grad():
            return torch.cat([agg.forecast(cc.to(device),ff.to(device)).cpu()
                for cc,ff in zip(c.split(32),f.split(32))])
    pv,pt=predict(vc,vf),predict(tc,tf)
    return agg.cpu(),log,pv,pt


def make_cache(generator,data,origins,length,horizon,batch_size,device,shared=None):
    preds=[]; costs=[]
    for start in range(0,len(origins),batch_size):
        oo=origins[start:start+batch_size]
        x=torch.tensor(windows(data,oo,length,horizon)[0],device=device)
        synchronize(device); begin=time.perf_counter()
        if shared is None:
            p,cost=generator.predict(x,torch.tensor(oo,device=device),horizon)
        else:
            p=torch.tensor(shared[start:start+len(oo)],device=device)
            # Recompute every candidate whose implementation depends on the
            # current generator, including retrievals.  Reusing old retrieval
            # tensors would silently preserve stale neighbour-selection code.
            ids=[i for i,s in enumerate(generator.specs)
                 if s.family in {'adapter','retrieval'} or s.name=='jitter']
            names=[generator.specs[i].name for i in ids]
            changed,cost=generator.predict(x,torch.tensor(oo,device=device),horizon,names,base_forecast=p[:,0])
            p[:,ids]=changed
            cost['reused_candidates']=p.shape[1]-len(ids)
        synchronize(device)
        cost.update(seconds=time.perf_counter()-begin,origins=len(oo))
        preds.append(p.cpu()); costs.append(cost)
    return torch.cat(preds),costs


def evaluate(cache,origins,data,splits,args,out):
    index={int(o):i for i,o in enumerate(origins)}
    sets={}; arrays={}; outputs={}; fitted={}; selections={}
    specs=candidate_specs(); names=[s.name for s in specs]
    for split in ['aggregator','validation','test']:
        oo=splits[split]
        x,y=windows(data,oo,args.context,args.horizon)
        c=cache[[index[int(o)] for o in oo]]
        past=cache[[index[int(o-args.horizon)] for o in oo]]
        past_y=torch.tensor(np.stack([data[o-args.horizon:o] for o in oo]))
        hist=errors(past,past_y)
        feat=features(torch.tensor(x),c,hist)
        sets[split]=(c,feat,torch.tensor(y),hist,torch.tensor(x))
        arrays[split+'_origins']=oo
        arrays[split+'_targets']=y
        arrays[split+'_features']=feat.numpy()
        arrays[split+'_historical_losses']=hist.numpy()
    ac,af,ay,ah,ax=sets['aggregator']; vc,vf,vy,vh,vx=sets['validation']; tc,tf,ty,th,tx=sets['test']
    predictions={}; valpredictions={}
    def record(name,p,v=None):
        outputs[name]=metrics(p,ty)
        predictions[name]=p.numpy()
        if v is not None:
            valpredictions[name]=v
            outputs[name]['validation_mse']=float((v-vy).square().mean())
    record('backbone',tc[:,0],vc[:,0])
    record('uniform',tc.mean(1),vc.mean(1))
    record('median',tc.median(1).values,vc.median(1).values)
    record('seasonal_naive',SeasonalNaive(args.period).predict(tx,args.horizon))
    for i,name in enumerate(names): record('candidate/'+name,tc[:,i],vc[:,i])
    best=int(errors(vc,vy).mean(0).argmin())
    record('validation_best',tc[:,best],vc[:,best]); selections['single_candidate']=names[best]
    for mode in ['weighted','classifier','risk','ranking']:
        agg,log,pv,pt=fit_aggregator(mode,ac,af,ay,vc,vf,tc,tf,args,args.aggregator_steps,args.seed)
        record(mode,pt,pv)
        torch.save(agg.state_dict(),out/f'{mode}.pt')
        fitted[mode]=dict(loss=log,parameters=sum(p.numel() for p in agg.parameters()))
    # Static greedy ensemble on the same aggregator fitting block.
    gw=greedy_weights(ac,ay)
    record('greedy',(tc*gw[None,:,None,None]).sum(1),(vc*gw[None,:,None,None]).sum(1))
    selections['greedy_weights']=gw.tolist()
    # Prespecified retrospective hyperparameter grid, validation only.
    best=None
    for temperature in [.1,.3,1.]:
        for shrinkage in [0.,.5,1.]:
            w=retrospective_weights(vh,temperature,shrinkage)
            loss=float(((vc*w[:,:,None,None]).sum(1)-vy).square().mean())
            if best is None or loss<best[0]: best=(loss,temperature,shrinkage)
    _,temp,shrink=best
    tw=retrospective_weights(th,temp,shrink); vw=retrospective_weights(vh,temp,shrink)
    record('retrospective',(tc*tw[:,:,None,None]).sum(1),(vc*vw[:,:,None,None]).sum(1))
    selections['retrospective']=dict(temperature=temp,shrinkage=shrink,validation_mse=best[0])
    # One deployment choice, made entirely on the validation split.
    allowed=['backbone','validation_best','weighted','classifier','risk','ranking','greedy','retrospective','uniform']
    choice=min(allowed,key=lambda n:outputs[n]['validation_mse'])
    record('validation_selected',torch.tensor(predictions[choice]),valpredictions[choice])
    selections['deployment_method']=choice
    # Oracle is a NON-DEPLOYABLE upper-bound diagnostic. Never in deployment choices.
    ix=errors(tc,ty).argmin(-1)
    record('oracle',tc[torch.arange(len(tc)),ix])
    families=['adapter','augmentation','retrieval']
    for n in range(1,4):
        for combo in itertools.combinations(families,n):
            ids=[i for i,s in enumerate(specs) if s.family in combo or i==0]
            record('families/'+'+'.join(combo),tc[:,ids].mean(1),vc[:,ids].mean(1))
    # Balanced nested budgets fixed before looking at any test target.
    # Keep the identity first, then add the corrected adapter heads before the
    # fixed augmentation and retrieval operators.
    representative={family:min((s for s in specs if s.name.startswith(family+'_')),
                               key=lambda s:abs(s.value-.3)).name
                    for family in ['input','output','joint']}
    ordered=['identity',representative['joint'],'ewma','retrieval_demo1',representative['input'],
             representative['output'],'slope_plus','retrieval_demo2']
    ordered += [n for n in names if n not in ordered]
    order=[names.index(n) for n in ordered]
    budget_steps=getattr(args,'budget_aggregator_steps',40)
    budget_methods=['weighted','classifier','risk','ranking']
    for k in [1,2,4,8,16,len(specs)]:
        ids=order[:k]
        budget={}
        def brecord(label, test_pred, val_pred):
            name=f'budget/{label}_{k}'
            record(name,test_pred,val_pred)
            budget[label]=name
        brecord('frozen',tc[:,0],vc[:,0])
        brecord('uniform',tc[:,ids].mean(1),vc[:,ids].mean(1))
        if k==len(specs):
            for mode in budget_methods:
                brecord(mode,tc[:,0]*0+torch.tensor(predictions[mode]),
                        torch.tensor(valpredictions[mode]))
        else:
            for mode in budget_methods:
                agg,_,pv,pt=fit_aggregator(mode,ac[:,ids],af[:,ids],ay,vc[:,ids],vf[:,ids],
                                          tc[:,ids],tf[:,ids],args,budget_steps,args.seed+k)
                brecord(mode,pt,pv)
        bw=greedy_weights(ac[:,ids],ay)
        brecord('greedy',(tc[:,ids]*bw[None,:,None,None]).sum(1),
                (vc[:,ids]*bw[None,:,None,None]).sum(1))
        retro_best=None
        for temperature in [.1,.3,1.]:
            for shrinkage in [0.,.5,1.]:
                rw=retrospective_weights(vh[:,ids],temperature,shrinkage)
                loss=float(((vc[:,ids]*rw[:,:,None,None]).sum(1)-vy).square().mean())
                if retro_best is None or loss<retro_best[0]: retro_best=(loss,temperature,shrinkage)
        _,temperature,shrinkage=retro_best
        rw=retrospective_weights(th[:,ids],temperature,shrinkage)
        rv=retrospective_weights(vh[:,ids],temperature,shrinkage)
        brecord('retrospective',(tc[:,ids]*rw[:,:,None,None]).sum(1),
                (vc[:,ids]*rv[:,:,None,None]).sum(1))
        bix=errors(vc[:,ids],vy).mean(0).argmin()
        brecord('validation_best',tc[:,ids][:,bix],vc[:,ids][:,bix])
        choices=['frozen','uniform']+budget_methods+['greedy','retrospective','validation_best']
        choice=min(choices,key=lambda label:outputs[budget[label]]['validation_mse'])
        brecord('validation_selected',torch.tensor(predictions[budget[choice]]),
                valpredictions[budget[choice]])
        ix=errors(tc[:,ids],ty).argmin(-1)
        brecord('oracle',tc[:,ids][torch.arange(len(tc)),ix],vc[:,ids][torch.arange(len(vc)),errors(vc[:,ids],vy).argmin(-1)])
    for variant in ['input','output','joint']:
        ids=[0]+[i for i,n in enumerate(names) if n.startswith(variant+'_')]
        record('adapters/'+variant,tc[:,ids].mean(1),vc[:,ids].mean(1))
    # Observed-input shift score. Threshold fit on validation, not test errors.
    def drift(x):
        half=x.shape[1]//2
        return ((x[:,half:].mean(1)-x[:,:half].mean(1)).abs()/x[:,:half].std(1,unbiased=False).clamp_min(.1)).mean(1)
    threshold=float(drift(vx).quantile(.75)); high=drift(tx)>threshold
    selections['shift_threshold']=threshold
    for name in allowed+['validation_selected']:
        e=(torch.tensor(predictions[name])-ty).square().mean((1,2))
        outputs[name]['high_shift_mse']=float(e[high].mean()) if high.any() else None
        outputs[name]['low_shift_mse']=float(e[~high].mean()) if (~high).any() else None
    arrays.update(test_candidates=tc.numpy(),validation_candidates=vc.numpy(),aggregator_candidates=ac.numpy(),
                  candidate_names=np.array(names),shift_scores=drift(tx).numpy(),high_shift=high.numpy())
    np.savez_compressed(out/'evidence.npz',**arrays)
    np.savez_compressed(out/'predictions.npz',**predictions)
    return outputs,selections,fitted


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='chronos-bolt-small')
    p.add_argument('--dataset',default='ETTh1')
    p.add_argument('--horizon',type=int,default=96)
    p.add_argument('--context',type=int,default=512)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--adapter-steps',type=int,default=32)
    p.add_argument('--reuse-from',default=None,help='Reuse only unchanged candidates from a prior, verified cell')
    p.add_argument('--aggregator-steps',type=int,default=160)
    p.add_argument('--budget-aggregator-steps',type=int,default=40)
    p.add_argument('--fit-windows',type=int,default=96)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--limit-eval',type=int,default=0)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',default='results_revision_d03_demo10')
    p.add_argument('--profile',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(args.seed); np.random.seed(args.seed)
    cell=f'{args.model}_{args.dataset}_h{args.horizon}_s{args.seed}'
    out=ROOT/args.output/cell; out.mkdir(parents=True,exist_ok=True)
    digest=source_digest()
    if (out/'metrics.json').exists():
        existing=json.loads((out/'metrics.json').read_text())
        if existing.get('source_sha256')!=digest: raise RuntimeError('Refusing stale cache; use a new output directory')
        print('ALREADY_COMPLETE',cell); return
    started=time.time(); data,meta=load_dataset(ROOT/'data'/f'{args.dataset}.csv')
    args.period=meta['period']
    splits=split_origins(meta,args.context,args.horizon,args.fit_windows)
    # Historical scoring labels must occur after adapter fitting finished.
    splits['aggregator']=splits['aggregator'][splits['aggregator']>=meta['adapter_end']+args.horizon]
    if len(splits['aggregator'])<2: raise ValueError('Too few purged aggregator windows')
    if args.limit_eval:
        for s in ['aggregator','validation','test']: splits[s]=splits[s][:args.limit_eval]
    model=load_backbone(args.model,ROOT/'checkpoints',args.device)
    if args.model=='naive': model.period=meta['period']
    before=state_digest(model)
    parameter_count=sum(p.numel() for p in model.parameters())
    trainx,trainy=windows(data,splits['adapter'],args.context,args.horizon)
    print(json.dumps(dict(stage='adapters',cell=cell,origins={k:len(v) for k,v in splits.items()},parameters=parameter_count)),flush=True)
    fitstart=time.time()
    adapter_batch = 1 if args.horizon >= 720 or 'base' in args.model else 2
    adapters,adapter_logs=fit_adapters(model,torch.tensor(trainx,device=args.device),
        torch.tensor(trainy,device=args.device),steps=args.adapter_steps,seed=args.seed,
        batch_size=adapter_batch)
    training_seconds=time.time()-fitstart
    after=state_digest(model)
    if before!=after: raise AssertionError('Frozen backbone changed')
    torch.save(adapters.checkpoint(),out/'adapters.pt')
    bank=RetrievalBank(torch.tensor(data,device=args.device),meta['adapter_end'],args.horizon)
    generator=Generator(model,adapters,bank,args.seed)
    origins=np.array(sorted({int(o-r*args.horizon) for split in ['aggregator','validation','test'] for o in splits[split] for r in [0,1]}))
    print(json.dumps(dict(stage='candidates',cell=cell,total=len(origins),adapter_seconds=training_seconds)),flush=True)
    if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    shared=None; reuse=None
    reference=(ROOT/args.reuse_from/cell if args.reuse_from else
               ROOT/args.output/f'{args.model}_{args.dataset}_h{args.horizon}_s0')
    if (args.seed or args.reuse_from) and (reference/'metrics.json').exists():
        prior=json.loads((reference/'metrics.json').read_text())
        matching=(prior['backbone_sha256']==before and prior['dataset']==meta and
                  all(prior['config'][k]==getattr(args,k) for k in
                      ['context','horizon','fit_windows','limit_eval','model','dataset','batch_size']))
        if matching:
            with np.load(reference/'cache.npz') as archive:
                if np.array_equal(archive['origins'],origins):
                    old=archive['candidates']
                    with np.load(reference/'evidence.npz') as evidence:
                        prior_names=evidence['candidate_names'].tolist()
                    shared=np.zeros((len(old),len(generator.specs),args.horizon,data.shape[1]),dtype=old.dtype)
                    for j,spec in enumerate(generator.specs):
                        if spec.family in {'identity','augmentation'} and spec.name!='jitter':
                            shared[:,j]=old[:,prior_names.index(spec.name)]
                    reuse=dict(source=str(reference.relative_to(ROOT)),source_sha256=prior['source_sha256'],
                        metrics_sha256=hashlib.sha256((reference/'metrics.json').read_bytes()).hexdigest(),
                        note='12 unchanged identity/augmentation candidates; adapters, all retrievals and jitter recomputed.')
    cache,costs=make_cache(generator,data,origins,args.context,args.horizon,args.batch_size,args.device,shared)
    peak=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0
    np.savez_compressed(out/'cache.npz',origins=origins,candidates=cache.numpy())
    result,selections,fitted=evaluate(cache,origins,data,splits,args,out)
    profile={}
    if args.profile:
        o=int(splits['test'][0]); x=torch.tensor(data[o-args.context:o],device=args.device)[None]
        out_name=f'output_{max(s.value for s in generator.specs if s.name.startswith("output_")):g}'
        for label,names in [('backbone',['identity']),('all',None),('output_adapter',['identity',out_name])]:
            times=[]
            for repeat in range(4):
                synchronize(args.device); tick=time.perf_counter()
                _,cost=generator.predict(x,torch.tensor([o],device=args.device),args.horizon,names)
                synchronize(args.device)
                if repeat: times.append(time.perf_counter()-tick)
            profile[label]=dict(median_seconds=float(np.median(times)),samples=times,**cost)
    final=dict(config=vars(args),dataset=meta,source_sha256=digest,backbone_sha256=before,
        backbone_unchanged=before==after,backbone_parameters=parameter_count,
        adapter_version=ADAPTER_VERSION,adapter_config=adapters.config,
        generator_version=GENERATOR_VERSION,retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K,
        aggregator_version=AGGREGATOR_VERSION,classifier_config=dict(seq_len=args.horizon,enc_dim=data.shape[1],hidden_dim=512,num_samples=len(generator.specs)),
        adapter_parameters=sum(p.numel() for p in adapters.parameters()),
        adapter_training_seconds=training_seconds,adapter_training_loss=adapter_logs,
        candidate_costs=costs,shared_candidate_cache=reuse,peak_gpu_bytes=peak,elapsed_seconds=time.time()-started,
        results=result,selections=selections,aggregator_fit=fitted,profile=profile,
        origins={k:v.tolist() for k,v in splits.items()},
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                         device=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu'))
    (out/'metrics.json').write_text(json.dumps(final,indent=2,allow_nan=False))
    print(json.dumps(dict(stage='complete',cell=cell,seconds=final['elapsed_seconds'],
                         backbone=result['backbone']['mse'],selected=result['validation_selected']['mse'],
                         choice=selections['deployment_method'])),flush=True)


if __name__=='__main__': main()
