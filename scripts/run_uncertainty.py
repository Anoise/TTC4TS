"""One UQ cell from saved candidates, with fresh, leakage-free lightweight fits."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from ttc4ts.aggregators import LearnedAggregator,greedy_weights,retrospective_weights,errors
from ttc4ts.data import load_dataset,windows
from ttc4ts.uncertainty import (ALPHAS,QUANTILES,UQ_VERSION,UncertaintyHead,
    uncertainty_features,ScaledConformal,RollingConformal,conformal_quantile,
    interval_metrics,weighted_interval_score)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def point_predictions(blocks,fit_idx,selection_idx,seed,out,device):
    """Refit/select without touching the original validation (now calibration)."""
    c,f,y,history=blocks['aggregator']
    all_blocks={'selection':tuple(v[selection_idx] for v in blocks['aggregator']),
                'calibration':blocks['validation'],'test':blocks['test']}
    predictions={}
    for split,(cc,ff,yy,hh) in all_blocks.items():
        predictions[split]={'backbone':cc[:,0], 'uniform':cc.mean(1)}
    fit_c,fit_f,fit_y=c[fit_idx],f[fit_idx],y[fit_idx]
    fitted={}
    for mode in ['weighted','classifier','risk','ranking']:
        torch.manual_seed(seed)
        agg=LearnedAggregator(c.shape[1],mode,seq_len=c.shape[2],enc_dim=c.shape[3],hidden_dim=512).to(device)
        log=agg.fit(fit_c.to(device),fit_f.to(device),fit_y.to(device),epochs=160,seed=seed)
        for split,(cc,ff,_,_) in all_blocks.items():
            with torch.no_grad():
                predictions[split][mode]=torch.cat([agg.forecast(a.to(device),b.to(device)).cpu()
                    for a,b in zip(cc.split(16),ff.split(16))])
        agg.cpu(); torch.save(agg.state_dict(),out/f'point_{mode}.pt')
        fitted[mode]=dict(first_loss=log[0],last_loss=log[-1],updates=len(log))
        del agg
    gw=greedy_weights(fit_c,fit_y)
    sel_c,_,sel_y,sel_h=all_blocks['selection']
    single=int(errors(sel_c,sel_y).mean(0).argmin())
    best=None
    for temp in [.1,.3,1.]:
        for shrink in [0.,.5,1.]:
            w=retrospective_weights(sel_h,temp,shrink)
            loss=float(((sel_c*w[:,:,None,None]).sum(1)-sel_y).square().mean())
            if best is None or loss<best[0]:best=(loss,temp,shrink)
    _,temp,shrink=best
    for split,(cc,_,_,hh) in all_blocks.items():
        p=predictions[split]
        p['greedy']=(cc*gw[None,:,None,None]).sum(1)
        p['validation_best']=cc[:,single]
        p['retrospective']=(cc*retrospective_weights(hh,temp,shrink)[:,:,None,None]).sum(1)
    allowed=['backbone','validation_best','weighted','classifier','risk','ranking','greedy','retrospective','uniform']
    losses={name:float((predictions['selection'][name]-sel_y).square().mean()) for name in allowed}
    choice=min(allowed,key=lambda name:losses[name])
    selection=dict(method=choice,mse=losses,single_index=single,greedy_weights=gw.tolist(),
                   retrospective=dict(temperature=temp,shrinkage=shrink),fits=fitted)
    return {split:dict(point=p[choice].numpy(),frozen=p['backbone'].numpy())
            for split,p in predictions.items()},selection


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cell',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',default='results_uncertainty_v1')
    args=parser.parse_args()
    torch.set_num_threads(2)
    source=ROOT/'results_revision_d03_demo10'/args.cell
    out=ROOT/args.output/args.cell;out.mkdir(parents=True,exist_ok=True)
    code={p.relative_to(ROOT).as_posix():sha(p) for p in
          [Path(__file__),ROOT/'ttc4ts/uncertainty.py',ROOT/'ttc4ts/aggregators.py',ROOT/'ttc4ts/data.py']}
    if (out/'metrics.json').exists():
        saved=json.loads((out/'metrics.json').read_text())
        if saved['code_sha256']!=code:raise RuntimeError('Refusing stale uncertainty results')
        print('ALREADY_COMPLETE',args.cell);return
    started=time.time()
    original=json.loads((source/'metrics.json').read_text())
    cfg=original['config']; h=cfg['horizon']; seed=cfg['seed']
    if h!=96:raise ValueError('Protocol is prespecified at H=96')
    if original['adapter_config']['max_delta']!=.3 or original['retrieval_top_k']!=10:
        raise ValueError('Wrong generator protocol')
    torch.manual_seed(seed);np.random.seed(seed)
    data,meta=load_dataset(ROOT/'data'/f"{cfg['dataset']}.csv")
    assert meta==original['dataset']
    blocks={};origins={}
    with np.load(source/'evidence.npz',allow_pickle=False) as evidence:
        names=evidence['candidate_names'].tolist()
        for split in ['aggregator','validation','test']:
            origins[split]=evidence[split+'_origins']
            blocks[split]=tuple(torch.from_numpy(evidence[split+'_'+key]).float()
                for key in ['candidates','features','targets','historical_losses'])
            actual=np.stack([data[o:o+h] for o in origins[split]])
            np.testing.assert_array_equal(actual,blocks[split][2].numpy())
    assert len(names)==80
    ao=origins['aggregator'];cut=int(.6*len(ao));fit_idx=np.arange(cut)
    selection_idx=np.flatnonzero((np.arange(len(ao))>=cut)&(ao>=ao[cut-1]+h))
    assert len(selection_idx)>=8
    assert ao[fit_idx[-1]]+h<=ao[selection_idx[0]]
    assert ao[selection_idx[-1]]+h<=origins['validation'][0]
    assert origins['validation'][-1]+h<=origins['test'][0]
    assert len(origins['validation'])>=19
    assert np.all(np.diff(origins['validation'])>=h) and np.all(np.diff(origins['test'])>=h)
    point,selection=point_predictions(blocks,fit_idx,selection_idx,seed,out,args.device)
    split_origins={'selection':ao[selection_idx],'calibration':origins['validation'],'test':origins['test']}
    cc={'selection':blocks['aggregator'][0][selection_idx],
        'calibration':blocks['validation'][0],'test':blocks['test'][0]}
    hh={'selection':blocks['aggregator'][3][selection_idx],
        'calibration':blocks['validation'][3],'test':blocks['test'][3]}
    targets={};feats={};spreads={};contexts={};volatility={}
    for split,oo in split_origins.items():
        x,y=windows(data,oo,cfg['context'],h);contexts[split]=x;targets[split]=y
        ff,sp=uncertainty_features(cc[split],torch.from_numpy(point[split]['point']),
                                  torch.from_numpy(x),hh[split],names)
        feats[split]=ff;spreads[split]=sp.numpy()
        volatility[split]=x.std(1).mean(-1)
    scales={};quantiles={};training={}
    for label,dim,mode in [('history',5,'scale'),('candidate',12,'scale'),('cqr',12,'quantile')]:
        torch.manual_seed(seed+100)
        head=UncertaintyHead(dim,mode).to(args.device)
        residual=torch.from_numpy(targets['selection']-point['selection']['point']).to(args.device)
        loss=head.fit(feats['selection'].to(args.device),residual,seed=seed+100)
        pred={}
        for split in split_origins:
            with torch.no_grad():pred[split]=torch.cat([head(x.to(args.device)).cpu()
                for x in feats[split].split(16)]).numpy()
        if mode=='scale':scales[label]=pred
        else:quantiles=pred
        head.cpu();torch.save(head.checkpoint(),out/f'{label}_head.pt')
        training[label]=dict(first_loss=loss[0],last_loss=loss[-1],updates=len(loss))
        del head
    del blocks
    cy,ty=targets['calibration'],targets['test'];cp=point['calibration']['point'];tp=point['test']['point']
    cb=point['calibration']['frozen'];tb=point['test']['frozen']
    one_c,one_t=np.ones_like(cp),np.ones_like(tp)
    floor=np.maximum(.01,.05*np.sqrt(np.square(targets['selection']-point['selection']['point']).mean(0)))
    spread_c=np.sqrt(spreads['calibration']**2+floor**2)
    spread_t=np.sqrt(spreads['test']**2+floor**2)
    # Correct Gaussian baseline uses only the pre-calibration fitting labels.
    sigma=np.sqrt(np.square(targets['selection']-point['selection']['frozen']).mean(0)).clip(.01)
    cqr_c=quantiles['calibration']+cp[...,None];cqr_t=quantiles['test']+tp[...,None]
    levels=np.asarray(QUANTILES)
    raw_c=np.moveaxis(np.quantile(cc['calibration'].numpy(),levels,axis=1),0,-1)
    raw_t=np.moveaxis(np.quantile(cc['test'].numpy(),levels,axis=1),0,-1)
    arrays={'target':ty,'point':tp,'frozen':tb,'history':contexts['test'][:,-96:],
            'test_origins':split_origins['test'],'calibration_origins':split_origins['calibration'],
            'fit_origins':ao[fit_idx],'selection_origins':ao[selection_idx],
            'candidate_scale':scales['candidate']['test'],
            'calibration_target':cy,'calibration_point':cp,'calibration_frozen':cb,
            'calibration_candidate_scale':scales['candidate']['calibration']}
    metrics={};all_intervals={};medians={};online={}
    threshold=float(np.quantile(volatility['selection'],.75));high=volatility['test']>threshold
    arrays['high_volatility']=high
    def record(method,alpha,lo,hi,median):
        tag=f'{method}_a{int(round(alpha*100)):02d}'
        lo=np.asarray(lo,dtype=np.float32);hi=np.asarray(hi,dtype=np.float32)
        m=interval_metrics(ty,lo,hi,alpha)
        covered=(ty>=lo)&(ty<=hi)
        m['high_volatility_coverage']=float(covered[high].mean()) if high.any() else None
        m['low_volatility_coverage']=float(covered[~high].mean()) if (~high).any() else None
        metrics[tag]=m; arrays[tag+'_lower']=lo;arrays[tag+'_upper']=hi
        all_intervals.setdefault(method,[]).append((lo,hi));medians[method]=median
    for alpha in ALPHAS:
        z=NormalDist().inv_cdf(1-alpha/2)
        record('frozen_gaussian',alpha,tb-z*sigma,tb+z*sigma,tb)
        for method,calpoint,testpoint,cscale,tscale in [
            ('frozen_split',cb,tb,one_c,one_t),('ttc_split',cp,tp,one_c,one_t),
            ('history_scale',cp,tp,scales['history']['calibration'],scales['history']['test']),
            ('candidate_spread',cp,tp,spread_c,spread_t),
            ('candidate_scale',cp,tp,scales['candidate']['calibration'],scales['candidate']['test'])]:
            cal=ScaledConformal(alpha).fit(cy,calpoint,cscale)
            lo,hi=cal.predict(testpoint,tscale);record(method,alpha,lo,hi,testpoint)
            arrays[f'{method}_a{int(round(alpha*100)):02d}_q']=cal.q
        li=int(np.argmin(np.abs(levels-alpha/2)));ui=int(np.argmin(np.abs(levels-(1-alpha/2))))
        record('raw_candidates',alpha,raw_t[...,li],raw_t[...,ui],raw_t[...,3])
        for label,qc,qt in [('candidate_quantile_cp',raw_c,raw_t),('cqr',cqr_c,cqr_t)]:
            # Nonnegative correction: a declared conservative CQR variant.
            scores=np.maximum.reduce([qc[...,li]-cy,cy-qc[...,ui],np.zeros_like(cy)])
            q=conformal_quantile(scores,alpha)
            record(label,alpha,qt[...,li]-q,qt[...,ui]+q,qt[...,3])
            arrays[f'{label}_a{int(round(alpha*100)):02d}_q']=q
        for label,calpoint,testpoint,cscale,tscale,adaptive in [
            ('frozen_rolling',cb,tb,one_c,one_t,False),
            ('ttc_rolling',cp,tp,one_c,one_t,False),
            ('bounded_aci',cp,tp,one_c,one_t,True),
            ('candidate_rolling',cp,tp,scales['candidate']['calibration'],scales['candidate']['test'],False)]:
            seq=RollingConformal(np.abs(cy-calpoint)/cscale,alpha,adaptive)
            lows=[];highs=[]
            for i in range(len(ty)):
                lo,hi=seq.predict(testpoint[i],tscale[i]);lows.append(lo);highs.append(hi)
                # This origin's entire horizon is observed before the next origin.
                seq.observe(ty[i])
            record(label,alpha,np.stack(lows),np.stack(highs),testpoint)
            online[f'{label}_{alpha}']=dict(updates=seq.updates,clipped_coordinate_updates=seq.clipped_updates)
    wis={}
    for label,intervals in all_intervals.items():
        values=weighted_interval_score(ty,medians[label],intervals)
        wis[label]=dict(mean=float(values.mean()),per_origin=values.mean((1,2)).tolist())
        arrays[label+'_median']=medians[label]
    simultaneous={}
    for label,cscale,tscale in [('ttc',one_c,one_t),
        ('candidate',scales['candidate']['calibration'],scales['candidate']['test'])]:
        cal=ScaledConformal(.1,True).fit(cy,cp,cscale)
        lo,hi=cal.predict(tp,tscale);lo=lo.astype('float32');hi=hi.astype('float32')
        simultaneous[label]=interval_metrics(ty,lo,hi,.1)
        arrays['simultaneous_'+label+'_lower']=lo;arrays['simultaneous_'+label+'_upper']=hi
        arrays['simultaneous_'+label+'_q']=cal.q
    np.savez_compressed(out/'intervals.npz',**arrays)
    result=dict(version=UQ_VERSION,cell=args.cell,model=cfg['model'],dataset=cfg['dataset'],seed=seed,horizon=h,
        protocol='60% aggregator fit; purged remainder select/UQ fit; original validation calibrate; original test evaluate',
        origins={k:v.tolist() for k,v in dict(fit=ao[fit_idx],**split_origins).items()},
        selection=selection,training=training,metrics=metrics,wis=wis,simultaneous=simultaneous,
        volatility_threshold=threshold,online=online,code_sha256=code,
        source_sha256={p.name:sha(p) for p in [source/'metrics.json',source/'evidence.npz']},
        intervals_sha256=sha(out/'intervals.npz'),elapsed_seconds=time.time()-started,
        environment=dict(torch=torch.__version__,numpy=np.__version__,device=args.device),
        channels=meta['columns'],dataset_sha256=meta['sha256'])
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cell=args.cell,seconds=result['elapsed_seconds'],selection=selection['method'],
        coverage90={k:v['coverage'] for k,v in metrics.items() if k.endswith('a10')})),flush=True)

if __name__=='__main__':main()
