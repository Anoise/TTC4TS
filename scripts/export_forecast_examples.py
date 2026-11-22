"""Extract reproducible, explicitly post-hoc success examples from saved forecasts.

No checkpoint is loaded, fitted or executed. Run in the completed campaign root.
The six model/dataset pairings, H=96, seed=0 and selection rule are fixed below.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ttc4ts.data import load_dataset

PAIRS = [
    ('chronos-bolt-small','ETTh1'),
    ('chronos-bolt-base','ETTh2'),
    ('timesfm-2.0-500m-pytorch','ETTm1'),
    ('timer-base-84m','ETTm2'),
    ('sundial-base-128m','Appliances'),
    ('chronos-2','BeijingWeather'),
]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    dest=ROOT/'outputs/forecast_examples'
    dest.mkdir(parents=True,exist_ok=True)
    metadata=[]
    for number,(model,dataset) in enumerate(PAIRS,1):
        cell=f'{model}_{dataset}_h96_s0'
        directory=ROOT/'results_revision_d03_demo10'/cell
        r=json.loads((directory/'metrics.json').read_text())
        with np.load(directory/'evidence.npz',allow_pickle=False) as f:
            targets=f['test_targets'].astype('float64')
            origins=f['test_origins']
        with np.load(directory/'predictions.npz',allow_pickle=False) as f:
            base=f['backbone'].astype('float64')
            ttc=f['validation_selected'].astype('float64')
            greedy=f['greedy'].astype('float64')
        for name,pred in [('backbone',base),('validation_selected',ttc),('greedy',greedy)]:
            actual=np.square(pred-targets).mean()
            assert np.isclose(actual,r['results'][name]['mse'],rtol=3e-6,atol=1e-7),(cell,name)
        data,meta=load_dataset(ROOT/'data'/f'{dataset}.csv')
        assert meta==r['dataset'],cell
        dates=pd.read_csv(ROOT/'data'/f'{dataset}.csv',usecols=['date'])['date'].astype(str).to_numpy()
        np.testing.assert_allclose(np.stack([data[o:o+96] for o in origins]),targets,rtol=0,atol=0)
        eb=np.square(base-targets).mean(1)
        et=np.square(ttc-targets).mean(1)
        variance=targets.var(1)
        gain=1-et/np.maximum(eb,1e-12)
        # Success examples require improvement and error below half target variance.
        # Pick the observed order statistic nearest the 75th percentile of their
        # gains; stable sorting resolves ties by origin index, then channel.
        eligible=(gain>0)&(et<.5*variance)&(variance>1e-8)&np.isfinite(gain)
        positions=np.argwhere(eligible)
        if len(positions)==0:
            raise RuntimeError(f'No qualifying success example for fixed pair {cell}')
        order=np.argsort(gain[eligible],kind='stable')
        quantile_index=int(np.floor(.75*(len(order)-1)+.5))
        i,c=map(int,positions[order[quantile_index]])
        o=int(origins[i]); filepath=dest/f'example_{number}.npz'
        np.savez_compressed(filepath,history=data[o-96:o,c],target=targets[i,:,c],
                            frozen=base[i,:,c],ttc4ts=ttc[i,:,c],greedy=greedy[i,:,c],
                            relative_time=np.arange(-96,96),dates=dates[o-96:o+96])
        row=dict(number=number,cell=cell,model=model,dataset=dataset,horizon=96,seed=0,
                 test_origin_index=i,origin=o,forecast_start=dates[o],forecast_end=dates[o+95],
                 channel_index=c,channel=meta['columns'][c],eligible_count=len(order),
                 origin_channel_count=int(gain.size),quantile_order_index=quantile_index,
                 frozen_mse=float(eb[i,c]),ttc4ts_mse=float(et[i,c]),gain=float(gain[i,c]),
                 target_variance=float(variance[i,c]),deployment_method=r['selections']['deployment_method'],
                 normalization='adapter-prefix z-score',
                 metrics_sha256=sha(directory/'metrics.json'),
                 evidence_sha256=sha(directory/'evidence.npz'),predictions_sha256=sha(directory/'predictions.npz'),
                 data_sha256=meta['sha256'],extracted_file=filepath.name,extracted_sha256=sha(filepath))
        metadata.append(row)
        print(json.dumps({k:row[k] for k in ['number','model','dataset','origin','channel','eligible_count','gain','frozen_mse','ttc4ts_mse','deployment_method']}),flush=True)
    payload=dict(selection_rule='Fixed six model/dataset pairs, H96, seed0; among origin/channel pairs with positive TTC4TS MSE reduction and TTC4TS MSE below half the target variance, select the order statistic nearest the 75th percentile of the reductions. Post-hoc illustrative success selection uses test targets; never used to fit or select a deployed predictor.',
                 script_sha256=sha(Path(__file__)),examples=metadata)
    (dest/'manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    import tarfile
    with tarfile.open(ROOT/'forecast_examples.tar.gz','w:gz') as tar:
        for path in sorted(dest.iterdir()): tar.add(path,arcname=path.relative_to(ROOT).as_posix())
    print(json.dumps(dict(archive='forecast_examples.tar.gz',sha256=sha(ROOT/'forecast_examples.tar.gz'))))

if __name__=='__main__': main()
