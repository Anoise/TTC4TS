"""Additional domains, fixed before inspecting the full ETT results."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
for directory in ['workspace', 'logs', 'data']:
    (ROOT / directory).mkdir(parents=True, exist_ok=True)
records=[]
sources=[('Appliances','https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip'),
         ('BeijingWeather','https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv')]
for name,url in sources:
    existing=ROOT/'data'/f'{name}.json'
    if existing.exists():
        records.append(dict(dataset=name,**json.loads(existing.read_text())))
        print('Already downloaded',name,flush=True)
        continue
    mirrors = [url]
    if name=='Appliances':
        mirrors.append('https://raw.githubusercontent.com/LuisM78/Appliances-energy-prediction-data/master/energydata_complete.csv')
    else:
        mirrors.append('https://raw.githubusercontent.com/PacktPublishing/Practical-Time-Series-Analysis/master/Data%20Files/PRSA_data_2010.1.1-2014.12.31.csv')
    failure=None
    for download_url in mirrors:
        try:
            with urllib.request.urlopen(download_url,timeout=20) as r: raw=r.read()
            break
        except Exception as e:
            print('Download failure',name,type(e).__name__,flush=True); failure=e
    else: raise failure
    if download_url.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            csv=[n for n in z.namelist() if n.lower().endswith('.csv')][0]
            f=pd.read_csv(z.open(csv))
    else:
        f=pd.read_csv(io.BytesIO(raw))
    if name=='Appliances':
        f=f.drop(columns=['rv1','rv2'])
        period=144
    else:
        dates=pd.to_datetime(f[['year','month','day','hour']])
        # Only complete meteorological variables. PM2.5 is deliberately excluded
        # because labels are missing; this is not a PM2.5 forecasting benchmark.
        f=f[['DEWP','TEMP','PRES','Iws','Is','Ir']].copy()
        f.insert(0,'date',dates)
        period=24
    assert not f.isna().any().any()
    dest=ROOT/'data'/f'{name}.csv'; f.to_csv(dest,index=False)
    n=len(f)
    meta=dict(train_end=int(.7*n),val_end=int(.8*n),test_end=n,period=period,
              source=url,download_url=download_url,license='CC-BY-4.0',columns=f.columns[1:].tolist(),
              raw_sha256=hashlib.sha256(raw).hexdigest(),download_bytes=len(raw))
    (ROOT/'data'/f'{name}.json').write_text(json.dumps(meta,indent=2))
    records.append(dict(dataset=name,processed_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),**meta))
    print(name,len(f),len(f.columns)-1,flush=True)
(ROOT/'workspace'/'uci_manifest.json').write_text(json.dumps(records,indent=2))
