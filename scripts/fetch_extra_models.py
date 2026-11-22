"""Download public model artifacts to a separate V2 checkpoint directory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parents[1]


def download(model,metadata_only=False):
    dest=ROOT/'checkpoints_extra'/model.split('/')[-1]; dest.mkdir(parents=True,exist_ok=True)
    metadata=None; failures=[]
    for host in ['huggingface.co','hf-mirror.com']:
        try:
            req=urllib.request.Request(f'https://{host}/api/models/{model}',headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req,timeout=25) as r: metadata=json.load(r)
            break
        except Exception as e:failures.append(str(e))
    if metadata is None:
        # Direct public file downloads can work even when the listing API does not.
        candidates=['config.json','model.safetensors','README.md']
        if 'timer' in model:candidates+=['configuration_timer.py','modeling_timer.py']
        if 'sundial' in model:candidates+=['configuration_sundial.py','modeling_sundial.py','generation_utils.py']
        metadata=dict(sha='main',siblings=[dict(rfilename=f) for f in candidates],listing_failures=failures)
    (dest/'hub_metadata.json').write_text(json.dumps(metadata,indent=2))
    sha=metadata['sha']; files=[r['rfilename'] for r in metadata.get('siblings',[])]
    chosen=[f for f in files if '/' not in f and (f.endswith(('.json','.py','.md')) or (not metadata_only and f.endswith('.safetensors')))]
    records=[]
    for file in chosen:
        target=dest/file
        source=f'https://huggingface.co/{model}/resolve/{sha}/{file}'
        if not target.exists():
            for host in ['huggingface.co','hf-mirror.com']:
                try:
                    url=source.replace('huggingface.co',host)+'?download=true'
                    request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
                    with urllib.request.urlopen(request,timeout=45) as r,open(str(target)+'.part','wb') as f:
                        while chunk:=r.read(4*1024*1024):f.write(chunk)
                    Path(str(target)+'.part').replace(target);break
                except Exception as e:
                    print('FAIL',model,file,host,str(e),flush=True)
            else:raise RuntimeError(f'Failed {model}/{file}')
        with target.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        records.append(dict(file=file,source=source,sha256=digest,bytes=target.stat().st_size))
        print('READY',model,file,target.stat().st_size,flush=True)
    (dest/'download_manifest.json').write_text(json.dumps(dict(model=model,revision=sha,files=records),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metadata-only',action='store_true')
    p.add_argument('--models',nargs='+',default=['thuml/timer-base-84m','thuml/sundial-base-128m','amazon/chronos-2'])
    a=p.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda m:download(m,a.metadata_only),a.models))
