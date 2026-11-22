"""Download public checkpoint files; save source URLs and checksums."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
JOBS=[('amazon/chronos-bolt-small',['config.json','model.safetensors']),
      ('amazon/chronos-bolt-base',['config.json','model.safetensors']),
      ('google/timesfm-2.0-500m-pytorch',['config.json','torch_model.ckpt'])]

def fetch(job, endpoint='https://huggingface.co'):
    model,files=job
    out=ROOT/'checkpoints'/model.split('/')[-1]
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in files:
        source=f'https://huggingface.co/{model}/resolve/main/{name}'
        url=endpoint.rstrip('/')+f'/{model}/resolve/main/{name}?download=true'
        dest=out/name
        for attempt in range(3):
            try:
                if not dest.exists():
                    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
                    with urllib.request.urlopen(req,timeout=90) as response, open(str(dest)+'.part','wb') as f:
                        count=0
                        while True:
                            chunk=response.read(4*1024*1024)
                            if not chunk: break
                            f.write(chunk); count+=len(chunk)
                        if name.endswith(('.ckpt','.safetensors')) and count<1_000_000:
                            raise ValueError(f'Truncated checkpoint: {count} bytes')
                    Path(str(dest)+'.part').replace(dest)
                digest=hashlib.file_digest(open(dest,'rb'),'sha256').hexdigest()
                records.append(dict(file=name,source=source,download_url=url,bytes=dest.stat().st_size,sha256=digest))
                print('READY',model,name,dest.stat().st_size,flush=True)
                break
            except Exception as e:
                print('RETRY',model,name,type(e).__name__,str(e)[:200],flush=True)
                if attempt==2: raise
                time.sleep(2)
    (out/'download_manifest.json').write_text(json.dumps(dict(model=model,files=records),indent=2))
    return model

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--endpoint',default='https://huggingface.co',help='Official Hub or a mirror endpoint')
    args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        print(list(pool.map(lambda job: fetch(job,args.endpoint),JOBS)),flush=True)
