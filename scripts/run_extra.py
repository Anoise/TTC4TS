"""Run the unchanged experiment protocol on a separately installed model family."""
import hashlib
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if '--model' in sys.argv and sys.argv[sys.argv.index('--model')+1]=='chronos-2':
    sys.path.insert(0,str(ROOT/'.runtime/chronos2'))
import run_experiment
from extra_backbones import load_extra
original_digest=run_experiment.source_digest
def digest():
    h=hashlib.sha256(original_digest().encode())
    for p in [Path(__file__),ROOT/'scripts/extra_backbones.py']:h.update(p.read_bytes())
    return h.hexdigest()
run_experiment.load_backbone=load_extra
run_experiment.source_digest=digest
if __name__=='__main__':run_experiment.main()
