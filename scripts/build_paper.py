"""Compile the unmodified ICLR style from the workspace root on Windows/Linux."""
import argparse
import os
from pathlib import Path
import re
import subprocess
ROOT=Path(__file__).resolve().parents[1]
paper=ROOT/'paper'
p=argparse.ArgumentParser();p.add_argument('--check-pages',action='store_true');a=p.parse_args()
env=dict(os.environ)
for name in ['TEXINPUTS','BIBINPUTS','BSTINPUTS']:env[name]=str(paper)+os.pathsep
commands=[['pdflatex','-output-directory=paper','-interaction=nonstopmode','-halt-on-error','paper/main.tex'],
          ['bibtex','paper/main']]
commands.extend([commands[0]]*2)
for i,cmd in enumerate(commands):
    r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace')
    (paper/f'build_{i}.log').write_text(r.stdout)
    if r.returncode:raise RuntimeError(r.stdout[-4000:])
log=(paper/'main.log').read_text(errors='replace')
if re.search(r'undefined|multiply.defined|Overfull',log):
    print('\n'.join(line for line in log.splitlines() if re.search(r'undefined|multiply.defined|Overfull',line)))
    raise RuntimeError('Resolve LaTeX warnings before publication')
page=int(re.search(r'\\newlabel\{main:end\}\{\{[^}]*\}\{(\d+)\}',(paper/'main.aux').read_text()).group(1))
print('Main text ends on page',page)
if a.check_pages and page>9:raise RuntimeError('ICLR main text exceeds 9 pages')
print(paper/'main.pdf')
