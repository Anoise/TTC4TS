"""Execute every offline notebook with this Python interpreter, failing on errors."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--notebook', nargs='*', help='Notebook filenames; default: all')
    p.add_argument('--timeout', type=int, default=600)
    args = p.parse_args()
    paths = [ROOT/'notebooks'/n for n in args.notebook] if args.notebook else sorted((ROOT/'notebooks').glob('*.ipynb'))
    # A temporary explicit kernel avoids using an unrelated global kernelspec.
    with tempfile.TemporaryDirectory(prefix='ttc4ts-kernel-') as directory:
        spec = Path(directory)/'ttc4ts-release'
        spec.mkdir()
        (spec/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable, '-m', 'ipykernel_launcher',
            '--HistoryManager.hist_file=:memory:', '-f', '{connection_file}'],
            env={'IPYTHONDIR': str(Path(directory)/'ipython'), 'JUPYTER_RUNTIME_DIR': str(Path(directory)/'runtime')},
            display_name='TTC4TS release', language='python')), encoding='utf-8')
        manager = KernelSpecManager(kernel_dirs=[directory], ensure_native_kernel=False)
        for path in paths:
            if not path.resolve().is_relative_to(ROOT/'notebooks'):
                raise ValueError('Notebook must be inside notebooks/')
            nb = nbformat.read(path, as_version=4)
            print('EXECUTE', path.name, flush=True)
            km = KernelManager(kernel_name='ttc4ts-release', kernel_spec_manager=manager)
            client = NotebookClient(nb, km=km, timeout=args.timeout, resources={'metadata': {'path': str(ROOT)}}, record_timing=False)
            try:
                client.execute()
            finally:
                if km.has_kernel:
                    km.shutdown_kernel(now=True)
            # Save portable metadata, while retaining genuine execution outputs.
            nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
            for cell in nb.cells:
                cell.metadata.pop('execution', None)
            nbformat.validate(nb)
            nbformat.write(nb, path)
            print('PASS', path.name, flush=True)
    print(f'Executed {len(paths)} notebooks. No GPU job runs with the default notebook settings.')


if __name__ == '__main__': main()
