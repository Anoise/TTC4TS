"""Create a validated source ZIP without Git history or private runtime files."""
import argparse
import hashlib
from pathlib import Path
import zipfile

from check_release import ROOT, public_files, validate_artifacts, validate_public_files, validate_supplementary, validate_notebooks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='dist/TTC4TS-source.zip')
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    validate_public_files()
    validate_artifacts()
    validate_supplementary()
    validate_notebooks()
    files = public_files()
    if output in files:
        raise ValueError('Archive destination collides with a source file')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            info = zipfile.ZipInfo('TTC4TS/'+path.relative_to(ROOT).as_posix(), (2026,9,26,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix+'.sha256').write_text(digest+'  '+output.name+'\n')
    print(f'{len(files)} files, {output.stat().st_size} bytes: {output}')
    print('SHA256:', digest)


if __name__ == '__main__':
    main()
