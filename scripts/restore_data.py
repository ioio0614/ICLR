#!/usr/bin/env python3
"""Verify and restore bundled inputs without networking or ML dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--destination',type=Path,default=ROOT,help='Optional separate directory for a restore test')
    a=p.parse_args();dest=a.destination.resolve();bundle=ROOT/'data_bundle'
    manifest=json.loads((bundle/'manifest.json').read_text())
    for part in manifest['parts']:
        f=bundle/part['file']
        if f.stat().st_size!=part['bytes'] or digest(f)!=part['sha256']:raise ValueError(f'Corrupted data part: {f.name}')
    expected={e['path']:e for e in manifest['files']}
    for name,e in expected.items():
        target=(dest/name).resolve()
        if not target.is_relative_to(dest):raise ValueError('Unsafe data path')
        if target.exists() and (not target.is_file() or digest(target)!=e['sha256']):
            raise ValueError(f'Existing local data differs; preserve it before restoring: {target}')
    if not a.verify_only:
        with tempfile.TemporaryFile() as combined:
            for part in manifest['parts']:
                with (bundle/part['file']).open('rb') as f:shutil.copyfileobj(f,combined)
            combined.seek(0);seen=set()
            with tarfile.open(fileobj=combined,mode='r|gz') as tar:
                for member in tar:
                    if not member.isfile() or member.name not in expected or member.name in seen:raise ValueError('Unexpected archive entry')
                    seen.add(member.name);e=expected[member.name]
                    if member.size!=e['bytes']:raise ValueError('Archive size mismatch')
                    target=dest/member.name
                    if target.exists():continue
                    target.parent.mkdir(parents=True,exist_ok=True)
                    fd,tmp=tempfile.mkstemp(prefix=target.name+'.restore-',dir=target.parent)
                    try:
                        with os.fdopen(fd,'wb') as out,tar.extractfile(member) as inp:shutil.copyfileobj(inp,out)
                        if digest(Path(tmp))!=e['sha256']:raise ValueError('Restored data checksum mismatch')
                        os.chmod(tmp,0o644);os.replace(tmp,target)
                    finally:
                        if Path(tmp).exists():Path(tmp).unlink()
                if seen!=set(expected):raise ValueError('Archive is incomplete')
    for name,e in expected.items():
        f=dest/name
        if not f.is_file() or f.stat().st_size!=e['bytes'] or digest(f)!=e['sha256']:raise ValueError(f'Missing or corrupted input: {name}')
    print('Verified',len(expected),'input files and',len(manifest['parts']),'archive parts.')

if __name__=='__main__':main()
