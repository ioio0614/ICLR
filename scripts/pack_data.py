#!/usr/bin/env python3
"""Pack included input data into reproducible gzip/tar parts for ordinary Git transfer."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
ROOTS=['ultrafeedback_binarized_cleaned','alpaca_eval','alpaca_assets','arena-hard-auto/data','mt_bench/raw']
PART_BYTES=40*1024**2

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    files=sorted(p for name in ROOTS for p in (ROOT/name).rglob('*') if p.is_file())
    out=ROOT/'data_bundle';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Data bundle already exists; preserve it rather than overwrite')
    entries=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in files]
    with tempfile.TemporaryFile() as archive:
        with gzip.GzipFile(fileobj=archive,mode='wb',compresslevel=3,mtime=0,filename='') as compressed:
            with tarfile.open(fileobj=compressed,mode='w|') as tar:
                for p in files:
                    info=tar.gettarinfo(str(p),arcname=str(p.relative_to(ROOT)))
                    info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
                    with p.open('rb') as f:tar.addfile(info,f)
        archive.seek(0);parts=[];number=0
        while True:
            data=archive.read(PART_BYTES)
            if not data:break
            p=out/f'inputs.tar.gz.part{number:03d}';p.write_bytes(data)
            parts.append({'file':p.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});number+=1
    manifest={'format':'tar.gz split into ordered byte parts','part_bytes':PART_BYTES,'parts':parts,'files':entries}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Input files:',len(files),'raw bytes:',sum(e['bytes'] for e in entries),
          'parts:',len(parts),'compressed bytes:',sum(e['bytes'] for e in parts))

if __name__=='__main__':main()
