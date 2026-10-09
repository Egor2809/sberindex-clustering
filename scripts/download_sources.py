"""Download and verify the fixed public snapshot; Windows, Linux and macOS.

RAR extraction requires bsdtar, unrar, 7z or a RAR-capable tar (Windows).
No model fitting is performed. Existing mismatched files are never overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
SOURCES=[
 ('hackathonlicence.zip','http://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip','a9f932ff4096a7df797d1547987937f34d3995ac445b4748177114488d12b010'),
 ('t_dict_municipal.rar','http://www.sberbank.com/common/files/t_dict_municipal.rar','319ed22684b77716641bc15b61f7325adc44e9fc47e9be2973d88412d27d21f5'),
 ('metadata_municipal_dict_sberindex_2.pdf','http://www.sberbank.ru/common/img/uploaded/files/pdf/sberindex/metadata_municipal_dict_sberindex_2.pdf','05707427a64061552d1f23f74f578a99fdabeff69f0c8a0fc99d924210fee0b3')]
REGISTRY_HASHES={
 't_dict_municipal_districts.xlsx':'4150658c3298fbc87ed79838f503f3a5b9a28da33257ff803c7d9231ddb775d6',
 't_dict_municipal_districts_poly.gpkg':'e62027630d48e4a13f9b6d173dd074d7358706743fe859ca3b7fa30fefa9813a'}


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def download(directory):
    directory.mkdir(parents=True,exist_ok=True)
    for name,url,expected in SOURCES:
        target=directory/name
        if not target.exists():
            temporary=target.with_suffix(target.suffix+'.part')
            request=urllib.request.Request(url,headers={'User-Agent':'SberAI-reproducibility/1.0'})
            try:
                with urllib.request.urlopen(request,timeout=90) as response,temporary.open('wb') as out:
                    shutil.copyfileobj(response,out,1024*1024)
                if digest(temporary)!=expected:raise ValueError('Source snapshot changed: '+name)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        if digest(target)!=expected:raise ValueError('Existing source differs: '+name)
        print('Verified '+name,flush=True)


def extract_registry(directory):
    for name,expected in REGISTRY_HASHES.items():
        if (directory/name).exists() and digest(directory/name)!=expected:
            raise ValueError('Existing registry file differs: '+name)
    names=[name for name in REGISTRY_HASHES if not (directory/name).exists()]
    if not names:return
    archive=str(directory/'t_dict_municipal.rar')
    for executable in ['bsdtar','unrar','7z','tar']:
        path=shutil.which(executable)
        if not path:continue
        if executable in ('bsdtar','tar'):
            command=[path,'-xf',archive,'-C',str(directory),*names]
        elif executable=='unrar':
            command=[path,'e','-o-',archive,*names,str(directory)+str(Path('/'))]
        else:
            command=[path,'e','-aos',archive,*names,'-o'+str(directory)]
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if result.returncode==0 and all((directory/n).is_file() for n in names):
            for name,expected in REGISTRY_HASHES.items():
                if digest(directory/name)!=expected:raise ValueError('Extracted registry differs: '+name)
            return
    raise RuntimeError('RAR extraction needs bsdtar (libarchive-tools), unrar or 7z. Extract the two named registry files into '+str(directory))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only',action='store_true',help='Check existing downloads without extracting')
    args=p.parse_args()
    destination=ROOT/'artifacts/sources/acquisition'
    if args.verify_only:
        for name,_,expected in SOURCES:
            if digest(destination/name)!=expected:raise ValueError('Source differs: '+name)
            print('Verified '+name)
    else:
        download(destination)
        extract_registry(destination)
        subprocess.run([sys.executable,str(ROOT/'scripts/unpack_sources.py')],check=True)
