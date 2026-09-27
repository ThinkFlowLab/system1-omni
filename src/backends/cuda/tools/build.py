"""Compile an exported bundle. Preserve separate TileLang/PyTorch arithmetic flags."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);a=p.parse_args()
m=json.loads((a.bundle/'manifest.json').read_text());root=Path(__file__).resolve().parents[1]
nvcc=str(Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/nvcc')
commands=[];objects=[];sources={}
for source,fast in [(a.bundle/'generated.cu',True),(root/'kernels/runtime.cu',False),(root/'kernels/model_ops.cu',False)]:
 sources[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
 obj=a.bundle/(source.stem+'.o');objects.append(str(obj))
 flags=[f for f in m['nvcc_flags'] if fast or f!='--use_fast_math']
 cmd=[nvcc,*flags,'--expt-relaxed-constexpr','-c','-Xcompiler=-fPIC','-O3',*[arg for d in m['include_dirs'] for arg in ['-I',d]],str(source),'-o',str(obj)]
 commands.append(cmd);subprocess.run(cmd,check=True)
cmd=[nvcc,'-shared',*objects,'-lcublas','-lcuda','-o',str(a.bundle/'liblaya_cuda.so')];commands.append(cmd);subprocess.run(cmd,check=True)
(a.bundle/'build-command.json').write_text(json.dumps(commands,indent=2))

(a.bundle/'build-manifest.json').write_text(json.dumps({'abi':1,'arch':'sm_90a','nvcc':subprocess.check_output([nvcc,'--version'],text=True),'sources':sources,'commands':commands,'library_sha256':hashlib.sha256((a.bundle/'liblaya_cuda.so').read_bytes()).hexdigest()},indent=2))
