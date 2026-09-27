"""Serial no-profiler benchmark. Run variants in alternating order under one GPU lease."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('variant',choices=['native','native_original','fast','fast_original']);p.add_argument('checkpoint');p.add_argument('bundle');p.add_argument('output',type=Path);p.add_argument('--samples',type=int,default=50);a=p.parse_args()
cases=[c for c in json.loads(Path(__file__).with_name('fixtures.json').read_text()) if c['name'] in ['short_1','short_3','long_1','long_3']]
result={'variant':a.variant,'warmup':10,'samples':a.samples,'cases':{}}
if a.variant.startswith('native'):
 cmd=['target/release/laya-run',a.checkpoint,a.bundle]+(['--original-rope'] if a.variant.endswith('original') else [])
 env=dict(os.environ);env.pop('LAYA_RAW_LOGITS',None);env.pop('LAYA_DUMP_DIR',None);env.pop('LAYA_DUMP_HIDDEN',None)
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,env=env)
 while True:
  line=proc.stderr.readline()
  if not line:raise RuntimeError('native exited before ready')
  if line.startswith('READY'):break
 maps=Path(f'/proc/{proc.pid}/maps').read_text();a.output.with_suffix('.maps').write_text(maps)
 assert 'libtorch' not in maps and 'libpython' not in maps
 def infer(req):
  proc.stdin.write(json.dumps(req)+'\n');proc.stdin.flush();response=json.loads(proc.stdout.readline());assert 'error' not in response,response
  while True:
   line=proc.stderr.readline()
   if not line:raise RuntimeError('native exited')
   if line.startswith('engine_wall_ms='):return response,float(line.split('=')[1])
else:
 import sys
 sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src/backends/cuda/kernels'))
 import torch
 from fast_candidate import make_router,metadata
 router,agent=make_router('fast_graph')
 if a.variant=='fast':
  from rope_selected import install
  install(agent._fast,'r1_h8')
 result['metadata']=metadata(agent)
 def infer(req):
  torch.cuda.synchronize();start=time.perf_counter_ns();response=router.predict(**req);torch.cuda.synchronize()
  return response,(time.perf_counter_ns()-start)/1e6
for c in cases:
 for _ in range(10):infer(c['request'])
 samples=[]
 for _ in range(a.samples):response,ms=infer(c['request']);samples.append(ms)
 result['cases'][c['name']]={'ms':samples,'response':response}
 print(a.variant,c['name'],sorted(samples)[len(samples)//2],flush=True)
if a.variant.startswith('native'):
 proc.stdin.close();assert proc.wait(timeout=10)==0
result['timing']='native: parse + pack + CUDA + decode + JSON/pipe write; fast: Router.predict + completion sync; warmed C1, no profiler'
a.output.write_text(json.dumps(result,indent=2))
