"""Held-out requests and Attention boundary checks on real exported activations."""
import ctypes,json,sys,subprocess
from pathlib import Path
import torch
from fast_candidate import make_router
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src/backends/cuda/kernels'))
from rope_selected import install
checkpoint,bundle,fixtures,out=sys.argv[1:];out=Path(out);out.mkdir(parents=True,exist_ok=True)
router,agent=make_router('fast_graph');install(agent._fast,'r1_h8')
cases=json.loads(Path(fixtures).read_text());refs=[router.predict(**c['request']) for c in cases]
# Native is a separate Rust process. Its stdin covers all cases and repeated shape switches.
data=''.join(json.dumps(c['request'])+'\n' for c in cases)
p=subprocess.run(['target/release/laya-run',checkpoint,bundle],input=data,text=True,capture_output=True,timeout=180,check=True)
(out/'native.jsonl').write_text(p.stdout);(out/'native.log').write_text(p.stderr);(out/'reference.json').write_text(json.dumps(refs,indent=2))
values=[json.loads(l) for l in p.stdout.splitlines()];assert len(values)==len(refs)
def cmp(a,b):
 if isinstance(a,dict):assert a.keys()==b.keys();return max([cmp(v,b[k]) for k,v in a.items()]+[0])
 if isinstance(a,(int,float)) and not isinstance(a,bool):return abs(a-b)
 if isinstance(a,list):assert len(a)==len(b);return max([cmp(x,y) for x,y in zip(a,b)]+[0])
 assert a==b,(a,b);return 0
errors=[cmp(a,b) for a,b in zip(refs,values)];assert max(errors,default=0)<=.002,errors
(out/'heldout.json').write_text(json.dumps({'cases':len(cases),'max_response_numeric_error':max(errors,default=0),'exact':sum(a==b for a,b in zip(refs,values)),'errors':errors},indent=2))
# No synthetic weights/activations: reuse the real long request's first QKV.
q=torch.frombuffer(bytearray(Path('evidence/probe/long_3/rotated.bin').read_bytes()),dtype=torch.bfloat16).reshape(4,512,3,16,64).cuda()
ptr=ctypes.c_void_p;lib=ctypes.CDLL(str(Path(bundle,'liblaya_cuda.so').resolve()));stream=ptr();lib.laya_init.argtypes=[ctypes.POINTER(ptr)];assert lib.laya_init(ctypes.byref(stream))==0
for name in ['laya_capture_begin','laya_graph_run','laya_graph_free','laya_stream_free','laya_sync']:
 getattr(lib,name).argtypes=[ptr,ptr] if name=='laya_graph_run' else [ptr]
lib.laya_capture_end.argtypes=[ptr,ctypes.POINTER(ptr)]
records=[]
from laya.tl_kernels import attn_kernel
# Dynamic and fixed shape exports share the same row-level empty-key contract.
# Slice only real exported QKV; no synthetic activations/weights are introduced.
source_q = q
for batch,length,fixed,lengths in [
 (4,512,False,[512,129,1,0]),
 (4,512,True,[512,129,1,0]),
 (1,512,True,[1]),
 (1,512,True,[129]),
 (1,512,True,[0]),
 (4,128,False,[128,65,1,0]),
]:
 q=source_q[:batch,:length].contiguous()
 for window,label in [(0,'full'),(64,'local')]:
  name='laya_attn_'+label+(f'_b{batch}_l{length}' if fixed else '')
  fn=getattr(lib,name);fn.argtypes=[ctypes.POINTER(ptr),ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr]
  lens=torch.tensor(lengths,device='cuda',dtype=torch.int32)
  y=torch.empty((batch,length,1024),device='cuda',dtype=torch.bfloat16);ref=torch.empty_like(y)
  attn_kernel(batch if fixed else None,length if fixed else None,16,64,window=window)(q,lens,ref)
  torch.cuda.synchronize()
  args=(ptr*3)(q.data_ptr(),lens.data_ptr(),y.data_ptr())
  def check_rows(mode):
   for b,n in enumerate(lengths):
    # Oracle uses the actual key interval intersection, independently of tiles.
    has_keys=torch.tensor([max(0,qi-window)<=min(n-1,qi+window) if window else n>0 for qi in range(length)],device='cuda',dtype=torch.bool)
    assert torch.isfinite(y[b]).all(),(name,lengths,b,mode,'nonfinite')
    assert torch.equal(y[b,has_keys],ref[b,has_keys]),(name,lengths,b,mode,'nonempty rows')
    assert torch.count_nonzero(y[b,~has_keys])==0,(name,lengths,b,mode,'empty rows')
    if window==64 and n==1 and length>=128:
     assert torch.count_nonzero(y[b,65:128])==0,(name,mode,'mixed tile regression')
  y.fill_(17);torch.cuda.synchronize()
  assert fn(args,batch,length,batch*length,stream)==0;assert lib.laya_sync(stream)==0
  check_rows('eager')
  g=ptr()
  assert lib.laya_capture_begin(stream)==0
  assert fn(args,batch,length,batch*length,stream)==0
  assert lib.laya_capture_end(stream,ctypes.byref(g))==0
  for rep in range(5):
   y.fill_(17);torch.cuda.synchronize()
   assert lib.laya_graph_run(g,stream)==0;assert lib.laya_sync(stream)==0
   check_rows(f'graph-{rep}')
  assert lib.laya_graph_free(g)==0
  records.append({'kernel':name,'batch':batch,'length':length,'lens':lengths,'eager':True,'replays':5,'valid_rows_bitwise':True,'nonempty_rows_bitwise':True,'empty_ranges_zero':True,'mixed_tile_rows_checked':True})
assert lib.laya_stream_free(stream)==0
(out/'attention-boundaries.json').write_text(json.dumps(records,indent=2));print('EXTENDED_PASS',len(cases),errors,flush=True)
