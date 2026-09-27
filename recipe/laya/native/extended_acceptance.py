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
for window,label in [(0,'full'),(64,'local')]:
 fn=getattr(lib,'laya_attn_'+label);fn.argtypes=[ctypes.POINTER(ptr),ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr]
 lens=torch.tensor([512,129,1,0],device='cuda',dtype=torch.int32);y=torch.empty((4,512,1024),device='cuda',dtype=torch.bfloat16);ref=torch.empty_like(y)
 attn_kernel(None,None,16,64,window=window)(q,lens,ref);torch.cuda.synchronize()
 args=(ptr*3)(q.data_ptr(),lens.data_ptr(),y.data_ptr());g=ptr()
 assert lib.laya_capture_begin(stream)==0;assert fn(args,4,512,2048,stream)==0;assert lib.laya_capture_end(stream,ctypes.byref(g))==0
 for rep in range(5):
  y.fill_(17);torch.cuda.synchronize();assert lib.laya_graph_run(g,stream)==0;assert lib.laya_sync(stream)==0
  for b,l in enumerate([512,129,1,0]):
   if l:assert torch.equal(y[b,:l],ref[b,:l]),(label,b,rep)
   if l==0:assert torch.count_nonzero(y[b])==0
   if window and l and l+window+64<512:
    # Whole query tiles beyond the local window have no key loop iterations.
    start=((l+window+63)//64)*64
    assert torch.count_nonzero(y[b,start:])==0,(label,b,start)
 assert lib.laya_graph_free(g)==0;records.append({'kernel':label,'replays':5,'valid_rows_bitwise':True,'empty_ranges_zero':True})
assert lib.laya_stream_free(stream)==0
(out/'attention-boundaries.json').write_text(json.dumps(records,indent=2));print('EXTENDED_PASS',len(cases),errors,flush=True)
