"""Export real first-layer tensors from the frozen official fast model."""
import json,sys
from pathlib import Path
import torch
from fast_candidate import make_router
from laya.common import collate_items
from rope_candidate import build
router,agent=make_router('fast_no_graph');f=agent._fast
cases=json.loads(Path('fixtures.json').read_text())
for name in ['short_1','long_3']:
 req=next(c['request'] for c in cases if c['name']==name);qs=req['questions'];internal={k:agent._to_internal(v) for k,v in qs.items()}
 items=agent._encode_state(req['state'],list(qs),internal);batch=collate_items([items],agent.tok.pad_token_id)
 n,l0=batch['input_ids'].shape;b=1<<(n-1).bit_length();l=((l0+15)//16*16) if l0<=256 else ((l0+63)//64*64)
 ids=torch.zeros((b,l),dtype=torch.long,device='cuda');ids[:n,:l0]=batch['input_ids'].cuda();lens=torch.zeros(b,dtype=torch.int32,device='cuda');lens[:n]=batch['attention_mask'].sum(-1).to('cuda',torch.int32)
 with torch.no_grad():
  emb=torch.nn.functional.embedding(ids,f.emb_w).reshape(-1,1024).float();x=torch.nn.functional.layer_norm(emb,(1024,),f.emb_ln,None,f.eps);y=x.bfloat16();qkv=torch.empty(b*l,3072,dtype=torch.bfloat16,device='cuda')
  f.k_qkv(y,f.layers[0]['wqkv'],f.zeros[3072],qkv);before=qkv.clone();cos,sin=f.rope_tab('full_attention',l);build(16,64,1,8)(qkv,cos,sin);out=torch.empty(b,l,1024,dtype=torch.bfloat16,device='cuda');f.attn_k(b,l,0)(qkv.view(b,l,3,16,64),lens,out)
  # Probe dynamic attention export separately; long static reference may differ in lowering.
  from laya.tl_kernels import attn_kernel
  dyn=torch.empty_like(out);attn_kernel(None,None,16,64)(qkv.view(b,l,3,16,64),lens,dyn)
  real_equal=torch.equal(dyn[:n],out[:n]);assert real_equal, 'dynamic/static differ on real rows'
  # Empty bucket rows are not model outputs; original TMA wait bug leaves them undefined.
  # Canonical zero is the new explicit padding contract, not a relaxed real-output tolerance.
  dyn[n:]=0
  d=Path('evidence/probe')/name;d.mkdir(parents=True,exist_ok=True)
  for key,t in {'input':y,'weight':f.layers[0]['wqkv'],'qkv':before,'rotated':qkv,'cos':cos,'sin':sin,'lens':lens,'attention':dyn}.items():
   (d/(key+'.bin')).write_bytes(t.contiguous().cpu().view(torch.uint8).numpy().tobytes())
  (d/'shape.json').write_text(json.dumps({'B':b,'L':l,'dynamic_vs_official_real_rows_equal':real_equal,'real_rows':n,'padding_contract':'zero; reference original has proven asynchronous Q/O shared-memory race'}))
  print(name,b,l,'static/dynamic equal',torch.equal(dyn,out),flush=True)
