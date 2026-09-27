import sys
sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parents[3]/"src/backends/cuda/kernels"))
"""Validation-only export of reference intermediate values and final responses."""
import json,os
from pathlib import Path
import torch
from fast_candidate import make_router
from rope_selected import install
from laya.common import collate_items
router,agent=make_router('fast_graph');f=agent._fast;install(f,'r1_h8')
for kind,label in [('full_attention','full'),('sliding_attention','local')]:
 for part,t in zip(['cos','sin'],f.rope[kind]):
  Path(f'generated/rope_{label}_{part}.f32').write_bytes(t.cpu().numpy().tobytes())
cases=json.loads(Path('fixtures.json').read_text());results=[]
for case in cases:
 name=case['name'];req=case['request'];qs=req['questions'];internal={k:agent._to_internal(v) for k,v in qs.items()};items=agent._encode_state(req['state'],list(qs),internal);batch=collate_items([items],agent.tok.pad_token_id)
 response=router.predict(**req)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):raw,act=agent._infer(batch)
 n,l0=batch['input_ids'].shape;b=1<<(n-1).bit_length();l=((l0+15)//16*16) if l0<=256 else ((l0+63)//64*64)
 result={'name':name,'request':req,'response':response,'raw_logits':raw.float().cpu().tolist(),'raw_actions':act.float().cpu().tolist(),'B':b,'L':l,'N':n,'L0':l0}
 if name in ('short_1','long_3'):
  # Read the graph output while owned by this serial request; save only real rows for gates.
  ids=torch.zeros((b,l),dtype=torch.long,device='cuda');ids[:n,:l0]=batch['input_ids'].cuda();lens=torch.zeros(b,dtype=torch.int32,device='cuda');lens[:n]=batch['attention_mask'].sum(-1).to('cuda',torch.int32);types=torch.zeros(b,dtype=torch.long,device='cuda');types[:n]=batch['qtype'].cuda()
  directory=Path('evidence/model')/name;directory.mkdir(parents=True,exist_ok=True)
  def save(key,t): (directory/(key+'.f32')).write_bytes(t.float().reshape(b,l,-1)[:n].contiguous().cpu().numpy().tobytes())
  old=f.k_addln;count=[0]
  def addln(*args):
   old(*args);count[0]+=1
   if count[0] in (2,4,6,56):save(f'encoder{count[0]//2-1}_residual',args[0]);save(f'encoder{count[0]//2-1}_normalized',args[4])
  f.k_addln=addln
  with torch.no_grad():
   emb=torch.nn.functional.embedding(ids,f.emb_w).reshape(-1,1024).float();initial=torch.nn.functional.layer_norm(emb,(1024,),f.emb_ln,None,f.eps);save('embedding',initial);h=f._encode(ids,lens,types);save('hidden',h)
  f.k_addln=old
 results.append(result);print('REFERENCE',name,flush=True)
Path('evidence/model-reference.json').write_text(json.dumps(results,indent=2))
Path('requests.jsonl').write_text('\n'.join(json.dumps(c['request']) for c in cases)+'\n')
print('REFERENCE_DONE',len(results),flush=True)
