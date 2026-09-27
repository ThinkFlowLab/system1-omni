"""Build-only rotary tables, preserving official FastLaya GPU BF16 rounding."""
import argparse, hashlib, importlib.metadata, json
from pathlib import Path
import torch
from laya import Agent
p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('bundle',type=Path);a=p.parse_args()
assert importlib.metadata.version('laya')=='0.3.20', 'requires laya==0.3.20'
agent=Agent(str(a.checkpoint),device='cuda',fast=False,compile=False)
assert agent.device.type=='cuda'
assert agent.accelerate(use_graphs=False,strict=True)
a.bundle.mkdir(parents=True,exist_ok=True);files={}
for kind,label in [('full_attention','full'),('sliding_attention','local')]:
 for part,t in zip(['cos','sin'],agent._fast.rope[kind]):
  name=f'rope_{label}_{part}.f32';data=t.cpu().numpy().tobytes()
  assert len(data)==512*32*4
  (a.bundle/name).write_bytes(data);files[name]=hashlib.sha256(data).hexdigest()
metadata={'abi':1,'laya':'0.3.20','hidden_size':1024,'head_dim':64,'max_len':512,'tables':files,'config_sha256':{name:hashlib.sha256((a.checkpoint/name).read_bytes()).hexdigest() for name in ['rl_agent_config.json','encoder/config.json']}}
(a.bundle/'tables.json').write_text(json.dumps(metadata,indent=2))
print('exported four rotary tables')
