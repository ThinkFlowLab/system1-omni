"""Build-only TileLang -> CUDA export. Runtime needs CUDA, not Python/TVM/Torch.

Host argument stacks are inspected, including dynamic TMA extents and strides.
Unknown symbols/launch layouts fail generation instead of guessing an ABI.
"""
import argparse, hashlib, importlib.util, json, re, subprocess, sys
from pathlib import Path
import tilelang
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels'))
import laya_tilelang as K

SLOT=re.compile(r'\(\(\(TVMFFIAny\*\)stack_ffi_any\)\[(\d+)\]\.v_(?:int64|ptr)\) = (.*);')
CALL=re.compile(r'TVMFFIFunctionCall\((\w+?)_packed, \(TVMFFIAny\*\) stack_ffi_any, (\d+),')

def host_calls(k):
    slots={}; calls=[]
    for line in k.get_host_source().splitlines():
        m=SLOT.search(line)
        if m: slots[int(m[1])]=m[2]; continue
        m=CALL.search(line)
        if m:
            if m[1] in ('__tvm_tensormap_create_tiled','main_kernel'):
                vals=[slots.get(i) for i in range(int(m[2]))]
                if None in vals: raise ValueError(('missing argument',m[1],vals))
                calls.append((m[1],vals))
            slots={}
    return calls

def integer(s):
    s=s.replace('(int64_t)','').replace('(','').replace(')','')
    return int(s)

def export(name,k):
    src=k.get_kernel_source(); signature=re.search(r'void main_kernel\((.*?)\);',src,re.S)[1]
    params=[p.strip() for p in signature.split(',')]
    names=[p.split()[-1].lstrip('*') for p in params]
    bindings=[]
    for i,p in enumerate(k.prim_func.params):
        buf=k.prim_func.buffer_map[p];n=buf.name
        dt=str(buf.dtype);ctype={'bfloat16':'bfloat16_t','float32':'float','int32':'int','int64':'int64_t'}[dt]
        bindings.append(f'  auto* {n}=static_cast<{ctype}*>(p[{i}]);')
    desc=[];launch=None
    for callee,args in host_calls(k):
        if callee=='main_kernel': launch=args;continue
        var,dtype,rank,tensor=args[:4];r=integer(rank)
        dt=integer(dtype)
        if dt not in (7,9) or not 1<=r<=5:raise ValueError(('unsupported TMA format',name,dtype,rank))
        dtype_enum={7:'CU_TENSOR_MAP_DATA_TYPE_FLOAT32',9:'CU_TENSOR_MAP_DATA_TYPE_BFLOAT16'}[dt]
        dims=args[4:4+r];stride=args[4+r:4+2*r];box=args[4+2*r:4+3*r];steps=args[4+3*r:4+4*r]
        inter,swizzle,l2,oob=map(integer,args[4+4*r:])
        if integer(stride[0])!={7:4,9:2}[dt] or inter!=0 or oob!=0 or swizzle not in range(4) or l2 not in range(4):raise ValueError('unsupported TMA layout')
        desc.append(f'''  alignas(64) CUtensorMap {var};
  {{ uint64_t dims[]={{{','.join('static_cast<uint64_t>('+v+')' for v in dims)}}}, strides[]={{{','.join('static_cast<uint64_t>('+v+')' for v in stride[1:])}}};
     uint32_t box[]={{{','.join(box)}}}, steps[]={{{','.join(steps)}}};
     CUresult rc=cuTensorMapEncodeTiled(&{var},{dtype_enum},{r},{tensor},dims,strides,box,steps,
       CU_TENSOR_MAP_INTERLEAVE_NONE,static_cast<CUtensorMapSwizzle>({swizzle}),static_cast<CUtensorMapL2promotion>({l2}),CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
     if(rc!=CUDA_SUCCESS)return 10000+static_cast<int>(rc); }}''')
    if launch is None:raise ValueError('no launch')
    # Scalar/pointer entries are exactly the recovered device signature order.
    args=launch[:len(params)];tail=launch[len(params):]
    for n,v in zip(names,args):
        if n!=v and n not in ('M','B','L'):raise ValueError(('argument changed',n,v))
    if len(tail)>=4 and integer(tail[-2])==1 and integer(tail[-3])==1 and integer(tail[-1])>1:
        grid=tail[:-4];block=list(map(integer,tail[-4:-1]));smem=integer(tail[-1])
    else:
        grid=tail[:-3];block=list(map(integer,tail[-3:]));smem=0
    if not 1<=len(grid)<=3 or block[1:]!=[1,1] or smem>227*1024:raise ValueError(('launch',tail))
    if len(grid)<3:grid+=['1']*(3-len(grid))
    symbol='laya_'+name+'_kernel';body=src[src.index('extern "C" __global__'):].replace('main_kernel',symbol)
    # TileLang reuses Q_s for O_s. Its generated wait is inside the key loop,
    # so zero-key rows can overwrite Q_s while the asynchronous Q load is live.
    # Wait before entering producer/consumer branches, including zero iterations.
    # Match the lowered structure strictly; do not silently patch a new lowering.
    if name.startswith('attn_'):
        qload=re.search(r'tl::tma_load\(QKV_desc, mbarrier\[(\d+)\].*?Q_s.*?;',body)
        if not qload: raise ValueError('attention Q TMA load changed')
        end=body.index('__syncthreads();',qload.end())+len('__syncthreads();')
        body=body[:end]+f'\n  mbarrier[{qload[1]}].wait(0); // Q load must complete even with no valid keys.\n'+body[end:]
    pre=src[:src.index('extern "C" __global__')]
    # Never inject unknown identifiers from a compiler expression into the wrapper.
    allowed=set(names)|{'M','B','L','int64_t'}|{str(k.prim_func.buffer_map[p].name) for p in k.prim_func.params}
    for _,values in host_calls(k):
        for v in values:
            for ident in re.findall(r'\b[A-Za-z_]\w*\b',v):
                if ident not in allowed and not ident.endswith('_desc'):raise ValueError(('unknown host symbol',ident))
    callargs=[]
    for p,v in zip(params,args):callargs.append(v)
    wrapper=f'''extern "C" int laya_{name}(void** p,int B,int L,int M,cudaStream_t stream) {{
  if(B<1 || B>16 || L<16 || L>512 || L%16 || M!=B*L)return -1;
{chr(10).join(bindings)}
{chr(10).join(desc)}
  {symbol}<<<dim3({','.join(grid)}),dim3({','.join(map(str,block))}),{smem},stream>>>({','.join(callargs)});
  return static_cast<int>(cudaGetLastError());
}}
'''
    init=f'if(auto e=cudaFuncSetAttribute({symbol},cudaFuncAttributeMaxDynamicSharedMemorySize,{smem});e!=cudaSuccess)return static_cast<int>(e);' if smem>49152 else ''
    return pre,body+wrapper,init,{'name':name,'params':names,'block':block,'smem':smem,'grid':grid,'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'emitted_sha256':hashlib.sha256(body.encode()).hexdigest(),'zero_key_wait':name.startswith('attn_'),'host_calls':host_calls(k)}

def main():
 p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--rope-source',type=Path,default=Path(__file__).resolve().parents[1]/'kernels/rope_selected.py');p.add_argument('--probe-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 spec=importlib.util.spec_from_file_location('rope_selected',a.rope_source);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
 kernels={'rope':r.build(16,64,1,8),'rope_original':K.rope_kernel(16,64),'qkv':K.gemm_kernel(3072,1024),'attn_full':K.attn_kernel(None,None,16,64)}
 if not a.probe_only:
  kernels.update({'out':K.gemm_kernel(1024,1024),'geglu':K.gemm_geglu_kernel(2624,1024),'down':K.gemm_kernel(1024,2624),'addln':K.add_ln_kernel(1024),'addln_bias':K.add_ln_kernel(1024,bias=True),'ln_bias':K.add_ln_kernel(1024,residual=False,bias=True),'head_in':K.gemm_kernel(3072,1024,bias=True),'head_out':K.gemm_kernel(1024,1024,bias=True),'ffn1':K.gemm_kernel(4096,1024,bias=True,act='relu'),'ffn2':K.gemm_kernel(1024,4096,bias=True),'attn_local':K.attn_kernel(None,None,16,64,window=64)})
 if not a.probe_only:
  for b in (1,4):
   for label,window in [('full',0),('local',64)]:kernels[f'attn_{label}_b{b}_l512']=K.attn_kernel(b,512,16,64,window=window)
 preambles=[];bodies=[];inits=[];manifest=[]
 for name,k in kernels.items():
  pre,body,init,meta=export(name,k);preambles.append(pre);bodies.append(body);inits.append(init);manifest.append(meta)
 # Only one copy of debug helper definitions. Other headers carry include guards.
 pre='\n'.join(dict.fromkeys(line for block in preambles for line in block.splitlines() if line.startswith('#include <tl_templates')))
 code='#include <cuda.h>\n#include <cuda_runtime.h>\n'+pre+'\n'+'\n'.join(bodies)+'\nextern "C" int laya_kernels_init(){'+''.join(inits)+'return 0;}\n'
 (a.output/'generated.cu').write_text(code)
 manifest={'tilelang_version':tilelang.__version__,'kernels':manifest,'nvcc_flags':['-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--use_fast_math','-DENABLE_BF16'],'include_dirs':[str(TILELANG_TEMPLATE_PATH),str(CUTLASS_INCLUDE_DIR)]}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print('EXPORTED',len(kernels),flush=True)
if __name__=='__main__':main()
