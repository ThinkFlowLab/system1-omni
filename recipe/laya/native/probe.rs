//! Standalone Rust/CUDA probe: no Cargo dependencies or Python runtime.
use std::{ffi::c_void,fs,path::Path};
type Ptr=*mut c_void;
unsafe extern "C" {
 fn laya_init(s:*mut Ptr)->i32;
 fn laya_alloc(p:*mut Ptr,n:usize)->i32;
 fn laya_free(p:Ptr)->i32;
 fn laya_upload(d:Ptr,s:*const u8,n:usize,st:Ptr)->i32;
 fn laya_download(d:*mut u8,s:Ptr,n:usize,st:Ptr)->i32;
 fn laya_sync(s:Ptr)->i32;
 fn laya_stream_free(s:Ptr)->i32;
 fn laya_capture_begin(s:Ptr)->i32;
 fn laya_capture_end(s:Ptr,g:*mut Ptr)->i32;
 fn laya_graph_run(g:Ptr,s:Ptr)->i32;
 fn laya_graph_free(g:Ptr)->i32;
 fn laya_qkv(p:*mut Ptr,b:i32,l:i32,m:i32,s:Ptr)->i32;
 fn laya_rope(p:*mut Ptr,b:i32,l:i32,m:i32,s:Ptr)->i32;
 fn laya_attn_full(p:*mut Ptr,b:i32,l:i32,m:i32,s:Ptr)->i32;
}
fn check(x:i32){assert_eq!(x,0,"CUDA status {x}");}
struct Buffer{p:Ptr,n:usize}
impl Buffer {
 fn new(bytes:&[u8],s:Ptr)->Self {let mut p=std::ptr::null_mut();unsafe{check(laya_alloc(&mut p,bytes.len()));check(laya_upload(p,bytes.as_ptr(),bytes.len(),s));check(laya_sync(s));} Self{p,n:bytes.len()}}
 fn equal(&self,expected:&[u8],s:Ptr){let mut b=vec![0;self.n];unsafe{check(laya_download(b.as_mut_ptr(),self.p,self.n,s));check(laya_sync(s));}assert_eq!(b.len(),expected.len());let n=b.iter().zip(expected).filter(|(a,b)|a!=b).count();if n>0 {fs::write("evidence/native-mismatch.bin",&b).unwrap();fs::write("evidence/reference-mismatch.bin",expected).unwrap();}assert_eq!(n,0,"{n} bytes differ");}
}
impl Drop for Buffer{fn drop(&mut self){unsafe{laya_free(self.p);}}}
fn main(){let a:Vec<_>=std::env::args().collect();let path=Path::new(&a[1]);let b:i32=a[2].parse().unwrap();let l:i32=a[3].parse().unwrap();let m=b*l;let mut s=std::ptr::null_mut();unsafe{check(laya_init(&mut s));}
 let read=|n:&str|fs::read(path.join(format!("{n}.bin"))).unwrap();
 let buf=|n:&str|Buffer::new(&read(n),s);
 {let y=buf("input");let w=buf("weight");let z=Buffer::new(&vec![0;3072*4],s);let q=buf("qkv");let cos=buf("cos");let sin=buf("sin");let lens=buf("lens");let out=Buffer::new(&vec![0;(m as usize)*1024*2],s);
 let mut qargs=[y.p,w.p,z.p,q.p];let mut rargs=[q.p,cos.p,sin.p];let mut aargs=[q.p,lens.p,out.p];
 unsafe{check(laya_qkv(qargs.as_mut_ptr(),b,l,m,s));}println!("checking qkv");q.equal(&read("qkv"),s);
 unsafe{check(laya_rope(rargs.as_mut_ptr(),b,l,m,s));}println!("checking rope");q.equal(&read("rotated"),s);
 unsafe{check(laya_attn_full(aargs.as_mut_ptr(),b,l,m,s));}println!("checking attention");out.equal(&read("attention"),s);
 let mut g=std::ptr::null_mut();unsafe{check(laya_capture_begin(s));check(laya_qkv(qargs.as_mut_ptr(),b,l,m,s));check(laya_rope(rargs.as_mut_ptr(),b,l,m,s));check(laya_attn_full(aargs.as_mut_ptr(),b,l,m,s));check(laya_capture_end(s,&mut g));}
 for _ in 0..5 {unsafe{check(laya_graph_run(g,s));}println!("checking rope");q.equal(&read("rotated"),s);println!("checking attention");out.equal(&read("attention"),s);}
 unsafe{check(laya_graph_free(g));}}
 unsafe{check(laya_stream_free(s));}println!("PASS B={b} L={l}: QKV, RoPE, attention bitwise; 5 graph replays; Rust-only runtime");}
