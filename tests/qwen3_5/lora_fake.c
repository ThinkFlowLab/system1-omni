// CPU implementation of the CUDA boundary used only by the LoRA dispatch test.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static int live, gemms, copies, ms[128], heights[128], base_calls, base_args[16][5];
int lora_fake_live(void) { return live; }
int lora_fake_gemms(void) { return gemms; }
int lora_fake_m(int i) { return ms[i]; }
int lora_fake_copies(void) { return copies; }
int lora_fake_height(int i) { return heights[i]; }
void lora_fake_reset(void) { gemms=0; copies=0; base_calls=0; }
uint32_t cs1_abi_version(void) { return LORA_FAKE_ABI; }
const char* cs1_error_string(int code) { (void)code; return "unexpected fake CUDA call"; }
int cs1_malloc(void** p, size_t n) { *p=malloc(n); if (!*p) return 1; ++live; return 0; }
int cs1_free(void* p) { --live; free(p); return 0; }
int cs1_stream_create(void** stream) { *stream=(void*)1; return 0; }
int cs1_stream_sync(void* stream) { (void)stream; return 0; }
int cs1_stream_destroy(void* stream) { (void)stream; return 0; }
int cs1_upload(void* dst, const void* src, size_t n, void* stream) { (void)stream; memcpy(dst,src,n); return 0; }
int cs1_download(void* dst, const void* src, size_t n, void* stream) { (void)stream; memcpy(dst,src,n); return 0; }
int cs1_copy2d(void* dst,size_t dpitch,const void* src,size_t spitch,size_t width,size_t height,void* stream) {
    (void)stream; heights[copies++]=(int)height;
    for (size_t row=0;row<height;++row) memcpy((char*)dst+row*dpitch,(const char*)src+row*spitch,width);
    return 0;
}
static float f32(uint16_t v) { uint32_t bits=(uint32_t)v<<16; float x; memcpy(&x,&bits,4); return x; }
static uint16_t bf16(float x) { uint32_t bits; memcpy(&bits,&x,4); return (uint16_t)((bits+0x7fff+((bits>>16)&1))>>16); }
int cs1_vision_to_float(const uint16_t* x,float* y,size_t n,void* stream) {
    (void)stream; for (size_t i=0;i<n;++i) y[i]=f32(x[i]); return 0;
}
int cs1_gemm_f32(void* gemm,const float* x,const float* w,float* y,int m,int n,int k,void* stream) {
    (void)gemm; (void)stream; ms[gemms++]=m;
    for(int row=0;row<m;++row) for(int out=0;out<n;++out) {
        float sum=0; for(int col=0;col<k;++col) sum+=x[row*k+col]*w[out*k+col]; y[row*n+out]=sum;
    }
    return 0;
}
int cs1_vision_lora_add(uint16_t* x,const float* delta,size_t n,float scale,void* stream) {
    (void)stream; for(size_t i=0;i<n;++i) x[i]=bf16(f32(x[i])+scale*delta[i]); return 0;
}

int lora_fake_base_calls(void) { return base_calls; }
int lora_fake_base_arg(int call,int argument) { return base_args[call][argument]; }
int cs1_gemm(void* gemm,const uint16_t* x,const uint16_t* w,uint16_t* y,int m,int n,int k,int ldy,void* stream) {
    (void)gemm; (void)stream;int call=base_calls++;
    base_args[call][0]=m;base_args[call][1]=n;base_args[call][2]=k;base_args[call][3]=ldy;base_args[call][4]=w[0];
    for(int row=0;row<m;++row)for(int out=0;out<n;++out) {
        float sum=0;for(int col=0;col<k;++col)sum+=f32(x[row*k+col])*f32(w[out*k+col]);y[row*ldy+out]=bf16(sum);
    }
    return 0;
}
