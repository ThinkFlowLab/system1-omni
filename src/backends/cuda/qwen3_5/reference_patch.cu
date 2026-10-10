#include <cuda_runtime.h>
// A shape-owned cuDNN plan: allocation and descriptor setup occur before graph capture.
#include "ops.h"
#include <new>
#ifdef CS1_REFERENCE_CUDNN
#include <cudnn.h>
namespace {
struct PatchPlan {
    cudnnHandle_t handle=nullptr;
    cudnnTensorDescriptor_t input=nullptr,output=nullptr;
    cudnnFilterDescriptor_t filter=nullptr;
    cudnnConvolutionDescriptor_t conv=nullptr;
    void* workspace=nullptr;
    size_t bytes=0;
    ~PatchPlan() {
        if(workspace) cudaFree(workspace);
        if(conv) cudnnDestroyConvolutionDescriptor(conv);
        if(filter) cudnnDestroyFilterDescriptor(filter);
        if(output) cudnnDestroyTensorDescriptor(output);
        if(input) cudnnDestroyTensorDescriptor(input);
        if(handle) cudnnDestroy(handle);
    }
};
}
#endif
extern "C" int cs1_reference_available() {
#ifdef CS1_REFERENCE_CUDNN
    return 1;
#else
    return 0;
#endif
}
extern "C" void* cs1_reference_patch_create(int rows,void* stream) {
#ifdef CS1_REFERENCE_CUDNN
    if(rows<=0 || rows>65536) return nullptr;
    PatchPlan* p=new(std::nothrow) PatchPlan;
    if(!p) return nullptr;
    bool ready=false;
    #define CHECK_CUDNN(expression) if((expression)!=CUDNN_STATUS_SUCCESS) break
    do {
        CHECK_CUDNN(cudnnCreate(&p->handle));
        CHECK_CUDNN(cudnnSetStream(p->handle,(cudaStream_t)stream));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&p->input));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&p->output));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&p->filter));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&p->conv));
        int xs[]={rows,3,2,16,16},xt[]={1536,512,256,16,1};
        int ys[]={rows,1152,1,1,1},yt[]={1152,1,1,1,1};
        int ws[]={1152,3,2,16,16},pad[]={0,0,0},step[]={2,16,16},dilation[]={1,1,1};
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(p->input,CUDNN_DATA_BFLOAT16,5,xs,xt));
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(p->output,CUDNN_DATA_BFLOAT16,5,ys,yt));
        CHECK_CUDNN(cudnnSetFilterNdDescriptor(p->filter,CUDNN_DATA_BFLOAT16,CUDNN_TENSOR_NCHW,5,ws));
        CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(p->conv,3,pad,step,dilation,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));
        CHECK_CUDNN(cudnnSetConvolutionMathType(p->conv,CUDNN_TENSOR_OP_MATH));
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(p->handle,p->input,p->filter,p->conv,p->output,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,&p->bytes));
        if(p->bytes && cudaMalloc(&p->workspace,p->bytes)!=cudaSuccess) break;
        ready=true;
    } while(false);
    #undef CHECK_CUDNN
    if(!ready) {delete p;return nullptr;}
    return p;
#else
    (void)rows;(void)stream;return nullptr;
#endif
}
extern "C" int cs1_reference_patch(void* plan,const void* x,const void* w,void* y) {
#ifdef CS1_REFERENCE_CUDNN
    auto* p=static_cast<PatchPlan*>(plan);
    if(!p || !x || !w || !y) return cudaErrorInvalidValue;
    const float alpha=1.f,beta=0.f;
    auto status=cudnnConvolutionForward(p->handle,&alpha,p->input,x,p->filter,w,p->conv,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,p->workspace,p->bytes,&beta,p->output,y);
    return status==CUDNN_STATUS_SUCCESS ? 0 : 20000+(int)status;
#else
    (void)plan;(void)x;(void)w;(void)y;return cudaErrorNotSupported;
#endif
}
extern "C" void cs1_reference_patch_destroy(void* plan) {
#ifdef CS1_REFERENCE_CUDNN
    delete static_cast<PatchPlan*>(plan);
#else
    (void)plan;
#endif
}
