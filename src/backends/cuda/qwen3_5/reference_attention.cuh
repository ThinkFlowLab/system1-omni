#pragma once
#include "common.cuh"
namespace cs1 { namespace reference {
inline int split_count(int tokens,int heads,int tile,int sms) {
    int blocks=(tokens+tile-1)/tile,workers=heads*((tokens+63)/64),splits=1;
    if(workers<0.8f*(sms*2)) {
        int maximum=min(min(128,sms*2),blocks);float best=0.f;
        for(int pass=0;pass<2;pass++) for(int n=1;n<=maximum;n++) {
            if(n>1 && (blocks+n-1)/n==(blocks+n-2)/(n-1)) continue;
            float waves=(float)(workers*n)/(sms*2),efficiency=waves/ceilf(waves);
            if(pass==0) best=fmaxf(best,efficiency);
            else if(efficiency>=0.85f*best) {splits=n;break;}
        }
    }
    return splits;
}
inline int sm_count() {
    int device=0,sms=0;
    if(cudaGetDevice(&device)!=cudaSuccess || cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device)!=cudaSuccess) return 0;
    return sms;
}
inline size_t workspace_floats(int capacity,int heads,int dim,int tile) {
    if(capacity<=0 || capacity>65536 || heads<=0 || heads>48) return 0;
    int sms=sm_count();if(sms<=0) return 0;
    size_t maximum=0;
    for(int end=64;end<capacity+64;end+=64) {
        int tokens=min(end,capacity);
        size_t count=(size_t)split_count(tokens,heads,tile,sms)*tokens*heads*(dim+1);
        maximum=maximum>count?maximum:count;
    }
    return maximum;
}
} }
