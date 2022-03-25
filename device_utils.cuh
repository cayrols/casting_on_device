#ifndef DEVICE_UTILS_CUH
#define DEVICE_UTILS_CUH

__global__
static void raiseFlag( const int val, volatile int *flag) {
  if ( threadIdx.x + blockIdx.x == 0 )
    *flag = val;
}

#endif
