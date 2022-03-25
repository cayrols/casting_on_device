#ifndef CASTING_KERNEL_CUH
#define CASTING_KERNEL_CUH

#include <cuda_fp16.h>

__inline__ __device__
void convert(const double a, __half *ca)
{
  *ca = __float2half ( float(a) );
}



__inline__ __device__
void convert(const __half a, double *ca)
{
  *ca = double( __half2float ( a ) );
}



__inline__ __device__
void convert(const __half a, float *ca)
{
  *ca = __half2float( a );
}



__inline__ __device__
void convert(const double a, float *ca)
{
  *ca = float( a );
}



__inline__ __device__
void convert(const float a, double *ca)
{
  *ca = double( a );
}



__inline__ __device__
void convert(const double a, double *ca)
{
  *ca = a;
}



__inline__ __device__
void convert(const float a, float *ca)
{
  *ca = a;
}

#endif //CASTING_KERNEL_CUH
