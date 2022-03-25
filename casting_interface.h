#ifndef CASTING_INTERFACE_C
#define CASTING_INTERFACE_C

#include <cuda_runtime.h>
#include <stdint.h>

enum{
  NOCASTING=0,
  DOUBLE2HALF,
  FLOAT2HALF,//Not implemented
  DOUBLE2FLOAT,
  HALF2DOUBLE,
  HALF2FLOAT,
  FLOAT2DOUBLE,
  DOUBLE2HALF_CHUNKINPLACE,
  DOUBLE2HALF_INPLACE,
  DOUBLE2HALF_BLOCK,
  HALF2DOUBLE_CHUNKINPLACE,
  HALF2DOUBLE_CHUNKINPLACEASYNC,
  HALF2DOUBLE_BLOCK,
  DOUBLE2FLOAT_BLOCK,
  FLOAT2DOUBLE_BLOCK,
  CASTING_TEST,
};

enum { 
  FP64TOFP32=1,
  FP64TOFP16,
  FP32TOFP16,
  CFP64TOCFP32,
  CFP64TOCFP16,
  CFP32TOCFP16,
};

#ifdef __cplusplus
extern "C" {
#endif


typedef struct{
  int   chunk_count;
  int   nthread;
  dim3  grid;
  int   casting_type;
  int   by_block;
} CastingParam_t;

#define CastingParamNULL() {\
  .chunk_count=0,           \
  .nthread=1024,            \
  .grid={1,1,1},            \
  .casting_type=NOCASTING,  \
  .by_block=0               \
};

size_t getCastingBufTypeSize(int conversion_type, int target);

size_t estimateCastBufsize(uint64_t nval, size_t datatypesize, void *data);

int getCastingType(int castOp, int block, int compression);

size_t casting( const void    *buf,
                int           n,
                int           casting_type,
                void          *cbuf,
                size_t        cbuf_size,
                int           chunk_count,
                dim3          grid,
                int           nthread,
                volatile int  *flag,
                size_t        *cchunk_size,
                int           batch_count,
                cudaStream_t  stream);

size_t casting_iface( const void    *buf,
                      size_t        buf_size,
                      uint64_t      n,
                      size_t        *cchunk_size,
                      int           max_batch_count,
                      void          *cbuf,
                      size_t        cbuf_size,
                      void          *param,
                      volatile int  *flag,
                      int           *batch_count,
                      cudaStream_t  stream);

#ifdef __cplusplus
}
#endif

#endif
