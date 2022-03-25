#include <stdio.h>
#include <cuda_runtime.h>

#include <cuda_fp16.h>

#include <casting_interface.h>
#include <device_utils.cuh>
#include <casting_template.cuh>


//From magma interface.cpp file, updated
int is_devPtr(const void *ptr)
{
  cudaError_t err;
  struct cudaPointerAttributes attr;

  err = cudaPointerGetAttributes( &attr, ptr);
  if ( ! err ){
    return (attr.type == cudaMemoryTypeDevice);
  }
  else if ( (err = cudaErrorInvalidValue) ) {
    cudaGetLastError();
    return 0;
  }
  //FIXME missing return is a reminder to fix it
}


#define TARGET_IN 0
#define caseBufTypeSize(_convType, _typeSizeIn, _typeSizeOut) \
                              case _convType:\
                                  return (target) ? _typeSizeOut : _typeSizeIn ;\
                                  break;
/** \fn size_t getCastingBufTypeSize(int conversion_type, int target)
 * \brief Returns the size of the conversion type used
 * \details This routine returns the size in bytes of the data either FROM or TO
 * \param conversion_type The type used to convert the data
 * \param target          Either 0:IN otherwise OUT
 * \remarks
 * \warning
*/
size_t getCastingBufTypeSize(int conversion_type, int target)
{
  switch ( conversion_type ){
    caseBufTypeSize(DOUBLE2HALF,                    sizeof(double), sizeof(__half))
    caseBufTypeSize(FLOAT2HALF,                     sizeof(float),  sizeof(__half))
    caseBufTypeSize(DOUBLE2FLOAT,                   sizeof(double), sizeof(float))
    caseBufTypeSize(HALF2DOUBLE,                    sizeof(__half), sizeof(double))
    caseBufTypeSize(HALF2FLOAT,                     sizeof(__half), sizeof(float))
    caseBufTypeSize(FLOAT2DOUBLE,                   sizeof(float),  sizeof(double))
    caseBufTypeSize(DOUBLE2HALF_CHUNKINPLACE,       sizeof(double), sizeof(__half))
    caseBufTypeSize(DOUBLE2HALF_INPLACE,            sizeof(double), sizeof(__half))
    caseBufTypeSize(HALF2DOUBLE_CHUNKINPLACE,       sizeof(__half), sizeof(double))
    caseBufTypeSize(HALF2DOUBLE_CHUNKINPLACEASYNC,  sizeof(__half), sizeof(double))
    caseBufTypeSize(HALF2DOUBLE_BLOCK,              sizeof(__half), sizeof(double))
    default:
      return 0;
  }
}



size_t estimateCastBufsize(uint64_t nval, size_t datatypesize, void *data)
{
  int castOp = *((int*)data);
  switch ( castOp ) {
    case FP64TOFP32 :
      return nval * sizeof(float);
      break;
    case FP64TOFP16 :
      //Fall to next case
    case FP32TOFP16 :
      return nval * sizeof(__half);
      break;
    case CFP64TOCFP32: 
      return nval * sizeof(float) * 2; //Trick since there is no such casting operation
      break;
    case CFP64TOCFP16: 
      //Fall to next case
    case CFP32TOCFP16: 
      return nval * sizeof(__half) * 2; //Trick since there is no such casting operation
      break;
    default:
      fprintf ( stderr, "Error, %d operation not implemented yet.\n", castOp );
      return 0;
  }
}



int getCastingType(int castOp, int block, int compression)
{
  switch ( castOp ) {
    case CFP64TOCFP32:
      if ( ! compression ) return FLOAT2DOUBLE;

      if ( block )  return DOUBLE2FLOAT_BLOCK;
      else          return DOUBLE2FLOAT;
      break;
    case CFP64TOCFP16:
      if ( ! compression ) return HALF2DOUBLE;

      if ( block )  return DOUBLE2HALF_BLOCK;
      else          return DOUBLE2HALF;
      break;
    case FP32TOFP16:
      if ( ! compression ) return HALF2FLOAT;

      return FLOAT2HALF; //TODO No block case for now
      break;
    default :
      fprintf ( stderr, "Unknown castingOp %d given as parameter\n", castOp );
  }
  return NOCASTING;
}



/** \fn size_t casting( const void *buf,int n,int casting_type,void *cbuf,size_t cbuf_size,int chunk_count,dim3 grid,int nthread,volatile int *flag,size_t *cchunk_size,int *batch_count,cudaStream_t stream)
 * \brief
 * \details
 * \param buf
 * \param n
 * \param casting_type
 * \param cbuf
 * \param cbuf_size
 * \param chunk_count
 * \param grid
 * \param nthread
 * \param flag
 * \param cchunk_size
 * \param batch_count
 * \param stream
 * \remarks Assume buf and cbuf are on the device
 * \warning
*/
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
                cudaStream_t  stream)
{
  switch ( casting_type ) {

    case DOUBLE2HALF:
      /* Fall to the next case */
    case DOUBLE2HALF_BLOCK:

      return castingTask <double, __half>( buf, n, cbuf, cbuf_size,
          chunk_count, grid, nthread, flag,
          cchunk_size, batch_count, stream );

    case HALF2DOUBLE:
      /* Fall to the next case */
    case HALF2DOUBLE_BLOCK:

      return castingTask <__half, double>( buf, n, cbuf, cbuf_size,
          chunk_count, grid, nthread, flag,
          cchunk_size, batch_count, stream );

    case DOUBLE2FLOAT:
      /* Fall to the next case */
    case DOUBLE2FLOAT_BLOCK:

      return castingTask <double, float>( buf, n, cbuf, cbuf_size,
          chunk_count, grid, nthread, flag,
          cchunk_size, batch_count, stream );

    case FLOAT2DOUBLE:
      /* Fall to the next case */
    case FLOAT2DOUBLE_BLOCK:

      return castingTask <float, double>( buf, n, cbuf, cbuf_size,
          chunk_count, grid, nthread, flag,
          cchunk_size, batch_count, stream );

    default:
      return 0;
  }
}



/** \fn size_t casting_iface( const void *buf,uint64_t n,int batch_count,size_t *cchunk_size,void *cbuf,size_t cbuf_size,void *param,volatile int *flag,cudaStream_t stream)
 * \brief Casts the input buffer buf and stores the result in cbuf
 * \details This routine checks the input buffer is on the device and then
 * calls the casting routine.
 * \param buf             The input buffer where the data to cast are stored
 * \param buf_size        The size of the input buffer
 * \param n               The number of data to cast
 * \param cchunk_size     The size of each casted chunk, its size is max_batch_count
 * \param max_batch_count The maximum number of chunks in cchunk_size array
 * \param cbuf            The output buffer that stores the result of the cast
 * \param cbuf_size       The size of the output buffer
 * \param param           The parameters used of the casting operation
 * \param flag            The memory where the flag is raised
 * \param batch_count     The number of chunks created
 * \param stream          The cuda stream used for the casting
 * \remarks 
 * \warning - The buffers HAVE TO be on the device
 *          - The batch_count may be greater than max_batch_count, therefore
 * a loop is needed, and the flag should not be reused since the value written
 * in it is in the range int([0, min(nchunk, max_batch_count)[)
*/
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
                      cudaStream_t  stream)
{
  int nchunk = 0;
  size_t outputMemsize = 0;
  CastingParam_t *cparam = (CastingParam_t*)param;

  /*
     Check if memory already on GPU otherwise raise a message and return
  */
  int bufOnGPU = is_devPtr( buf );
  if ( bufOnGPU == -1 ){
    fprintf(stderr, "Unable to check whether buf is on DEVICE.\n");
    return 0;
  }

  if ( ! bufOnGPU ){
    fprintf(stderr, "Error, d_buf is on HOST.\n");
    return 0;
  }

  // Ensure that we have enough space in cchunk_size
  nchunk = ( n + cparam->chunk_count - 1 ) / cparam->chunk_count;
  *batch_count = ( nchunk > max_batch_count ? max_batch_count : nchunk);

  outputMemsize = casting( buf, n, cparam->casting_type, cbuf, cbuf_size,
      cparam->chunk_count, cparam->grid, cparam->nthread,
      flag, cchunk_size, *batch_count,
      stream );

  return outputMemsize;
}
