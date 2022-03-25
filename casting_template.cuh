#ifndef CASTING_TEMPLATE_CUH
#define CASTING_TEMPLATE_CUH

#include <casting_kernel.cuh>

template <typename typeIn, typename typeOut>
__inline__ __device__
void convert_type(int m, const typeIn *dA, typeOut *dB)
{
  for ( int tid = threadIdx.x + blockIdx.x * blockDim.x;
      tid < m;
      tid += blockDim.x * gridDim.x )
  {
    convert ( dA[tid], dB + tid );
  }
}



template <typename typeIn, typename typeOut>
__global__
void convertType(int n, const typeIn *d_buf, typeOut *d_cbuf)
{
  convert_type<typeIn, typeOut> ( n, d_buf, d_cbuf );
}



/** \fn template <typename typeIn, typename typeOut>size_t castingTask( const void *buf,int n,void *cbuf,size_t cbuf_size,int chunk_count,dim3 grid,int nthread,volatile int *flag,size_t *cchunk_size,int batch_count,cudaStream_t stream)
 * \brief Iterates over the chunks and submits a kernel for each as well as raises a flag
 * \details This routine iterates over the chunks and for each, it submits
 * a kernel +  an extra kernel to raise a flag to notify that the associated
 * chunk has been processed.
 * \param buf             Buffer that contains the data
 * \param n               Number of data to convert
 * \param cbuf            Buffer that contains the converted data
 * \param cbuf_size       Size of the converted buffer [NOT USED]
 * \param chunk_count     Size of the chunk
 * \param grid            Grid used for the the kernels
 * \param nthread         Number of threads used for the kernels
 * \param flag            Location of the flag raised by the kernels
 * \param cchunk_size     Size of each chunks
 * \param batch_count      Number of chunks
 * \param stream          Cuda stream used for the scheduling of the kernels
 * \remarks
 * \warning
*/
template <typename typeIn, typename typeOut>
size_t castingTask( const void    *buf,
                    int           n,
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
  int disp = 0;

  //Treat the batch_count - 1 blocks
  for ( int i = 0; i < batch_count - 1; i++ ){
    convertType < typeIn, typeOut >
      <<< grid, nthread, 0, stream >>>
      ( chunk_count,
        ((const typeIn *)buf) + disp, ((typeOut*)cbuf) + disp 
      );
    // Raise a flag when converted
    if ( flag ) raiseFlag <<<1, 1, 0, stream >>> ( i, flag );

    disp += chunk_count;
    if ( cchunk_size ) 
      *cchunk_size = chunk_count * sizeof(typeOut);//TODO Change into cchunk_size[i]
  }

  //Treat the last block
  chunk_count = n - chunk_count * ( batch_count - 1 );

  convertType < typeIn, typeOut >
    <<< grid, nthread, 0, stream >>>
    ( chunk_count,
      ((const typeIn *)buf) + disp, ((typeOut*)cbuf) + disp );
  if ( flag ) raiseFlag <<<1, 1, 0, stream >>> ( batch_count - 1, flag );

  //XXX For now, there is a test on the batch_count, since the chunk_count is updated.
  // Without this test, the value would be changed for the first blocks.
  // This MUST disapear when cchunk_size is managed as an array
  if ( batch_count == 1 ) {
    if ( cchunk_size ) 
      *cchunk_size = chunk_count * sizeof(typeOut);//TODO Change into cchunk_size[i]
  }

  return n * sizeof(typeOut);
}

#endif //CASTING_TEMPLATE_CUH
