 #include "kernel_operator.h"
 constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
 class KernelSoftplus {
 public:
     __aicore__ inline KernelSoftplus() {}
     __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
     {
         this->blockLength = totalLength / AscendC::GetBlockNum();
         this->tileNum = tileNum;
         this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
 
         xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
         zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
         pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
         pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
         pipe.InitBuffer(expTmpBuffer, this->tileLength * sizeof(DTYPE_X));
         pipe.InitBuffer(addTmpBuffer, this->tileLength * sizeof(DTYPE_X));
     }
     __aicore__ inline void Process()
     {
         int32_t loopCount = this->tileNum * BUFFER_NUM;   
         for (int32_t i = 0; i < loopCount; i++) {
             CopyIn(i);
             Compute(i);
             CopyOut(i);
         }
     }
 
 private:
     __aicore__ inline void CopyIn(int32_t progress)
     {
         AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
         AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
         inQueueX.EnQue(xLocal);
     }
     __aicore__ inline void Compute(int32_t progress)
     {
         AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
         AscendC::LocalTensor<DTYPE_X> expLocal = expTmpBuffer.Get<DTYPE_X>();
         AscendC::Exp(expLocal, xLocal, this->tileLength);

         AscendC::LocalTensor<DTYPE_X> addLocal = addTmpBuffer.Get<DTYPE_X>();
         const DTYPE_X one = 1.0f;
         AscendC::Adds(addLocal, expLocal, one, this->tileLength);

         AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
         AscendC::Ln(zLocal, addLocal, this->tileLength);

         outQueueZ.EnQue<DTYPE_Z>(zLocal);
         inQueueX.FreeTensor(xLocal);
     }
     __aicore__ inline void CopyOut(int32_t progress)
     {
 
         AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
         AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
         outQueueZ.FreeTensor(zLocal);
     }
 
 private:
     AscendC::TPipe pipe;
     AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
     AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
     AscendC::TBuf<AscendC::QuePosition::VECCALC> expTmpBuffer, addTmpBuffer;
     AscendC::GlobalTensor<DTYPE_X> xGm;
     AscendC::GlobalTensor<DTYPE_Z> zGm;
     uint32_t blockLength;
     uint32_t tileNum;
     uint32_t tileLength;
 };
 
 extern "C" __global__ __aicore__ void softplus_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
     GET_TILING_DATA(tiling_data, tiling);
     KernelSoftplus op;
     op.Init(x, z, tiling_data.totalLength, tiling_data.tileNum);
     op.Process();
 }
 
 #ifndef ASCENDC_CPU_DEBUG
 // call of kernel function
 void softplus_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *z,
                    uint8_t *workspace, uint8_t *tiling) {
     softplus_custom<<<blockDim, l2ctrl, stream>>>(x, z, workspace, tiling);
 }
 #endif