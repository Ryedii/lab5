/**
 * @file l1loss_custom.cpp
 *
 * Copyright (C) 2022-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelL1loss {
public:
    __aicore__ inline KernelL1loss() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        
        // TODO set global buffer, 
        //   write similar codes for y and z
        // xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // TODO init buffer 
        // pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
    }
    __aicore__ inline void Process()
    {
        /// TODO 
        // int32_t loopCount = ??? ;
        
        for (int32_t i = 0; i < loopCount; i++) {
            // TODO each loop do ???
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // TODO alloc local tensors from inputQueue
        //      give the codes for yLocal
        // AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();

        // TODO copy from global memory to local tensor
        // AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);

        // TODO enqueue local tensors to input-queue 
        //      note: in AscendC, queue is used to sync between CopyIn/Compute/CopyOut
        //      API like this
        // inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // TODO 
        // deque input tensors like this, alloc output tensor from output-queue
        // AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();

        // TODO find related AscendC compute API to do L1loss operation

        // TODO enqueue output tensor, free input tensors
        // inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO alloc output tensor from output-queue, copy to global memory

        // TODO free output tensor
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void l1loss_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // create Kernel, call init and process methods
    KernelL1loss op;
    // TODO launch kernel
    // op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    // op.Process();
}

// this api is used for CPU debugging, 
#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void l1loss_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y, uint8_t *z,
                   uint8_t *workspace, uint8_t *tiling)
{
    l1loss_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, workspace, tiling);
}
#endif