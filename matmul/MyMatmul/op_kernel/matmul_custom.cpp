#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace matmul;

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, 
                                                    GM_ADDR workspace,
                                                    GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    const TCubeTiling &cubeTiling = tilingData.cubeTilingData; 

    AscendC::TPipe pipe;

    // create matmul object
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>>
        matmulObj;

    // create global tensors representing input and output matrices
    AscendC::GlobalTensor<half> aGlobal;
    AscendC::GlobalTensor<half> bGlobal;
    AscendC::GlobalTensor<float> cGlobal;
    AscendC::GlobalTensor<float> biasGlobal;
    
    // set global, 
    // hint use `SetGlobalBuffer`
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);

    int32_t offsetA = 0;
    int32_t offsetB = 0;
    int32_t offsetC = 0;
    int32_t offsetBias = 0;

    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    offsetBias = nCoreIndx * tiling.singleCoreN;

    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    
    if (GetBlockIdx() >= 1) {
        return;
    }
    if constexpr (setTmpSpace) {
        AscendC::TBuf<> tmpMMFormatUb;
        AscendC::LocalTensor<uint8_t> mmformatUb;
        pipe->InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
        mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
        matmulObj.SetLocalWorkspace(mmformatUb);
    }

    // Register matmul object 
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, &tilingData.cubeTilingData);

    // Set Tensors of matmul obj
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);

    // launch matmul iteration, 
    matmulObj.IterateAll(cGlobal);

    // end matmul
    matmulObj.End();
}