#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template <typename aType, typename bType, typename cType, typename biasType> class MatmulKernel {
public:
    __aicore__ inline MatmulKernel(){};

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling) {
        this->tiling = tiling;
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
        bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
        cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
        biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);
    
        int32_t offsetA = 0;
        int32_t offsetB = 0;
        int32_t offsetC = 0;
        int32_t offsetBias = 0;
        CalcOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias);
        aGlobal = aGlobal[offsetA];
        bGlobal = bGlobal[offsetB];
        cGlobal = cGlobal[offsetC];
        biasGlobal = biasGlobal[offsetBias];
        if (GetSysWorkSpacePtr() == nullptr) {
            return;
        }
    }

    template <bool setTmpSpace = false> __aicore__ inline void Process(AscendC::TPipe *pipe) {
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
    
        matmulObj.SetTensorA(aGlobal);
        matmulObj.SetTensorB(bGlobal);
        matmulObj.SetBias(biasGlobal);
        matmulObj.IterateAll(cGlobal);
        matmulObj.End();
    }
    
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB, int32_t &offsetC, int32_t &offsetBias) {
        auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
        auto mCoreIndx = blockIdx % mSingleBlocks;
        auto nCoreIndx = blockIdx / mSingleBlocks;
    
        offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
        offsetB = nCoreIndx * tiling.singleCoreN;
        offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
        offsetBias = nCoreIndx * tiling.singleCoreN;
    }
    
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>> matmulObj;
    
    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    AscendC::GlobalTensor<biasType> biasGlobal;
    TCubeTiling tiling;
};

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, 
                                                    GM_ADDR workspace,
                                                    GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    MatmulKernel<half, half, float, float> matmulKernel;
    AscendC::TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tilingData.cubeTilingData);
    matmulKernel.Init(a, b, bias, c, workspace, tilingData.cubeTilingData);
    if (TILING_KEY_IS(1)) {
        matmulKernel.Process(&pipe);
    } else if (TILING_KEY_IS(2)) {
        matmulKernel.Process<true>(&pipe);
    }
}