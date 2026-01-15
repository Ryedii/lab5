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
    // TODO

    // create global tensors representing input and output matrices
    AscendC::GlobalTensor<half> aGlobal;
    AscendC::GlobalTensor<half> bGlobal;
    AscendC::GlobalTensor<float> cGlobal;
    AscendC::GlobalTensor<float> biasGlobal;
    
    // set global, 
    // hint use `SetGlobalBuffer`
    // TODO

    // Register matmul object 
    // TODO

    // Set Tensors of matmul obj
    // TODO
    
    // launch matmul iteration, 
    // TODO

    // end matmul
    // TODO
}