/**
 * @file l1loss_custom_tiling.h
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef L1LOSS_CUSTOM_TILING_H
#define L1LOSS_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)

// TODO
// user defined tiling parameters
// TILING_DATA_FIELD_DEF(uint32_t, totalLength);
// TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

// Register the TilingData class
REGISTER_TILING_DATA_CLASS(L1lossCustom, TilingData)
} // namespace optiling
#endif // L1LOSS_CUSTOM_TILING_H