## 实验指导 exp5.pdf

根据实验指导文件 **exp5.pdf**，本次实验（实验五）的主题是：

> **AscendC 算子实现**

---

### 一、实验概览

本实验基于 **华为昇腾（Ascend）AI 处理器架构**，使用 **AscendC 编程范式**，在 **香橙派 AI Pro 开发板**（搭载昇腾 310B NPU）上开发并部署三个自定义算子：

1. **L1Loss**（向量级 element-wise 算子）
2. **Softplus**（向量级激活函数算子）
3. **Matmul**（矩阵乘法 + 偏置）

目标是掌握从算子设计、编码、编译、安装到测试的完整流程，并深入理解昇腾 NPU 上的异构计算机制。

---

### 二、实验目的

- 理解算子在昇腾芯片上的执行原理：包括 **数据搬运、内存管理、AI Core 计算、同步机制**。
- 掌握 **AscendC 算子开发的标准流程**：
  - 使用 `msopgen` 工具生成工程模板
  - 实现 Host 侧（控制逻辑）和 Kernel 侧（计算逻辑）
  - 编译、打包、安装算子
  - 通过 AscendC 调用并验证功能
- 初步了解 **端侧大模型部署中的算子优化策略**（如针对内存/算力受限的优化）。

---

### 三、三个算子的具体要求

#### 1. **L1Loss 算子**
- 功能：对两个输入张量逐元素计算绝对值差：`|x - y|`
- **不进行 reduce（如求平均）**，仅输出每个元素的 loss。
- 提供了部分代码模板（需替换 `msopgen` 生成的文件），需完成 `op_host` 和 `op_kernel` 中标记为 `TODO` 的部分。

#### 2. **Softplus 算子**
- 公式：`Softplus(x) = log(1 + exp(x))`（设定 β=1）
- 也是 element-wise 算子。
- 输入输出形状固定为 `(8, 2048)`，因此 **tiling 参数可硬编码在 kernel 中**，无需通过 host 传递。
- **无提供模板代码**，需完全自行实现 kernel 逻辑。

#### 3. **Matmul 算子**
- 公式：`C = A * B + bias`
  - `A`: [M, K]
  - `B`: [K, N]
  - `bias`: [1, N]（广播加到每一行）
- 需使用 **AscendC 高阶 API**（因香橙派硬件限制，不能用基础 API）。
- 提供了部分模板代码（`MyMatmul/*`），需完成 host 和 kernel 中的 `TODO`。

---

### 四、通用开发流程（每个算子都需完成）

1. **创建工程**  
   ```bash
   msopgen gen -i xxx.json -c ai_core -chip Ascend310B -lan cpp -out CustomOp
   ```

2. **实现算子**
   - `op_host/`：处理 shape 推导、tiling 参数设置
   - `op_kernel/`：实现核心计算、数据搬移（使用 AscendC API）

3. **编译安装**
   ```bash
   bash build.sh
   bash custom_opp_ubuntu_aarch64.run  # 安装算子包
   ```

4. **测试验证**
   - 进入对应目录下的 `AclNNInvocation/`
   - 执行 `./run.sh`
   - 若输出 “test pass”，则算子功能正确

---

### 五、评分标准

| 得分 | 要求 |
|------|------|
| 50   | 成功实现 **1 个** 算子并通过测试 |
| 75   | 成功实现 **2 个** 算子并通过测试 |
| 85   | 成功实现 **全部 3 个** 算子并通过测试 |
| <50  | 代码无法运行，人工审计评分 |

此外，**实验报告与思考题占 15%**，包括：
- 如何将自定义算子集成到 MindSpore/CANN？
- 了解“算子融合”及其优化价值
- 分析自己算子的优化空间

---

### 六、关键注意事项

- **CANN 版本必须为 8.0.0**，API 可能与其他版本不兼容。
- 建议以 **root 用户** 操作，避免权限问题。
- 参考官方 AscendC 文档及 Gitee 上的 `add`、`matmul` 示例。
- 所有算子均在 **香橙派 AI Pro（Ascend 310B）** 上运行。


## L1Loss

### 一、开发流程

```bash
# Enter workspace
workspace="/home/HwHiAiUser/wangty/lab5/l1loss"

# Write op_host
cd $workspace/L1lossCustom/op_host

# Write op_kernel
cd $workspace/L1lossCustom/op_kernel

# Something should not be written by user group or others, which will cause security risks
cd $workspace
chmod 600 L1lossCustom.json
chmod 700 .

# Build Operator
cd $workspace
msopgen gen -i L1lossCustom.json -c ai_core-Ascend310B1 -lan cpp -out CustomOp
cp -rf L1lossCustom/* CustomOp

cd $workspace/CustomOp
export ASCEND_CUSTOM_OPP_PATH=$HOME/my_opp
bash build.sh  # expect: xxx/build_out/custom_opp_ubuntu_aarch64.run generated

cd $workspace/CustomOp/build_out
bash custom_opp_ubuntu_aarch64.run  # expect: Uncompressing version:1.0 100%

# Test Operator
cd $workspace
bash AclNNInvocation/run.sh  # expect: test pass
```

二、