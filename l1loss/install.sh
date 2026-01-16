workspace="/home/HwHiAiUser/wangty/lab5/l1loss"
OP_NAME=L1lossCustom

# Write op_host and op_kernel
cd $workspace/$OP_NAME/op_host
cd $workspace/$OP_NAME/op_kernel

cd $workspace
rm -rf CustomOp

# # Resolve some permission issues
# chmod 600 L1lossCustom.json
# chmod 700 .
# sudo chmod 755 /usr/local/Ascend/ascend-toolkit/latest/opp
# sudo chown -R HwHiAiUser:HwHiAiUser /usr/local/Ascend/ascend-toolkit/latest/opp/vendors

# Generate the op framework
msopgen gen -i $OP_NAME.json -c ai_core-Ascend310B1 -lan cpp -out CustomOp
# Copy op implementation files to CustomOp
cp -rf $OP_NAME/* CustomOp
# Build CustomOp project
(cd CustomOp && bash build.sh)  # expect: xxx/build_out/custom_opp_ubuntu_aarch64.run generated
(cd CustomOp/build_out && bash custom_opp_ubuntu_aarch64.run)