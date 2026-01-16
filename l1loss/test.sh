workspace="/home/HwHiAiUser/wangty/lab5/l1loss"

(cd CustomOp/build_out && bash custom_opp_ubuntu_aarch64.run)  # expect: Uncompressing version:1.0 100%

mkdir AclNNInvocation/input
mkdir AclNNInvocation/output
# modify 'rm' to 'rm -f' in AclNNInvocation/run.sh

bash AclNNInvocation/run.sh  # ex