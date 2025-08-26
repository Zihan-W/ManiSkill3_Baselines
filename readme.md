# 下载ManiSkill的官方demo数据集
python -m mani_skill.utils.download_demo ${ENV_ID}
# python -m mani_skill.utils.download_demo # with no args this prints all available datasets

# 3. 使用ManiSkill的官方例程Replay并转换数据集
# Replay demonstrations with control_mode=pd_joint_delta_pos
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos/StackCube-v1/motionplanning/trajectory.h5 \
  --save-traj --target-control-mode pd_joint_delta_pos \
  --obs-mode rgb --count 100