import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import torch.nn.functional as F

try:
    from .VAE import VAE
    from .VAE import SmallCNN
    from .VAE import make_loader
except ImportError:
    from VAE import VAE
    from VAE import SmallCNN
    from VAE import make_loader



def load_VAE_reward_model(img, act, model_path, device):
    # 配置参数
    # StackCube-v1 motionplanning 数据集
    # dataset_path = "/home/wzh-2004/Downloads/trajectory.h5"   # 数据集路径
    dataset_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/StackCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"   # 数据集路径
    # StackCube-v1 teleop 数据集
    # dataset_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/StackCube-v1/demos/StackCube-v1/teleop/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"   # 数据集路径
    # StackCube-v1 teleop model权重
    model_path   = "/home/wzh-2004/ManiSkill3_Baselines/ms3_baselines/VAE/ckpt_VAE/20251028-182808/vae_best.pt" if model_path is None else model_path

    # PushCube-v1 motionplanning 数据集
    # dataset_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/PushCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"   # 数据集路径
    # model_path = "/home/wzh-2004/ManiSkill3_Baselines/ms3_baselines/VAE/ckpt_VAE/test/vae_best.pt"

    latent_dim   = 32                          # VAE 潜在空间维度
    max_action   = 1.0                         # 动作最大值 (动作归一化后)
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    print(f"Using device: {device}")

    if img is None or act is None:
        # 创建数据集和数据加载器
        ds, dl  = make_loader(dataset_path)
        # 取出一条样本，用于初始化网络
        img, act = ds[0]         # 第 0 条样本 (img: (2C,H,W), act: (A,))
        # 创建 VAEReward 模型
        vae_reward = VAEReward(img, act, latent_dim, max_action, model_path, device)
        return vae_reward, ds
    else:
        vae_reward = VAEReward(img, act, latent_dim, max_action, model_path, device)
        return vae_reward

def main():

    vae, ds = load_VAE_reward_model(None,None, model_path=None, device=None)

    # # 如果需要在验证集上计算奖励，取消注释以下代码
    # # 取验证集
    # val_ratio = 0.1  # 10% 做验证
    # N = len(ds)
    # idxs = np.arange(N)
    # np.random.shuffle(idxs)
    # val_size = max(1, int(N * val_ratio))
    # val_idxs = idxs[:val_size].tolist()
    # train_idxs = idxs[val_size:].tolist()

    # val_ds   = Subset(ds, val_idxs)

    # val_dl   = DataLoader(val_ds, batch_size=128, shuffle=False,
    #                     num_workers=8, pin_memory=True,
    #                     persistent_workers=True, prefetch_factor=4)

    # for batch in val_dl:
    #     imgs, acts = batch
    #     rewards = vae.compute_reward(imgs, acts)
    #     print("Rewards:", rewards)
    # print("Finished computing rewards on validation set.")


    # 在单条轨迹上计算奖励
    traj_id = 0
    traj_key = f"traj_{traj_id}"

    # 找到该 traj 的所有 dataset 索引（按时间顺序）
    traj_indices = [i for i, (k, t) in enumerate(ds.indices) if k == traj_key]
    # 有必要按 t 排序（通常 ds.indices 已按 traj 内顺序构建）
    traj_indices = sorted(traj_indices, key=lambda i: ds.indices[i][1])

    # -------- 按轨迹顺序分批计算 reward --------
    import matplotlib.pyplot as plt

    batch_size = len(traj_indices) + 1  # 根据显存调整
    recon_rewards_list = []
    for start in range(0, len(traj_indices), batch_size):
        batch_idxs = traj_indices[start:start + batch_size]
        # 从 dataset 读取并堆叠为 batch
        imgs = torch.stack([ds[i][0] for i in batch_idxs], dim=0)   # (B, 2C, H, W)
        acts = torch.stack([ds[i][1] for i in batch_idxs], dim=0)   # (B, A)
        # 计算 reward（VAEReward.compute_reward 返回 numpy array (B,)）
        batch_recon_rewards = vae.compute_reward(imgs, acts)
        recon_rewards_list.append(batch_recon_rewards)
    print("Recon rewards:", recon_rewards_list)

    # 合并并保持时间顺序
    if len(recon_rewards_list) == 0:
        print(f"No rewards computed for {traj_key}.")
    else:
        recon_rewards = np.concatenate(recon_rewards_list, axis=0)  # (T,)
        timesteps = np.arange(len(recon_rewards))

        # 绘图
        plt.figure(figsize=(12, 5))
        # plt.subplot(2, 1, 1)
        plt.plot(timesteps, recon_rewards, marker='o', linestyle='-', color='b')
        plt.xlabel("Timestep")
        plt.ylabel("Distance Reward")
        plt.title(f"Distance Rewards for {traj_key}")
        plt.grid(True)

        os.makedirs("reward_viz", exist_ok=True)

        plt.savefig("reward_viz/curve.png", dpi=150)
        plt.show()

    print("Finished computing rewards for single trajectory.")

    return None

if __name__ == "__main__":
    main()
