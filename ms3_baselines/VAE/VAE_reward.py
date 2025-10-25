import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import torch.nn.functional as F

try:
    from .VAE import VAE
    from .VAE import FeatureExtractor
    from .VAE import make_loader
except ImportError:
    from VAE import VAE
    from VAE import FeatureExtractor
    from VAE import make_loader

class VAEReward:
    def __init__(self, img, act, prop, latent_dim, max_action, model_path, device):
        self.device = device

        self.feature_net = FeatureExtractor(img, prop).to(device)
        self.state_dim = self.feature_net.feat_dim
        self.action_dim = act.shape[0]
        self.prop_dim = self.feature_net.prop_dim
        self.latent_dim = latent_dim
        self.max_action = max_action

        self.vae = VAE(state_dim=self.state_dim,
                       action_dim=self.action_dim,
                       latent_dim=self.latent_dim,
                       max_action=self.max_action,
                       device=device,
            ).to(device)


        _sd = torch.load(model_path, map_location=device)

        _feature_net_sd = _sd['feature_net']
        _vae_sd = _sd['vae']
        _act_mean = _sd['act_mean']
        _act_std  = _sd['act_std']

        self.feature_net.load_state_dict(_feature_net_sd)
        self.vae.load_state_dict(_vae_sd)
        self.act_mean = _act_mean.to(device)
        self.act_std  = _act_std.to(device)
        self.act_mean.requires_grad_(False)
        self.act_std.requires_grad_(False)

        self.feature_net.eval()
        self.vae.eval()

    def compute_reward(self, img, action, prop):
        """
        计算 VAE 重构误差作为奖励
        img: (B, 2C, H, W) 或 (B, H, W, 2C) 或 (2C, H, W) 等多种形式，支持 uint8/0-255 或 已归一化 float
        action: (B, A) 归一化动作
        返回: (B,) 奖励值，重构误差的负值
        """
        with torch.no_grad():
            # --- 确保为 torch.Tensor 并处理可能的单样本（无 batch 维） ---
            if not isinstance(img, torch.Tensor):
                img = torch.as_tensor(img)
            if img.ndim == 3:
                # (C,H,W) 或 (H,W,C) -> 加 batch
                img = img.unsqueeze(0)

            # --- 如果是 channel-last（B,H,W,C），把它变成 (B,C,H,W) ---
            if img.ndim == 4:
                # 若最后一维看起来像通道数（常见 1,2,3,4,6 等），而第二维不是通道，则认为是 HWC 格式
                last_dim = img.shape[-1]
                second_dim = img.shape[1]
                if last_dim in (1, 2, 3, 4, 6) and second_dim not in (1, 2, 3, 4, 6):
                    img = img.permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W

            # --- 确保为浮点并归一化到 [0,1]（如果原始为 0-255） ---
            if not torch.is_floating_point(img):
                img = img.float()
            # 如果像素值明显超过 1（如 255），则认为未归一化，执行 /255.0
            try:
                if img.max() > 1.1 or img.min() < 0.0:
                    img = img / 255.0
            except Exception:
                # 安全兜底（若 img 在 GPU 上或其他异常），先把数据移至 cpu 计算范围再恢复
                tmp = img.cpu().float()
                if tmp.max() > 1.1 or tmp.min() < 0.0:
                    img = tmp / 255.0
                else:
                    img = tmp
                # 后面会把 img 移到 device

            # 移动到设备并通过特征网络
            _img = img.to(self.device)
            _prop = prop.to(self.device)
            feat = self.feature_net(_img, _prop)  # 提取特征

            # --- 处理 action 张量格式 ---
            if not isinstance(action, torch.Tensor):
                action = torch.as_tensor(action)
            if action.ndim == 1:
                action = action.unsqueeze(0)
            action = action.to(self.device).float()
            action_norm = (action - self.act_mean) / (self.act_std + 1e-12)

            # 编码并用后验均值重构
            z_in = self.vae.encode(feat, action_norm)
            mu = self.vae.mean(z_in)
            recon = self.vae.decode(feat, mu)

            # 计算重构奖励
            d = ((recon - action_norm) ** 2).mean(dim=1)  # (B,)
            distance_reward = -d  # 奖励是重构误差的负值

            # 计算方向奖励（每个样本）
            a_direction = self.vae.dir_head(feat)  # (B, A) 动作方向
            a_direction = F.normalize(a_direction, dim=1, eps=1e-8)
            a_unit = F.normalize(action_norm, dim=1, eps=1e-8)
            cos_sim = torch.sum(a_direction * a_unit, dim=1)  # (B,)
            direction_reward = cos_sim  # (B,) 最大化余弦相似度（使方向一致）

        return distance_reward.cpu().numpy(), direction_reward.cpu().numpy()
        # return (recon_reward + direction_reward).cpu().numpy()

    def compute_penalty(self, direction_rewards, time_penalty_coef=0.01):
        time_penalty = - time_penalty_coef * torch.ones_like(direction_rewards)
        return time_penalty

def load_VAE_reward_model(img, act, prop, model_path, device):
    # 配置参数
    # StackCube-v1 motionplanning 数据集
    # dataset_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/StackCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"   # 数据集路径
    # StackCube-v1 teleop 数据集
    # dataset_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/StackCube-v1/demos/StackCube-v1/teleop/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"   # 数据集路径
    # StackCube-v1 teleop model权重
    # model_path   = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/ManiSkill3_Baselines/ms3_baselines/VAE/ckpt_VAE/stack_cube_version1/vae_best.pt" if model_path is None else model_path

    # PushCube-v1 motionplanning 数据集
    dataset_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/PushCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"   # 数据集路径
    model_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/ManiSkill3_Baselines/ms3_baselines/VAE/ckpt_VAE/push_cube_version5/vae_best.pt"

    latent_dim   = 32                          # VAE 潜在空间维度
    max_action   = 1.0                         # 动作最大值 (动作归一化后)
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    print(f"Using device: {device}")

    if img is None or act is None or prop is None:
        # 创建数据集和数据加载器
        ds, dl  = make_loader(dataset_path)
        # 取出一条样本，用于初始化网络
        img, act, prop = ds[0]         # 第 0 条样本 (img: (2C,H,W), act: (A,), prop: (P,))
        # 创建 VAEReward 模型
        vae_reward = VAEReward(img, act, prop, latent_dim, max_action, model_path, device)
        return vae_reward, ds
    else:
        vae_reward = VAEReward(img, act, prop, latent_dim, max_action, model_path, device)
        return vae_reward

def main():

    vae, ds = load_VAE_reward_model(None, None, None, model_path=None, device=None)

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
    traj_id = 99
    traj_key = f"traj_{traj_id}"

    # 找到该 traj 的所有 dataset 索引（按时间顺序）
    traj_indices = [i for i, (k, t) in enumerate(ds.indices) if k == traj_key]
    # 有必要按 t 排序（通常 ds.indices 已按 traj 内顺序构建）
    traj_indices = sorted(traj_indices, key=lambda i: ds.indices[i][1])

    # -------- 按轨迹顺序分批计算 reward --------
    import matplotlib.pyplot as plt

    batch_size = len(traj_indices) + 1  # 根据显存调整
    recon_rewards_list = []
    direction_rewards_list = []
    for start in range(0, len(traj_indices), batch_size):
        batch_idxs = traj_indices[start:start + batch_size]
        # 从 dataset 读取并堆叠为 batch
        imgs = torch.stack([ds[i][0] for i in batch_idxs], dim=0)   # (B, 2C, H, W)
        acts = torch.stack([ds[i][1] for i in batch_idxs], dim=0)   # (B, A)
        props = torch.stack([ds[i][2] for i in batch_idxs], dim=0)   # (B, P)
        # 计算 reward（VAEReward.compute_reward 返回 numpy array (B,)）
        batch_recon_rewards, batch_direction_rewards = vae.compute_reward(imgs, acts, props)
        recon_rewards_list.append(batch_recon_rewards)
        direction_rewards_list.append(batch_direction_rewards)
    print("Recon rewards:", recon_rewards_list)
    print("Direction rewards:", direction_rewards_list)

    # 合并并保持时间顺序
    if len(recon_rewards_list) == 0:
        print(f"No rewards computed for {traj_key}.")
    else:
        recon_rewards = np.concatenate(recon_rewards_list, axis=0)  # (T,)
        direction_rewards = np.concatenate(direction_rewards_list, axis=0)  # (T,)
        timesteps = np.arange(len(recon_rewards))

        # 绘图
        plt.figure(figsize=(12, 5))
        plt.subplot(2, 1, 1)
        plt.plot(timesteps, recon_rewards, marker='o', linestyle='-', color='b')
        plt.xlabel("Timestep")
        plt.ylabel("Distance Reward")
        plt.title(f"Distance Rewards for {traj_key}")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(timesteps, direction_rewards, marker='o', linestyle='-', color='g')
        plt.xlabel("Timestep")
        plt.ylabel("Forward Reward")
        plt.title(f"Forward Rewards for {traj_key}")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    print("Finished computing rewards for single trajectory.")

    return None

if __name__ == "__main__":
    main()
