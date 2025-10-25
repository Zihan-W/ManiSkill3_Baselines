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

class VAEReward:
    def __init__(self, img, act, latent_dim, max_action, model_path, device):
        self.device = device

        self.feature_net = SmallCNN(in_ch=img.shape[0], feat_dim=256).to(device)
        self.state_dim = self.feature_net.feat_dim
        self.action_dim = act.shape[0]
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

    # ---------- 辅助函数：把一批观测和动作转成特征 ----------
    def _preprocess_img(self, img):
        """
        输入 img 可以是:
            (B, C, H, W)
            (B, H, W, C)
            (C, H, W)
        值范围可以是 uint8/0-255 或 float[0,1]。
        输出:
            state_feat: (B, state_dim) from feature_net
        """
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)
        if img.ndim == 3:
            img = img.unsqueeze(0)  # 单样本 -> batch

        # 如果是 (B,H,W,C) 转成 (B,C,H,W)
        if img.ndim == 4:
            last_dim = img.shape[-1]
            second_dim = img.shape[1]
            if last_dim in (1, 2, 3, 4, 6) and second_dim not in (1, 2, 3, 4, 6):
                img = img.permute(0, 3, 1, 2)

        if not torch.is_floating_point(img):
            img = img.float()

        # 如果像素是0~255就归一化到0~1
        try:
            if img.max() > 1.1 or img.min() < 0.0:
                img = img / 255.0
        except Exception:
            tmp = img.cpu().float()
            if tmp.max() > 1.1 or tmp.min() < 0.0:
                img = tmp / 255.0
            else:
                img = tmp

        img = img.to(self.device)
        state_feat = self.feature_net(img)  # (B, state_dim)
        return state_feat

    def _preprocess_action(self, action):
        """
        输入 action: (B, A) 或 (A,)
        归一化: (action - mean) / std
        """
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = action.to(self.device).float()
        action_norm = (action - self.act_mean) / (self.act_std + 1e-12)
        return action_norm

    @torch.no_grad()
    def compute_reward(self, img, action):
        """
        计算 VAE 重构误差作为奖励
        img: (B, 2C, H, W) 或 (B, H, W, 2C) 或 (2C, H, W) 等多种形式，支持 uint8/0-255 或 已归一化 float
        action: (B, A) 归一化动作
        返回: (B,) 奖励值，重构误差的负值
        """
        # 1. 预处理
        img = self._preprocess_img(img)          # (B, state_dim)
        action_norm = self._preprocess_action(action)   # (B, A)

        # 编码并用后验均值重构
        z_in = self.vae.encode(img, action_norm)
        mu = self.vae.mean(z_in)
        recon = self.vae.decode(img, mu)

        # 计算重构奖励
        d = ((recon - action_norm) ** 2).mean(dim=1)  # (B,)
        distance_reward = -d  # 奖励是重构误差的负值

        # 计算方向奖励（每个样本）
        a_direction = self.vae.dir_head(img)  # (B, A) 动作方向
        a_direction = F.normalize(a_direction, dim=1, eps=1e-8)
        a_unit = F.normalize(action_norm, dim=1, eps=1e-8)
        cos_sim = torch.sum(a_direction * a_unit, dim=1)  # (B,)
        direction_reward = cos_sim  # (B,) 最大化余弦相似度（使方向一致）

        # 计算进展奖励（每个样本）
        forward_reward = self.vae.progress_head(img).squeeze(1)  # (B,)

        return distance_reward.cpu().numpy(), direction_reward.cpu().numpy(), forward_reward.cpu().numpy()
        # return (recon_reward + direction_reward).cpu().numpy()

    @torch.no_grad()
    def compute_progress(self, img_t, img_tp1):
        """
        输入:
            img_t, img_tp1: 邻近两个状态的观测 (可以是batch)
        输出:
            phi_t           (B,)
            phi_tp1         (B,)
            progress_delta  (B,) = phi_tp1 - phi_t
        用于 PPO 里面 progress 奖励。
        """
        state_t   = self._preprocess_img(img_t)    # (B, state_dim)
        state_tp1 = self._preprocess_img(img_tp1)  # (B, state_dim)

        phi_t   = self.vae.progress_head(state_t).squeeze(1)    # (B,)
        phi_tp1 = self.vae.progress_head(state_tp1).squeeze(1)  # (B,)

        progress_delta = phi_tp1 - phi_t                        # (B,)
        return phi_t, phi_tp1, progress_delta

    def compute_penalty(self, forward_rewards, time_penalty_coef=0.01):
        time_penalty = - time_penalty_coef * torch.ones_like(forward_rewards)
        return time_penalty

def load_VAE_reward_model(img, act, model_path, device):
    # 配置参数
    # StackCube-v1 motionplanning 数据集
    # dataset_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/StackCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"   # 数据集路径
    # StackCube-v1 teleop 数据集
    # dataset_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/StackCube-v1/demos/StackCube-v1/teleop/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"   # 数据集路径
    # StackCube-v1 teleop model权重
    # model_path   = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/ManiSkill3_Baselines/ms3_baselines/VAE/ckpt_VAE/stack_cube_version1/vae_best.pt" if model_path is None else model_path

    # PushCube-v1 motionplanning 数据集
    dataset_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/PushCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"   # 数据集路径
    # model_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/ManiSkill3_Baselines/ms3_baselines/VAE/ckpt_VAE/push_cube_version2/vae_best.pt"
    # 带进展相位奖励
    model_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/ManiSkill3_Baselines/ms3_baselines/VAE/ckpt_VAE/push_cube_version4/vae_best.pt"

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
    traj_id = 90
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
    forward_phi_list = []

    for start in range(0, len(traj_indices), batch_size):
        batch_idxs = traj_indices[start:start + batch_size]
        # 从 dataset 读取并堆叠为 batch
        imgs = torch.stack([ds[i][0] for i in batch_idxs], dim=0)   # (B, 2C, H, W)
        acts = torch.stack([ds[i][1] for i in batch_idxs], dim=0)   # (B, A)
        # 计算 reward（VAEReward.compute_reward 返回 numpy array (B,)）
        batch_recon_rewards, batch_direction_rewards, batch_phi = vae.compute_reward(imgs, acts)
        recon_rewards_list.append(batch_recon_rewards)
        direction_rewards_list.append(batch_direction_rewards)
        forward_phi_list.append(batch_phi)

    # progress_delta 用 phi_{t+1}-phi_t
    all_progress_delta = []
    for i in range(len(traj_indices)-1):
        idx_t   = traj_indices[i]
        idx_tp1 = traj_indices[i+1]
        img_t, act_t = ds[idx_t]
        img_tp1, act_tp1 = ds[idx_tp1]

        _, _, prog_delta = vae.compute_progress(img_t, img_tp1)
        all_progress_delta.append(prog_delta.squeeze().cpu().item())

    print("Recon rewards:", recon_rewards_list)
    print("Direction rewards:", direction_rewards_list)
    print("Forward phi:", forward_phi_list)
    print("Progress deltas:", all_progress_delta)

    timesteps = np.arange(len(traj_indices))
    timesteps_prog = np.arange(len(traj_indices)-1)

    # 合并并保持时间顺序
    if len(recon_rewards_list) == 0:
        print(f"No rewards computed for {traj_key}.")
    else:
        recon_rewards = np.concatenate(recon_rewards_list, axis=0)  # (T,)
        direction_rewards = np.concatenate(direction_rewards_list, axis=0)  # (T,)
        forward_rewards = np.concatenate(forward_phi_list, axis=0)  # (T,)
        timesteps = np.arange(len(recon_rewards))

        # 绘图
        plt.figure(figsize=(12, 5))
        plt.subplot(4, 1, 1)
        plt.plot(timesteps, recon_rewards, marker='o', linestyle='-', color='b')
        plt.xlabel("Timestep")
        plt.ylabel("Distance Reward")
        plt.title(f"Distance Rewards for {traj_key}")
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(timesteps, direction_rewards, marker='o', linestyle='-', color='g')
        plt.xlabel("Timestep")
        plt.ylabel("Direction Reward")
        plt.title(f"Direction Rewards for {traj_key}")
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(timesteps, forward_rewards, marker='o', linestyle='-', color='r')
        plt.title(f"{traj_key} phi(s) (learned progress score)")
        plt.grid(True)

        plt.subplot(4, 1, 4)
        plt.plot(timesteps_prog, all_progress_delta, marker='o')
        plt.title(f"{traj_key} progress_delta = phi(t+1)-phi(t)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    print("Finished computing rewards for single trajectory.")

    return None

if __name__ == "__main__":
    main()
