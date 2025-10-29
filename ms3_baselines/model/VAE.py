import h5py


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split, Subset

import numpy as np
import os
from datetime import datetime

# plotting
import matplotlib.pyplot as plt

from utils import make_loader, split_by_trajectory, compute_action_stats
from utils import SmallCNN

# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device,
                 hidden=512):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        # VAE
        z = self.encode(state, action)
        mu = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mu + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mu, std


    def encode(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))
        return z

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

@torch.no_grad()
def evaluate(feature_net, vae, val_loader, act_mean, act_std, device):
    feature_net.eval(); vae.eval()
    act_mean = act_mean.to(device)
    act_std = act_std.to(device)
    rec_total, kl_total, n_batch = 0.0, 0.0, 0.0
    for img_batch, action_batch in val_loader:
        img_batch = img_batch.to(device, non_blocking=True)
        action_batch = action_batch.to(device, non_blocking=True)

        img_feat = feature_net(img_batch)
        action_norm = (action_batch - act_mean) / act_std

        pred_norm, mu, std = vae(img_feat, action_norm)

        # 重构损失 + KL损失
        rec_loss = F.mse_loss(pred_norm, action_norm)
        log_std  = std.clamp_min(1e-8).log()
        kl       = -0.5 * torch.mean(torch.sum(1 + 2*log_std - mu.pow(2) - std.pow(2), dim=1))

        rec_total += rec_loss.item()
        kl_total += kl.item()
        n_batch += 1

    return rec_total / max(1, n_batch), kl_total / max(1, n_batch)

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
        _vae_sd = _sd['model']
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

    def compute_reward(self, img, action,
                       space: str = "raw",        # "raw"(默认) 或 "norm"
                       reduction: str = "mean",    # "mean"(默认) 或 "sum"
                       scale: float = 1.0,
                       offset: float = 0.0):
        """
        计算基于 VAE 的重构奖励（负的重构误差）。
        参数：
          - space: "raw" 在原动作空间比较；"norm" 在标准化后的动作空间比较
          - reduction: "mean" 逐维平均；"sum" 逐维求和
          - scale/offset: 线性标定，reward = -(scale * d + offset)
        返回：
          - numpy 数组，shape=(B,)
        """
        with torch.no_grad():
            # --- img 预处理 ---
            if not isinstance(img, torch.Tensor):
                img = torch.as_tensor(img)
            if img.ndim == 3:  # (C,H,W) or (H,W,C) -> (1, ...)
                img = img.unsqueeze(0)
            if img.ndim == 4:
                last_dim, second_dim = img.shape[-1], img.shape[1]
                if last_dim in (1,2,3,4,6) and second_dim not in (1,2,3,4,6):
                    img = img.permute(0,3,1,2)  # HWC -> CHW
            if not torch.is_floating_point(img):
                img = img.float()
            try:
                if img.max() > 1.1 or img.min() < 0.0:
                    img = img / 255.0
            except Exception:
                tmp = img.detach().cpu().float()
                if tmp.max() > 1.1 or tmp.min() < 0.0:
                    tmp = tmp / 255.0
                img = tmp
            img = img.to(self.device)
            img_feat = self.feature_net(img)

            # --- action 预处理 ---
            if not isinstance(action, torch.Tensor):
                action = torch.as_tensor(action)
            if action.ndim == 1:
                action = action.unsqueeze(0)
            action = action.to(self.device).float()

            safe_std = torch.clamp(self.act_std, min=1e-3)
            action_norm = (action - self.act_mean) / safe_std

            # 用后验均值重构
            z_in = self.vae.encode(img_feat, action_norm)
            mu   = self.vae.mean(z_in)
            recon = self.vae.decode(img_feat, mu)

            # 选择比较空间
            if space == "norm":
                diff2 = (recon - action_norm) ** 2
            elif space == "raw":
                recon_act = recon * safe_std + self.act_mean
                diff2 = (recon_act - action) ** 2
            else:
                raise ValueError("space must be 'raw' or 'norm'")

            d = diff2.mean(dim=1) if reduction == "mean" else diff2.sum(dim=1)

            # 一次性调试打印：帮助核对标尺（只在首次调用输出）
            if not hasattr(self, "_dbg_once"):
                self._dbg_once = True
                # 4种度量的中位数，方便和旧实现对齐
                d_norm_mean = ((recon - action_norm)**2).mean(dim=1).median().item()
                d_norm_sum  = ((recon - action_norm)**2).sum(dim=1).median().item()
                d_raw_mean  = ((recon * safe_std + self.act_mean - action)**2).mean(dim=1).median().item()
                d_raw_sum   = ((recon * safe_std + self.act_mean - action)**2).sum(dim=1).median().item()
                A = action.shape[1]
                print(f"[DBG] action_dim={A}")
                print("[DBG] norm_mean  median:", d_norm_mean, "  norm_sum  median:", d_norm_sum)
                print("[DBG] raw_mean   median:", d_raw_mean,  "  raw_sum   median:",  d_raw_sum)
                if d_norm_mean != 0:
                    print("[DBG] ratios ~ sum/mean: norm", d_norm_sum/max(d_norm_mean,1e-12),
                          "raw", d_raw_sum/max(d_raw_mean,1e-12))

            reward = -(scale * d + offset)
            return reward.detach().cpu().numpy()

    # def compute_reward(self, img, action):
    #     """
    #     计算 VAE 重构误差作为奖励
    #     img: (B, 2C, H, W) 或 (B, H, W, 2C) 或 (2C, H, W) 等多种形式，支持 uint8/0-255 或 已归一化 float
    #     action: (B, A) 归一化动作
    #     返回: (B,) 奖励值，重构误差的负值
    #     """
    #     with torch.no_grad():
    #         # --- 确保为 torch.Tensor 并处理可能的单样本（无 batch 维） ---
    #         if not isinstance(img, torch.Tensor):
    #             img = torch.as_tensor(img)
    #         if img.ndim == 3:
    #             # (C,H,W) 或 (H,W,C) -> 加 batch
    #             img = img.unsqueeze(0)

    #         # --- 如果是 channel-last（B,H,W,C），把它变成 (B,C,H,W) ---
    #         if img.ndim == 4:
    #             # 若最后一维看起来像通道数（常见 1,2,3,4,6 等），而第二维不是通道，则认为是 HWC 格式
    #             last_dim = img.shape[-1]
    #             second_dim = img.shape[1]
    #             if last_dim in (1, 2, 3, 4, 6) and second_dim not in (1, 2, 3, 4, 6):
    #                 img = img.permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W

    #         # --- 确保为浮点并归一化到 [0,1]（如果原始为 0-255） ---
    #         if not torch.is_floating_point(img):
    #             img = img.float()
    #         # 如果像素值明显超过 1（如 255），则认为未归一化，执行 /255.0
    #         try:
    #             if img.max() > 1.1 or img.min() < 0.0:
    #                 img = img / 255.0
    #         except Exception:
    #             # 安全兜底（若 img 在 GPU 上或其他异常），先把数据移至 cpu 计算范围再恢复
    #             tmp = img.cpu().float()
    #             if tmp.max() > 1.1 or tmp.min() < 0.0:
    #                 img = tmp / 255.0
    #             else:
    #                 img = tmp
    #             # 后面会把 img 移到 device

    #         # 移动到设备并通过特征网络
    #         img = img.to(self.device)
    #         img = self.feature_net(img)  # 提取特征

    #         # --- 处理 action 张量格式 ---
    #         if not isinstance(action, torch.Tensor):
    #             action = torch.as_tensor(action)
    #         if action.ndim == 1:
    #             action = action.unsqueeze(0)
    #         action = action.to(self.device).float()
    #         action_norm = (action - self.act_mean) / (self.act_std + 1e-12)

    #         # 编码并用后验均值重构
    #         z_in = self.vae.encode(img, action_norm)
    #         mu = self.vae.mean(z_in)
    #         recon = self.vae.decode(img, mu)

    #         ## 第一种，不反归一化即输出奖励
    #         # 计算重构奖励
    #         # d = ((recon - action_norm) ** 2).mean(dim=1)  # (B,)

    #         ## 第二种，反归一化即输出奖励
    #         # recon 是标准化空间的输出 -> 反标准化回原动作空间
    #         safe_std = torch.clamp(self.act_std, min=1e-3)
    #         recon_action = recon * safe_std + self.act_mean
    #         # 在原始动作空间上计算误差
    #         d = ((recon_action - action) ** 2).mean(dim=1)

    #         distance_reward = -d  # 奖励是重构误差的负值

    #     return distance_reward.cpu().numpy()

    def compute_penalty(self, forward_rewards, time_penalty_coef=0.01):
        time_penalty = - time_penalty_coef * torch.ones_like(forward_rewards)
        return time_penalty + forward_rewards

def testing():
    from configs.cfg_parser import parse_cfg
    cfg_name = 'VAE'
    cfg = parse_cfg(cfg_name) if cfg_name is not None else parse_cfg()

    # 加载路径
    h5_path = cfg["paths"]["h5_dataset_path"]
    save_eval_dir  = "runs/test_vae_reward"  # 临时保存目录
    load_model_path= cfg["paths"]["load_model_path"]
    latent_dim = cfg["model"]["latent_dim"]
    max_action = cfg["model"]["max_action"]
    print(f"load_model_path: {load_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集和数据加载器
    ds, dl = make_loader(h5_path, mode="single", img_size=None, num_workers=16)

    # 取出一条样本，用于初始化网络
    img, act = ds[0]         # 第 0 条样本 (img: (2C,H,W), act: (A,))
    # 创建 VAEReward 模型
    vae_reward = VAEReward(img, act, latent_dim, max_action, load_model_path, device)

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
        batch_recon_rewards = vae_reward.compute_reward(imgs, acts)
        recon_rewards_list.append(batch_recon_rewards)
    print("Recon rewards:", recon_rewards_list)

    rewards = np.concatenate(recon_rewards_list, axis=0)  # (T,)
    timesteps = np.arange(len(rewards))

    # --- 可视化与保存 ---
    out_dir = os.path.join(save_eval_dir, "eval_viz")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(timesteps, rewards, marker='o', linewidth=1)
    plt.xlabel("Timestep")
    plt.ylabel("Reward (log p_ID - log p_BG - λ·Var_MC)")
    plt.title(f"Trajectory {traj_key} Rewards")
    plt.grid(True)
    png_path = os.path.join(out_dir, f"{traj_key}_reward.png")
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close()

    # 同步保存数值
    npy_path = os.path.join(out_dir, f"{traj_key}_reward.npy")
    csv_path = os.path.join(out_dir, f"{traj_key}_reward.csv")
    np.save(npy_path, rewards)
    np.savetxt(csv_path, rewards, delimiter=",")
    print(f"[Eval] traj_key={traj_key}, T={len(rewards)}")
    print(f"[Eval] saved plot: {png_path}")
    print(f"[Eval] saved npy : {npy_path}")
    print(f"[Eval] saved csv : {csv_path}")


    return None

def main():
    from configs.cfg_parser import parse_cfg
    cfg_name = 'VAE'
    cfg = parse_cfg(cfg_name) if cfg_name is not None else parse_cfg()

    # 加载路径
    h5_path = cfg["paths"]["h5_dataset_path"]
    save_ckpt_dir = cfg["paths"]["save_ckpt_dir"]
    print(f"save_ckpt_dir: {save_ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集和数据加载器
    ds, dl = make_loader(h5_path, mode="single", img_size=None, num_workers=16)
    pair_ds, pair_dl = make_loader(h5_path, mode="pair", batch_size=265, num_workers=16)

    # 划分训练集和验证集
    train_ds, val_ds = split_by_trajectory(ds, val_ratio=0.1, seed=42)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_dl   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    # 取出一条样本，用于初始化网络
    img, act = ds[0]          # 第 0 条样本 (img: (2C,H,W), act: (A,))
    feature_net = SmallCNN(in_ch=img.shape[0], feat_dim=256).to(device)
    state_dim = feature_net.feat_dim
    action_dim = act.shape[0]
    latent_dim = 32
    max_action = 1.0
    vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
    print("successfully create VAE")
    print(f"VAE parameters: state_dim={state_dim}, action_dim={action_dim}, latent_dim={latent_dim}, max_action={max_action}, device={device}")

    # 计算动作均值和标准差
    act_mean_full, act_std_full = compute_action_stats(train_ds)   # ds 是上面 make_loader 返回的 Dataset
    act_mean_full = act_mean_full.to(device)
    act_std_full  = act_std_full.to(device)
    act_mean_full.requires_grad_(False)
    act_std_full.requires_grad_(False)
    print(f"[act stats] mean[:4]={act_mean_full[:4].tolist()}, std[:4]={act_std_full[:4].tolist()}")

    # 设置优化器
    params = list(feature_net.parameters()) + list(vae.parameters())
    optim = torch.optim.Adam(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

    # --- Logger ---
    import wandb

    wandb.init(project=cfg['log']['wandb_project'],
               group=cfg['log']['wandb_group'],
               config=cfg)


    # 循环训练
    epochs = 100
    running = 0.0
    best_val = float("inf")

    # history for plotting
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        print(f"Starting Training Epoch {epoch+1}/{epochs}:")
        feature_net.train()
        vae.train()
        running = 0.0

        for img_batch, action_batch in train_dl:
            img_batch = img_batch.to(device)       # (B, 2C, H, W)
            action_batch = action_batch.to(device) # (B, A)
            # imgs: (B, C, H, W)  acts: (B, A)
            # print("img_batch.shape, action_batch.shape", img_batch.shape, action_batch.shape)
            img_feat = feature_net(img_batch)  # (B, state_dim)
            # print("img_feat.shape", img_feat.shape)

            # 标准化动作
            action_batch_norm = (action_batch - act_mean_full) / act_std_full
            pred_norm, mu, std = vae(img_feat, action_batch_norm)

            rec_loss = F.mse_loss(pred_norm, action_batch_norm)
            # KL = 0.5 * sum( std^2 + mu^2 - 1 - log(std^2) )
            log_std = std.clamp_min(1e-8).log()
            kl = -0.5 * torch.mean(torch.sum(1 + 2*log_std - mu.pow(2) - std.pow(2), dim=1))

            loss = rec_loss + kl

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += loss.item()

        train_loss = running / max(1, len(train_dl))
        val_rec, val_kl = evaluate(feature_net, vae, val_dl, act_mean_full, act_std_full, device)
        monitor_val_loss =  val_rec + val_kl
        val_loss = monitor_val_loss
        print(f"[epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"monitor_val_loss={monitor_val_loss:.4f} | "
            f"val_rec={val_rec:.4f} val_kl={val_kl:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/rec_loss": val_rec,
            "val/kl_loss": val_kl,
        })

        # record histories
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # 调整学习率
        scheduler.step(monitor_val_loss)

        # 保存最好模型
        min_delta = 1e-3
        if epoch == 0:
            os.makedirs(save_ckpt_dir, exist_ok=True)

        if monitor_val_loss < best_val - min_delta:
            best_val = monitor_val_loss
            torch.save({
                "feature_net": feature_net.state_dict(),
                "model": vae.state_dict(),
                "act_mean": act_mean_full.detach().cpu(),
                "act_std": act_std_full.detach().cpu(),
            }, os.path.join(save_ckpt_dir, "vae_best.pt"))
            print(f"[SAVE] best checkpoint @ val={best_val:.4f}")

    # 绘图并保存（训练完成后）
    try:
        epochs_range = list(range(1, len(train_loss_history) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, train_loss_history, label="train_loss", marker="o")
        plt.plot(epochs_range, val_loss_history, label="val_loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train and Val Loss")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(save_ckpt_dir, "loss_curve.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved loss curve to: {plot_path}")
    except Exception as e:
        print(f"Failed to save loss curve: {e}")

if __name__ == "__main__":
    main()
    # testing()