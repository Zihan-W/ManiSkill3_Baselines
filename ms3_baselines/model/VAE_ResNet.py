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

from utils import make_loader, split_by_trajectory, compute_action_stats, compute_qpos_stats
from vision_encoder import LateFusionResNet

from VAE import VAE, evaluate, testing

class VAEReward:
    def __init__(self, img, act, qpos, latent_dim, max_action, model_path, device):
        self.device = device

        self.feature_net = LateFusionResNet(in_channels=img.shape[0], feat_dim=256, pretrained=True).to(device)
        self.state_dim = self.feature_net.feat_dim
        self.action_dim = act.shape[0]
        self.qpos_dim = qpos.shape[0]
        self.latent_dim = latent_dim
        self.max_action = max_action

        self.vae = VAE(state_dim=self.state_dim,
                       qpos_dim=self.qpos_dim,
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
        _qpos_mean = _sd.get('qpos_mean', None)
        _qpos_std  = _sd.get('qpos_std', None)

        self.feature_net.load_state_dict(_feature_net_sd)
        self.vae.load_state_dict(_vae_sd)

        self.feature_net.eval()
        self.vae.eval()

        # action stats (fallback to zeros/ones if missing)
        if _act_mean is None or _act_std is None:
            # safe fallback
            self.act_mean = torch.zeros(self.action_dim, device=device)
            self.act_std = torch.ones(self.action_dim, device=device)
        else:
            self.act_mean = _act_mean.to(device)
            self.act_std  = _act_std.to(device)
        self.act_mean.requires_grad_(False)
        self.act_std.requires_grad_(False)

        # qpos stats (if checkpoint contains them use, else fallback to zeros/ones)
        if _qpos_mean is None or _qpos_std is None:
            self.qpos_mean = torch.zeros(self.qpos_dim, device=device)
            self.qpos_std = torch.ones(self.qpos_dim, device=device)
        else:
            self.qpos_mean = _qpos_mean.to(device)
            self.qpos_std = _qpos_std.to(device)
        self.qpos_mean.requires_grad_(False)
        self.qpos_std.requires_grad_(False)

        self.feature_net.eval()
        self.vae.eval()

    def compute_reward(self, img, action, qpos):
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
            img = img.to(self.device)
            img = self.feature_net(img)  # 提取特征

            # --- 处理 action 张量格式 ---
            if not isinstance(action, torch.Tensor):
                action = torch.as_tensor(action)
            if action.ndim == 1:
                action = action.unsqueeze(0)
            action = action.to(self.device).float()
            action_norm = (action - self.act_mean) / (self.act_std + 1e-12)

            # --- 处理 qpos ---
            if not isinstance(qpos, torch.Tensor):
                qpos = torch.as_tensor(qpos)
            if qpos.ndim == 1:
                qpos = qpos.unsqueeze(0)
            qpos = qpos.to(self.device).float()
            qpos_norm = (qpos - self.qpos_mean) / (self.qpos_std + 1e-12)

            # 编码并用后验均值重构 (VAE.encode expects state, qpos_norm)
            z_in = self.vae.encode(img, qpos_norm)
            mu = self.vae.mean(z_in)
            recon_action = self.vae.decode(img, mu)  # recon is action in original scale
            recon_norm = (recon_action - self.act_mean) / (self.act_std + 1e-12)

            # 计算重构奖励，标准化空间输出奖励
            d = ((recon_norm - action_norm) ** 2).mean(dim=1)  # (B,)

            distance_reward = -d  # 奖励是重构误差的负值

        return distance_reward.cpu().numpy()

    def compute_penalty(self, forward_rewards, time_penalty_coef=0.01):
        time_penalty = - time_penalty_coef * torch.ones_like(forward_rewards)
        return time_penalty + forward_rewards

@torch.no_grad()
def evaluate(feature_net, vae, val_loader, act_mean, act_std, qpos_mean, qpos_std, device):
    feature_net.eval()
    vae.eval()
    act_mean = act_mean.to(device)
    act_std = act_std.to(device)
    qpos_mean = qpos_mean.to(device)
    qpos_std = qpos_std.to(device)
    rec_total, kl_total, n_batch = 0.0, 0.0, 0.0
    for img_batch, action_batch, qpos_batch in val_loader:
        img_batch = img_batch.to(device, non_blocking=True)
        action_batch = action_batch.to(device, non_blocking=True)
        qpos_batch = qpos_batch.to(device, non_blocking=True)

        img_feat = feature_net(img_batch)
        action_norm = (action_batch - act_mean) / (act_std + 1e-12)
        qpos_norm = (qpos_batch - qpos_mean) / (qpos_std + 1e-12)

        pred_action, mu, std = vae(img_feat, qpos_norm)
        pred_norm = (pred_action - act_mean) / (act_std + 1e-12)

        # 重构损失 + KL损失
        rec_loss = F.mse_loss(pred_norm, action_norm)
        log_std  = std.clamp_min(1e-8).log()
        kl       = -0.5 * torch.mean(torch.sum(1 + 2*log_std - mu.pow(2) - std.pow(2), dim=1))

        rec_total += rec_loss.item()
        kl_total += kl.item()
        n_batch += 1

    return rec_total / max(1, n_batch), kl_total / max(1, n_batch)

def testing():
    from configs.cfg_parser import parse_cfg
    cfg_name = 'VAE_ResNet'
    cfg = parse_cfg(cfg_name) if cfg_name is not None else parse_cfg()

    # 加载路径
    h5_path = cfg["test"]["h5_dataset_path"]
    save_eval_dir  = "runs/" + cfg['model']['name'] + "_" + cfg['task']  # 临时保存目录
    load_model_path= cfg["paths"]["load_model_path"]
    latent_dim = cfg["model"]["latent_dim"]
    max_action = cfg["model"]["max_action"]
    print(f"load_model_path: {load_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集和数据加载器
    ds, dl = make_loader(h5_path, mode="single", img_size=None, num_workers=16, camera_mask=['base'])

    # 取出一条样本，用于初始化网络
    img, act, qpos = ds[0]         # 第 0 条样本 (img: (2C,H,W), act: (A,), qpos: (Q,))
    # 创建 VAEReward 模型
    vae_reward = VAEReward(img, act, qpos, latent_dim, max_action, load_model_path, device)

    # 在单条轨迹上计算奖励
    traj_id = cfg["test"]["traj_id"]
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
        qpos = torch.stack([ds[i][2] for i in batch_idxs], dim=0)   # (B, Q)
        # 计算 reward（VAEReward.compute_reward 返回 numpy array (B,)）
        batch_recon_rewards = vae_reward.compute_reward(imgs, acts, qpos)
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
    cfg_name = 'VAE_ResNet'
    cfg = parse_cfg(cfg_name) if cfg_name is not None else parse_cfg()

    # 加载路径
    h5_path = cfg["train"]["h5_dataset_path"]
    save_ckpt_dir = cfg["paths"]["save_ckpt_dir"]
    print(f"save_ckpt_dir: {save_ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集和数据加载器ResNet
    ds, dl = make_loader(h5_path, mode="single", img_size=None, num_workers=16, camera_mask=None)
    pair_ds, pair_dl = make_loader(h5_path, mode="pair", batch_size=265, num_workers=16, camera_mask=None)

    # 划分训练集和验证集
    train_ds, val_ds = split_by_trajectory(ds, val_ratio=0.1, seed=42)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_dl   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    # 取出一条样本，用于初始化网络
    img, act, qpos = ds[0]          # 第 0 条样本 (img: (2C,H,W), act: (A,), qpos: (Q,))
    feature_net = LateFusionResNet(in_channels=img.shape[0], feat_dim=256, pretrained=True).to(device)

    # 冻结所有 encoder 参数
    for param in feature_net.encoder_left.parameters():
        param.requires_grad = False
    for param in feature_net.encoder_right.parameters():
        param.requires_grad = False

    # 只训练 fusion 层 + fc_left + fc_right
    for param in feature_net.fc_left.parameters():
        param.requires_grad = True
    for param in feature_net.fc_right.parameters():
        param.requires_grad = True
    for param in feature_net.fusion.parameters():
        param.requires_grad = True

    state_dim = feature_net.feat_dim
    action_dim = act.shape[0]
    qpos_dim = qpos.shape[0]
    latent_dim = 8
    max_action = 1.0
    vae = VAE(state_dim=state_dim, qpos_dim=qpos_dim, action_dim=action_dim, latent_dim=latent_dim, max_action=max_action, device=device).to(device)
    print("successfully create VAE")
    print(f"VAE parameters: state_dim={state_dim}, action_dim={action_dim}, latent_dim={latent_dim}, max_action={max_action}, device={device}")

    # 计算动作均值和标准差
    act_mean_full, act_std_full = compute_action_stats(train_ds)   # ds 是上面 make_loader 返回的 Dataset
    act_mean_full = act_mean_full.to(device)
    act_std_full  = act_std_full.to(device)
    act_mean_full.requires_grad_(False)
    act_std_full.requires_grad_(False)
    # 计算 qpos stats
    qpos_mean_full, qpos_std_full = compute_qpos_stats(train_ds)
    qpos_mean_full = qpos_mean_full.to(device)
    qpos_std_full = qpos_std_full.to(device)
    qpos_mean_full.requires_grad_(False)
    qpos_std_full.requires_grad_(False)
    print(f"[act stats] mean[:4]={act_mean_full[:4].tolist()}, std[:4]={act_std_full[:4].tolist()}")

    # 设置优化器
    params = list(feature_net.parameters()) + list(vae.parameters())
    optim = torch.optim.Adam(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

    # --- Logger ---
    import wandb
    wandb_project_name = cfg['log']['wandb_project']
    wandb_group_name = cfg['log']['wandb_group']
    wandb.init(project=wandb_project_name,
               group=wandb_group_name,
               config=cfg)

    # 循环训练
    epochs = 100
    running = 0.0
    best_val = float("inf")

    # history for plotting
    train_loss_history = []
    val_loss_history = []

    def kl_beta_schedule(epoch, max_beta=0.2, warmup_epochs=30):
        return min(max_beta, (epoch + 1) / warmup_epochs * max_beta)

    for epoch in range(epochs):
        print(f"Starting Training Epoch {epoch+1}/{epochs}:")
        feature_net.train()
        vae.train()
        running = 0.0
        rec_loss_epoch = 0.0
        kl_loss_epoch = 0.0

        beta = kl_beta_schedule(epoch)

        for img_batch, action_batch, qpos_batch in train_dl:
            img_batch = img_batch.to(device)       # (B, 2C, H, W)
            action_batch = action_batch.to(device) # (B, A)
            qpos_batch = qpos_batch.to(device)     # (B, Q)
            # imgs: (B, C, H, W)  acts: (B, A)
            # print("img_batch.shape, action_batch.shape", img_batch.shape, action_batch.shape)
            img_feat = feature_net(img_batch)  # (B, state_dim)
            # print("img_feat.shape", img_feat.shape)

            # 标准化动作 and qpos
            action_batch_norm = (action_batch - act_mean_full) / (act_std_full + 1e-12)
            qpos_batch_norm = (qpos_batch - qpos_mean_full) / (qpos_std_full + 1e-12)

            pred_action, mu, std = vae(img_feat, qpos_batch_norm)
            pred_norm = (pred_action - act_mean_full) / (act_std_full + 1e-12)

            rec_loss = F.mse_loss(pred_norm, action_batch_norm)
            # KL = 0.5 * sum( std^2 + mu^2 - 1 - log(std^2) )
            log_std = std.clamp_min(1e-8).log()
            kl = -0.5 * torch.mean(torch.sum(1 + 2*log_std - mu.pow(2) - std.pow(2), dim=1))

            loss = rec_loss + beta * kl

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += loss.item()
            rec_loss_epoch += rec_loss.item()
            kl_loss_epoch += kl.item()

        train_loss = running / max(1, len(train_dl))
        rec_loss_epoch /= max(1, len(train_dl))
        kl_loss_epoch /= max(1, len(train_dl))
        val_rec, val_kl = evaluate(feature_net, vae, val_dl, act_mean_full, act_std_full, qpos_mean_full, qpos_std_full, device)
        monitor_val_loss =  val_rec + beta * val_kl
        val_loss = monitor_val_loss
        print(f"[epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"monitor_val_loss={monitor_val_loss:.4f} | "
            f"val_rec={val_rec:.4f} val_kl={val_kl:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/rec_loss": rec_loss_epoch,
            "train/kl_loss": kl_loss_epoch,
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
                "qpos_mean": qpos_mean_full.detach().cpu(),
                "qpos_std": qpos_std_full.detach().cpu(),
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
    # main()
    testing()