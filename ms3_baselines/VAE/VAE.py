import h5py


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split, Subset

import numpy as np
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional

class H5TrajectoryDataset(Dataset):
    """
    统一版 HDF5 轨迹数据集：
    - mode='single': 返回 (img_t, act_t)
    - mode='pair'  : 返回 (img_t, act_t, img_{t+1}, act_{t+1})，只在同一条轨迹内取相邻时间步

    目录假设（与你当前h5一致）：
      <traj_key>/obs/sensor_data/base_camera/rgb -> (T, H, W, C) uint8
      <traj_key>/obs/sensor_data/hand_camera/rgb -> (T, H, W, C) uint8
      <traj_key>/actions                         -> (T, A)       float

    其它：
    - 图像会拼到通道维 (2C, H, W) 并归一化到 [0,1]
    - 可选 resize 到 img_size（若不设则不resize）
    - 支持 lazy open（多进程DataLoader友好）
    - 暴露 indices / traj_to_idxs 便于你按轨迹抽样或画曲线
    """

    def __init__(self,
                 h5_path: str,
                 mode: str = "single",      # 'single' or 'pair'
                 img_size: Optional[int] = None,
                 traj_whitelist: Optional[List[str]] = None):
        super().__init__()
        assert mode in ("single", "pair")
        self.h5_path = h5_path
        self.mode = mode
        self.img_size = img_size
        self._h5 = None  # lazy open

        self.indices: List[Tuple[str,int]] = []        # (traj_key, t) 仅起点
        self.traj_to_idxs: Dict[str, List[int]] = {}   # 每条轨迹对应的 dataset 索引（for 'single' 模式）
        self.traj_lengths: Dict[str, int] = {}         # 每条轨迹的 T

        # 预扫描文件，建立索引
        with h5py.File(self.h5_path, "r") as f:
            all_keys = list(f.keys())
            if traj_whitelist is not None:
                all_keys = [k for k in all_keys if k in traj_whitelist]
            for k in all_keys:
                grp = f[k]
                base = grp["obs/sensor_data/base_camera/rgb"]  # (T,H,W,C)
                if "obs/sensor_data/hand_camera/rgb" in grp:
                    hand = grp["obs/sensor_data/hand_camera/rgb"]  # (T,H,W,C)
                    has_hand = True
                else:
                    hand = None
                    has_hand = False
                acts = grp["actions"]                           # (T,A)
                if has_hand:
                    T = min(len(base), len(hand), len(acts))
                else:
                    T = min(len(base), len(acts))
                self.traj_lengths[k] = T

                if self.mode == "single":
                    # 所有时间步都可用
                    start = len(self.indices)
                    for t in range(T):
                        self.indices.append((k, t))
                    # 记录这条轨迹在整个 ds.indices 中的所有位置（便于之后按轨迹抽样）
                    end = len(self.indices)
                    self.traj_to_idxs[k] = list(range(start, end))
                else:
                    # pair 模式：只能到 T-2 的起点，样本是 (t, t+1)
                    for t in range(T - 1):
                        self.indices.append((k, t))

    def __len__(self):
        return len(self.indices)

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def _load_one(self, traj_key: str, t: int):
        """读取单步 (img_t, act_t)，兼容没有 hand_camera 的情况"""
        traj = self._h5[traj_key]
        base = traj["obs/sensor_data/base_camera/rgb"][t]   # (H,W,C) uint8
        if "obs/sensor_data/hand_camera/rgb" in traj:
            hand = traj["obs/sensor_data/hand_camera/rgb"][t]   # (H,W,C) uint8
            has_hand = True
        else:
            hand = None
            has_hand = False
        a    = traj["actions"][t]                           # (A,)
        proprioceptive = traj["obs/agent/qpos"][t][:-2]  # (P,)

        base = torch.from_numpy(base).permute(2,0,1).float() / 255.0  # (C,H,W)
        if has_hand:
            hand = torch.from_numpy(hand).permute(2,0,1).float() / 255.0  # (C,H,W)
            img  = torch.cat([base, hand], dim=0)                         # (2C,H,W)
        else:
            img = base  # (C,H,W)

        if self.img_size is not None:
            # resize 到 (img_size,img_size)
            img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False).squeeze(0)
        act = torch.from_numpy(a).float()                              # (A,)

        proprioceptive = torch.from_numpy(proprioceptive).float()      # (P,)

        return img, act, proprioceptive

    def __getitem__(self, idx: int):
        self._ensure_open()
        traj_key, t = self.indices[idx]
        if self.mode == "single":
            img, act, prop = self._load_one(traj_key, t)
            return img, act, prop
        else:
            # pair: (t, t+1) 保证是同一条轨迹
            img_t,  act_t, prop_t = self._load_one(traj_key, t)
            img_tp1,act_tp1, prop_tp1 = self._load_one(traj_key, t+1)
            return img_t, act_t, img_tp1, act_tp1, prop_t, prop_tp1

    # ======= 一些实用辅助 =======

    def get_traj_keys(self) -> List[str]:
        return list(self.traj_lengths.keys())

    def get_traj_indices(self, traj_key: str) -> List[int]:
        """
        仅在 mode='single' 下有意义：返回这条轨迹在数据集中所有 index（按时间顺序）。
        用于你“画一整条轨迹的 reward 曲线”的场景。
        """
        return self.traj_to_idxs.get(traj_key, [])

def make_loader(h5_path: str,
                batch_size: int = 128,
                shuffle: bool = True,
                num_workers: int = 0,
                mode: str = "single",
                img_size: Optional[int] = None,
                traj_whitelist: Optional[List[str]] = None):
    persistent = (num_workers > 0)
    prefetch = 4 if num_workers > 0 else None
    ds = H5TrajectoryDataset(h5_path=h5_path, mode=mode, img_size=img_size, traj_whitelist=traj_whitelist)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=prefetch
    )
    return ds, dl

# 实现简单的CNN编码器
class SmallCNN(nn.Module):
    def __init__(self, in_ch=6, img_feat_dim=256):
        super().__init__()
        self.img_feat_dim = img_feat_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),   nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, img_feat_dim)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x)

class MLP(nn.Module):
    def __init__(self, prop_dim, prop_feat_dim, hidden_dims: List[int], activate_final: bool = False):
        super(MLP, self).__init__()
        layers = []
        last_dim = prop_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(last_dim, hdim))
            layers.append(nn.ReLU(inplace=True))
            last_dim = hdim
        layers.append(nn.Linear(last_dim, prop_feat_dim))
        if activate_final:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class FeatureExtractor(nn.Module):
    def __init__(self, img, prop, hidden_dims: List[int] = [512, 512]):
        super(FeatureExtractor, self).__init__()
        in_ch=img.shape[0]
        img_feat_dim =256
        self.cnn = SmallCNN(in_ch=in_ch, img_feat_dim=img_feat_dim)

        self.prop_dim = prop.shape[0]
        prop_feat_dim = 16
        self.mlp = MLP(self.prop_dim, prop_feat_dim, hidden_dims=hidden_dims)

        self.feat_dim = img_feat_dim + prop_feat_dim

    def forward(self, img_batch, prop_batch):
        img_feat = self.cnn(img_batch)
        prop_feat = self.mlp(prop_batch)
        feat = torch.cat([img_feat, prop_feat], dim=1)
        return feat

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

        # ====== 新增 head 1：方向场 ======
        self.dir_head = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, action_dim)  # 最后不激活，后面做 normalize
        )

        # ====== 新增 head 2：置信度（判别器）======
        self.conf_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden//2, 1)  # 输出 logit
        )

    def forward(self, state, action):
        # VAE
        z = self.encode(state, action)
        mu = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mu + std * torch.randn_like(std)

        u = self.decode(state, z)

        # 方向头
        a_dir = self.dir_head(state)                       # (B, A)
        a_dir = F.normalize(a_dir, dim=1, eps=1e-8)        # 单位化
        # 置信头，判别 (s, a_norm) 是否在流形上
        conf_logit = self.conf_head(torch.cat([state, action], dim=1)).squeeze(1)  # (B,)

        return u, mu, std, a_dir, conf_logit

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

def compute_action_stats(dataset, batch_size=2048, num_workers=4, pin_memory=True,
                         use_cuda=None, stats_path: str | None = None):
    """
    始终重新计算动作的 mean/std（不做任何文件读写）。
    参数说明类似之前的实现，但 stats_path 参数会被忽略以保持向后兼容。
    返回：mean, std（均为 CPU 上的 torch.Tensor）
    """
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    tmp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=(num_workers > 0))

    n = 0
    s1 = None
    s2 = None
    with torch.no_grad():
        for batch in tmp_loader:
            # 支持 dataset 返回 (img, act) 或 (img, act, prop) 等多种形式：
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                a = batch[1]
            else:
                a = batch

            if use_cuda:
                a = a.to(device, non_blocking=True)
            a = a.float()

            s1 = a.sum(dim=0) if s1 is None else s1 + a.sum(dim=0)
            s2 = (a * a).sum(dim=0) if s2 is None else s2 + (a * a).sum(dim=0)
            n += a.shape[0]

    if s1 is None:
        raise RuntimeError("No data found in dataset to compute action stats")

    mean_dev = s1 / n
    var_dev = s2 / n - mean_dev ** 2
    mean = mean_dev.cpu()
    std = torch.sqrt(var_dev.clamp_min(1e-12)).cpu()

    return mean, std

def split_by_trajectory(ds: H5TrajectoryDataset, val_ratio: float = 0.1, seed: int = 0):
    """
    按“整条轨迹”为单位做 train/val 划分，避免同一条轨迹的数据泄漏到验证集。
    仅需 `ds.mode='single'`（pair 模式也能用，但通常我们验证用 single）。
    返回：train_subset, val_subset
    """
    rng = np.random.RandomState(seed)
    traj_keys = ds.get_traj_keys()
    rng.shuffle(traj_keys)

    n_val = max(1, int(len(traj_keys) * val_ratio))
    val_trajs = set(traj_keys[:n_val])
    train_trajs = set(traj_keys[n_val:])

    # 找到属于这些轨迹的样本索引
    if ds.mode == "single":
        train_indices, val_indices = [], []
        for k in train_trajs:
            train_indices.extend(ds.get_traj_indices(k))
        for k in val_trajs:
            val_indices.extend(ds.get_traj_indices(k))
    else:
        # pair 模式下，indices 是 (traj,t) 起点；按 traj_key 过滤即可
        train_indices = [i for i,(k,_) in enumerate(ds.indices) if k in train_trajs]
        val_indices   = [i for i,(k,_) in enumerate(ds.indices) if k in val_trajs]

    train_subset = Subset(ds, sorted(train_indices))
    val_subset   = Subset(ds, sorted(val_indices))
    return train_subset, val_subset

@torch.no_grad()
def evaluate(feature_net, vae, val_loader, act_mean, act_std, device):
    feature_net.eval(); vae.eval()
    rec_total, kl_total, dir_total, conf_total, n_batch = 0.0, 0.0, 0.0, 0.0, 0
    for img_batch, action_batch, prop_batch in val_loader:
        img_batch = img_batch.to(device)
        action_batch = action_batch.to(device)
        prop_batch = prop_batch.to(device)

        state_feat = feature_net(img_batch, prop_batch)
        action_norm = (action_batch - act_mean) / act_std

        pred_norm, mu, std, a_direction, conf_logit = vae(state_feat, action_norm)

        # 重构损失 + KL损失
        rec_loss = F.mse_loss(pred_norm, action_norm)
        log_std  = std.clamp_min(1e-8).log()
        kl       = -0.5 * torch.mean(torch.sum(1 + 2*log_std - mu.pow(2) - std.pow(2), dim=1))

        # 方向损失（cosine）
        a_unit = F.normalize(action_norm, dim=1, eps=1e-8)
        cos_sim = torch.sum(a_direction * a_unit, dim=1)
        dir_loss = (1.0 - cos_sim).mean()

        # 置信度损失
        y_pos = torch.ones_like(conf_logit)
        bce_pos = F.binary_cross_entropy_with_logits(conf_logit, y_pos)

        idx_neg = torch.randperm(action_norm.size(0), device=device)
        a_neg = action_norm[idx_neg]
        conf_neg = vae.conf_head(torch.cat([state_feat, a_neg], dim=1)).squeeze(1)
        y_neg = torch.zeros_like(conf_neg)
        bce_neg = F.binary_cross_entropy_with_logits(conf_neg, y_neg)
        conf_loss = 0.5 * (bce_pos + bce_neg)

        rec_total += rec_loss.item()
        kl_total += kl.item()
        dir_total += dir_loss.item()
        conf_total += conf_loss.item()
        n_batch += 1

    return rec_total / max(1, n_batch), kl_total / max(1, n_batch), dir_total / max(1, n_batch), conf_total / max(1, n_batch)

if __name__ == "__main__":
    # StackCube-v1
    # h5_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/StackCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"

    # PushCube-v1
    h5_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/PushCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"

    base_ckpt_dir = os.path.join(os.path.dirname(__file__), "ckpt_VAE")
    stats_path = base_ckpt_dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_path = os.path.join(base_ckpt_dir, timestamp)
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"ckpt path: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集和数据加载器
    ds, dl = make_loader(h5_path, mode="single", img_size=None, num_workers=8)
    pair_ds, pair_dl = make_loader(h5_path, mode="pair", batch_size=64, num_workers=8)

    # 划分训练集和验证集
    train_ds, val_ds = split_by_trajectory(ds, val_ratio=0.1, seed=42)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_dl   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    # 取出一条样本，用于初始化网络
    img, act, prop = ds[0]          # 第 0 条样本 (img: (2C,H,W), act: (A,))
    feature_net = FeatureExtractor(img, prop).to(device)
    state_dim = feature_net.feat_dim
    action_dim = act.shape[0]
    latent_dim = 32
    max_action = 1.0
    vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
    print("successfully create VAE")
    print(f"VAE parameters: state_dim={state_dim}, action_dim={action_dim}, latent_dim={latent_dim}, max_action={max_action}, device={device}")

    # 计算动作均值和标准差
    act_mean_full, act_std_full = compute_action_stats(train_ds, stats_path=stats_path)   # ds 是上面 make_loader 返回的 Dataset
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

    # 循环训练
    epochs = 100
    running = 0.0
    best_val = float("inf")
    λ_dir = 1.0
    λ_conf = 1.0

    for epoch in range(epochs):
        print(f"Starting Training Epoch {epoch+1}/{epochs}:")
        feature_net.train()
        vae.train()
        running = 0.0

        # 降低lambda_conf的权重，逐步让模型专注于重构和方向一致性
        if epoch >= epochs // 2:
            for p in vae.conf_head.parameters():
                p.requires_grad_(False)
            λ_conf = 0.0  # 训练损失里去掉，验证仍可观察
        elif epoch >= epochs // 4:
            λ_conf = 0.01
        elif epoch >= epochs // 8:
            λ_conf = 0.1

        for img_batch, action_batch, prop_batch in train_dl:
            img_batch = img_batch.to(device)       # (B, 2C, H, W)
            action_batch = action_batch.to(device) # (B, A)
            prop_batch = prop_batch.to(device)     # (B, P)

            state_feat = feature_net(img_batch, prop_batch)  # (B, state_dim=feat_dim)
            # print("img_feat.shape", img_feat.shape)

            # 标准化动作
            action_batch_norm = (action_batch - act_mean_full) / act_std_full
            pred_norm, mu, std, a_direction, conf_logit = vae(state_feat, action_batch_norm)

            rec_loss = F.mse_loss(pred_norm, action_batch_norm)
            # KL = 0.5 * sum( std^2 + mu^2 - 1 - log(std^2) )
            log_std = std.clamp_min(1e-8).log()
            kl = -0.5 * torch.mean(torch.sum(1 + 2*log_std - mu.pow(2) - std.pow(2), dim=1))

            # 方向损失， 使得重构动作方向与原动作方向一致
            a_unit   = F.normalize(action_batch_norm, dim=1, eps=1e-8)
            cos_sim  = torch.sum(a_direction * a_unit, dim=1)            # cos(theta)
            direction_loss = (1.0 - cos_sim).mean()

            # 置信度损失，判别 (s, a_norm) 在流形上
            # 正样本标签 1
            y_pos = torch.ones_like(conf_logit)
            bce_pos = F.binary_cross_entropy_with_logits(conf_logit, y_pos)

            # 负样本：同一 batch 打乱动作
            idx_neg = torch.randperm(action_batch_norm.size(0), device=action_batch_norm.device)
            a_neg   = action_batch_norm[idx_neg]
            conf_neg = vae.conf_head(torch.cat([state_feat, a_neg], dim=1)).squeeze(1)
            y_neg = torch.zeros_like(conf_neg)
            bce_neg = F.binary_cross_entropy_with_logits(conf_neg, y_neg)
            conf_loss = 0.5 * (bce_pos + bce_neg)

            loss = rec_loss + kl + λ_dir * direction_loss + λ_conf * conf_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += loss.item()

        train_loss = running / max(1, len(train_dl))
        val_rec, val_kl, val_dir, val_conf = evaluate(feature_net, vae, val_dl, act_mean_full, act_std_full, device)
        monitor_val_loss =  val_rec + val_kl + λ_dir * val_dir
        val_loss = monitor_val_loss + λ_conf * val_conf
        print(f"[epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"monitor_val_loss={monitor_val_loss:.4f} | "
            f"val_rec={val_rec:.4f} val_kl={val_kl:.4f} "
            f"val_dir={val_dir:.4f} val_conf={val_conf:.4f}")

        # 调整学习率
        scheduler.step(monitor_val_loss)

        # 保存最好模型
        min_delta = 1e-3
        if monitor_val_loss < best_val - min_delta:
            best_val = monitor_val_loss
            torch.save({
                "feature_net": feature_net.state_dict(),
                "vae": vae.state_dict(),
                "dir_head": vae.dir_head.state_dict(),
                "dyn_head": vae.conf_head.state_dict(),
                "act_mean": act_mean_full.detach().cpu(),
                "act_std": act_std_full.detach().cpu(),
            }, os.path.join(ckpt_path, "vae_best.pt"))
            print(f"[SAVE] best checkpoint @ val={best_val:.4f}")