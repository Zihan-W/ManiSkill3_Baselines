import h5py


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split, Subset

import numpy as np
import os
from datetime import datetime

# 实现简单的 Dataset 读取 HDF5 格式轨迹数据
class SimpleH5Dataset(Dataset):
    """
    最小实现：读取 HDF5 轨迹数据
    目录假设：
      <traj_key>/obs/sensor_data/base_camera/rgb -> (T, H, W, C) uint8
      <traj_key>/actions                         -> (T, A)     float
    返回：
      image: (C,H,W) float32 in [0,1]
      action: (A,)    float32
    """
    def __init__(self, h5_path: str):
        super().__init__()
        self.h5_path = h5_path
        self._h5 = None   # 懒打开（兼容多进程 DataLoader）

        # 预扫描所有 (traj_key, t)
        self.indices = []
        with h5py.File(self.h5_path, "r") as f:
            for k in f.keys():
                traj_single = f[k]
                _base_camera = traj_single["obs/sensor_data/base_camera/rgb"]   # (T,H,W,C)
                _hand_camera = traj_single["obs/sensor_data/hand_camera/rgb"]   # (T,H,W,C)
                _actions = traj_single["actions"]   # (T,A)
                T = min(len(_base_camera), len(_hand_camera), len(_actions))
                for t in range(T):
                    self.indices.append((k, t))

    def __len__(self):
        return len(self.indices)

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx: int):
        self._ensure_open()
        traj_key, t = self.indices[idx]
        _traj_single = self._h5[traj_key]

        # 读图像 -> (C,H,W) float32 in [0,1]
        _base_camera = _traj_single["obs/sensor_data/base_camera/rgb"][t]   # (H,W,C), uint8
        _hand_camera = _traj_single["obs/sensor_data/hand_camera/rgb"][t]   # (H,W,C), uint8
        # torch 到 tensor (H, W, C), int
        _base_camera = torch.from_numpy(_base_camera)
        _hand_camera = torch.from_numpy(_hand_camera)
        # <torch.Tensor> -> (C,H,W), float32 in [0,1]
        _base_camera = _base_camera.permute(2,0,1).float() / 255.0  # (C,H,W), float32 in [0,1]
        _hand_camera = _hand_camera.permute(2,0,1).float() / 255.0  # (C,H,W), float32 in [0,1]
        img = torch.cat([_base_camera, _hand_camera], dim=0)  # (2C,H,W)

        # 读动作 -> (A,) float32
        _actions = _traj_single["actions"][t]                                # (A,)
        actions = torch.from_numpy(_actions).float()

        return img, actions

def make_loader(h5_path: str, batch_size: int = 128, shuffle: bool = True,
            num_workers: int = 0):
    persistent = (num_workers > 0)
    prefetch = 4 if num_workers > 0 else None
    ds = SimpleH5Dataset(h5_path=h5_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                    num_workers=num_workers, pin_memory=True,
                    persistent_workers=persistent, prefetch_factor=prefetch)
    return ds, dl

# 实现简单的CNN编码器
class SmallCNN(nn.Module):
    def __init__(self, in_ch=6, feat_dim=256):
        super().__init__()
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),   nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, feat_dim)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x)

# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
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
        z = self.encode(state, action)

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std


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
    - 如果 stats_path 是目录：自动保存为 <stats_path>/action_stats.pt
    - 如果 stats_path 是文件：直接用该文件名
    - 若文件已存在则加载返回
    """
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 规范化保存路径
    save_path = None
    if stats_path is not None:
        if os.path.isdir(stats_path):
            save_path = os.path.join(stats_path, "action_stats.pt")
        else:
            # 既可能是“不存在的文件名”，也可能是“已存在的文件”
            dirpath = os.path.dirname(stats_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            save_path = stats_path

        # 尝试读取缓存
        if os.path.exists(save_path):
            try:
                data = torch.load(save_path, map_location="cpu")
                if isinstance(data, dict) and "mean" in data and "std" in data:
                    return data["mean"].float(), data["std"].float()
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    return torch.as_tensor(data[0]).float(), torch.as_tensor(data[1]).float()
            except Exception as e:
                print(f"[WARN] Failed to load cached stats from {save_path}: {e}")

    # 计算
    tmp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=(num_workers > 0))
    n = 0
    s1 = None
    s2 = None
    with torch.no_grad():
        for _, a in tmp_loader:
            if use_cuda:
                a = a.to(device, non_blocking=True)
            a = a.float()
            s1 = a.sum(dim=0) if s1 is None else s1 + a.sum(dim=0)
            s2 = (a * a).sum(dim=0) if s2 is None else s2 + (a * a).sum(dim=0)
            n += a.shape[0]

    # compute mean/var on the same device as s1/s2 to avoid mixing CPU/GPU tensors
    if s1 is None:
        raise RuntimeError("No data found in dataset to compute action stats")
    mean_dev = s1 / n
    var_dev = s2 / n - mean_dev ** 2
    # move to CPU for saving/return
    mean = mean_dev.cpu()
    var = var_dev.cpu()
    std  = torch.sqrt(var.clamp_min(1e-12))

    # 保存
    if save_path is not None:
        try:
            torch.save({"mean": mean, "std": std}, save_path)
            print(f"[OK] Saved action stats to {save_path}")
        except Exception as e:
            print(f"[WARN] Failed to save stats to {save_path}: {e}")

        # 额外存 .npy 方便别处用
        try:
            np.save(os.path.splitext(save_path)[0] + "_mean.npy", mean.numpy())
            np.save(os.path.splitext(save_path)[0] + "_std.npy",  std.numpy())
        except Exception as e:
            print(f"[WARN] Failed to save npy: {e}")

    return mean, std

@torch.no_grad()
def evaluate(feature_net, vae, val_loader, act_mean, act_std, device):
    feature_net.eval(); vae.eval()
    rec_total, kl_total, n_batch = 0.0, 0.0, 0
    for img_batch, action_batch in val_loader:
        img_batch = img_batch.to(device)
        action_batch = action_batch.to(device)

        img_feat = feature_net(img_batch)
        action_norm = (action_batch - act_mean) / act_std

        pred_norm, mu, std = vae(img_feat, action_norm)

        # 重构 + KL（与训练一致的公式）
        rec_loss = F.mse_loss(pred_norm, action_norm)
        log_std = std.clamp_min(1e-8).log()
        kl = -0.5 * torch.mean(torch.sum(1 + 2*log_std - mu.pow(2) - std.pow(2), dim=1))

        rec_total += rec_loss.item()
        kl_total  += kl.item()
        n_batch   += 1

    return rec_total / max(1, n_batch), kl_total / max(1, n_batch)

if __name__ == "__main__":
    h5_path = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/StackCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"
    base_ckpt_dir = os.path.join(os.path.dirname(__file__), "ckpt_VAE")
    stats_path = base_ckpt_dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_path = os.path.join(base_ckpt_dir, timestamp)
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"ckpt path: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集和数据加载器
    ds, dl = make_loader(h5_path, num_workers=8)

    # 划分训练集和验证集
    val_ratio = 0.1  # 10% 做验证
    N = len(ds)
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    val_size = max(1, int(N * val_ratio))
    val_idxs = idxs[:val_size].tolist()
    train_idxs = idxs[val_size:].tolist()

    train_ds = Subset(ds, train_idxs)
    val_ds   = Subset(ds, val_idxs)

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,
                        num_workers=8, pin_memory=True,
                        persistent_workers=True, prefetch_factor=4)
    val_dl   = DataLoader(val_ds, batch_size=128, shuffle=False,
                        num_workers=8, pin_memory=True,
                        persistent_workers=True, prefetch_factor=4)

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
    act_mean_full, act_std_full = compute_action_stats(train_ds, stats_path=stats_path)   # ds 是上面 make_loader 返回的 Dataset
    act_mean_full = act_mean_full.to(device)
    act_std_full  = act_std_full.to(device)
    act_mean_full.requires_grad_(False)
    act_std_full.requires_grad_(False)
    print(f"[act stats] mean[:4]={act_mean_full[:4].tolist()}, std[:4]={act_std_full[:4].tolist()}")

    # 设置优化器
    params = list(feature_net.parameters()) + list(vae.parameters())
    optim = torch.optim.Adam(params, lr=1e-3)

    # 循环训练
    epochs = 100
    running = 0.0
    best_val = float("inf")

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
            action_batch = (action_batch - act_mean_full) / act_std_full

            pred_norm, mu, std = vae(img_feat, action_batch)

            rec_loss = F.mse_loss(pred_norm, action_batch)
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
        val_loss = val_rec + val_kl
        print(f"[epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} | val_rec={val_rec:.4f} val_kl={val_kl:.4f} val_sum={val_loss:.4f}")

        # 保存最好模型
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            torch.save({
                "feature_net": feature_net.state_dict(),
                "vae": vae.state_dict(),
                "act_mean": act_mean_full.detach().cpu(),
                "act_std": act_std_full.detach().cpu(),
            }, os.path.join(ckpt_path, "vae_best.pt"))
            print(f"[SAVE] best checkpoint @ val={best_val:.4f}")