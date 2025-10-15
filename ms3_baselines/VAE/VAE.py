import os
import math
import time
import random
import h5py
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import datetime


# =========================
#  Config
# =========================
@dataclass
class TrainCfg:
    h5_path: str = "/path/to/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"
    out_dir: str = "./output/vae_posmanifold"
    device: str = "cuda"
    img_size: int = 96
    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    pin_memory: bool = True
    amp: bool = True
    grad_clip: float = 1.0
    warmup_epochs: int = 2
    latent_dim: int = 32
    beta_kl_max: float = 1.0
    early_stop_patience: int = 7
    val_ratio: float = 0.05
    seed: int = 42
    action_dim_override: int | None = None
    max_trajs: int | None = None


# =========================
#  Utils (seed / lr schedule / warmup)
# =========================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # 性能优先


class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# =========================
#  HDF5 Dataset
# =========================
class TwoCamH5Dataset(Dataset):
    """
    每个样本 (state_t, action_t)；state_t = base/hand 两路 RGB 在通道维拼接 (2C,H,W)，
    图像归一化到 [0,1] 并 resize 到 img_size；动作做 (a - mu)/std 标准化（若提供）。
    """
    def __init__(self, h5_path, img_size=96, split_indices=None, max_trajs=None,
                 action_mean: np.ndarray | None = None, action_std: np.ndarray | None = None):
        super().__init__()
        self.h5_path = h5_path
        self.img_size = img_size
        self._h5 = None  # 延迟到 worker 中打开

        # 保存动作统计用于标准化（转为 torch，方便 __getitem__ 里广播）
        if action_mean is not None:
            self.action_mean = torch.from_numpy(action_mean.squeeze(0)).float()
            self.action_std  = torch.from_numpy(action_std.squeeze(0)).float()
        else:
            self.action_mean, self.action_std = None, None

        # 预扫描，建立 (traj_key, t) 索引
        self.indices = []  # list of (traj_key, t)
        with h5py.File(self.h5_path, "r") as f:
            keys = list(f.keys())
            if max_trajs is not None:
                keys = keys[:max_trajs]
            for k in keys:
                traj = f[k]
                base = traj["obs"]["sensor_data"]["base_camera"]["rgb"]   # (T, H, W, C)
                hand = traj["obs"]["sensor_data"]["hand_camera"]["rgb"]   # (T, H, W, C)
                acts = traj["actions"]                                     # (T, A)
                T = min(len(base), len(hand), len(acts))
                for t in range(T):
                    self.indices.append((k, t))

        if split_indices is not None:
            self.indices = split_indices

    def __len__(self):
        return len(self.indices)

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx):
        self._ensure_open()
        traj_key, t = self.indices[idx]
        traj = self._h5[traj_key]

        base = traj["obs"]["sensor_data"]["base_camera"]["rgb"][t]  # (H,W,C)
        hand = traj["obs"]["sensor_data"]["hand_camera"]["rgb"][t]  # (H,W,C)
        act  = traj["actions"][t]                                    # (A,)

        base = torch.from_numpy(base).permute(2,0,1).float() / 255.0
        hand = torch.from_numpy(hand).permute(2,0,1).float() / 255.0
        state = torch.cat([base, hand], dim=0)                        # (2C,H,W)

        if state.shape[-1] != self.img_size or state.shape[-2] != self.img_size:
            state = F.interpolate(state.unsqueeze(0), size=(self.img_size, self.img_size),
                                  mode="bilinear", align_corners=False).squeeze(0)

        action = torch.from_numpy(act).float()  # (A,)
        if self.action_mean is not None:
            action = (action - self.action_mean) / self.action_std
        return state, action


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def build_dataloaders(cfg: TrainCfg):
    # 先统计动作的均值/方差（全库或前 max_trajs 条）
    with h5py.File(cfg.h5_path, "r") as f:
        keys = list(f.keys())
        if cfg.max_trajs is not None:
            keys = keys[:cfg.max_trajs]
        xs = []
        for k in keys:
            xs.append(f[k]["actions"][:])     # (T, A)
        A = np.concatenate(xs, axis=0)        # (N, A)
    action_mean = A.mean(axis=0, keepdims=True)
    action_std  = A.std(axis=0, keepdims=True) + 1e-6

    # 保存标准化参数
    os.makedirs(cfg.out_dir, exist_ok=True)
    np.save(os.path.join(cfg.out_dir, "action_mean.npy"), action_mean)
    np.save(os.path.join(cfg.out_dir, "action_std.npy"), action_std)

    # 构建索引并划分
    full = TwoCamH5Dataset(cfg.h5_path, img_size=cfg.img_size, max_trajs=cfg.max_trajs,
                           action_mean=action_mean, action_std=action_std)
    N = len(full)
    val_N = max(1, int(N * cfg.val_ratio))
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    val_idx = idxs[:val_N].tolist()
    train_idx = idxs[val_N:].tolist()

    train_ds = TwoCamH5Dataset(cfg.h5_path, img_size=cfg.img_size,
                               split_indices=[full.indices[i] for i in train_idx],
                               max_trajs=cfg.max_trajs,
                               action_mean=action_mean, action_std=action_std)
    val_ds   = TwoCamH5Dataset(cfg.h5_path, img_size=cfg.img_size,
                               split_indices=[full.indices[i] for i in val_idx],
                               max_trajs=cfg.max_trajs,
                               action_mean=action_mean, action_std=action_std)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                              drop_last=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                            drop_last=False, worker_init_fn=worker_init_fn)

    # 动作维度、输入通道
    with h5py.File(cfg.h5_path, "r") as f:
        first_traj = f[list(f.keys())[0]]
        action_dim = int(first_traj["actions"].shape[-1])
        k0 = list(f.keys())[0]
        c = int(f[k0]["obs"]["sensor_data"]["base_camera"]["rgb"].shape[-1])
    if cfg.action_dim_override is not None:
        action_dim = cfg.action_dim_override
    in_ch = c * 2

    return train_loader, val_loader, in_ch, action_dim, action_mean, action_std


# =========================
#  Model: CNN encoder + Conditional VAE
# =========================
class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, feat_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),   nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, feat_dim)

    def forward(self, x):
        x = self.net(x)           # (B,256,1,1)
        x = x.flatten(1)          # (B,256)
        x = self.fc(x)            # (B,feat_dim)
        return x


class VAE(nn.Module):
    """
    条件 VAE：
      Encoder: [s_feat ⊕ a] -> (mu, logvar)
      Decoder: [s_feat ⊕ z] -> â
    """
    def __init__(self, in_ch, action_dim, latent_dim, max_action=1.0, feat_dim=256):
        super().__init__()
        self.max_action = max_action
        self.latent_dim = latent_dim

        self.s_enc = SmallCNN(in_ch=in_ch, feat_dim=feat_dim)

        # Encoder MLP
        self.e1 = nn.Linear(feat_dim + action_dim, 512)
        self.e2 = nn.Linear(512, 512)
        self.mean = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

        # Decoder MLP
        self.d1 = nn.Linear(feat_dim + latent_dim, 512)
        self.d2 = nn.Linear(512, 512)
        self.d3 = nn.Linear(512, action_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, s_img, a):
        s_feat = self.s_enc(s_img)                      # (B, feat_dim)
        h = torch.relu(self.e1(torch.cat([s_feat, a], dim=1)))
        h = torch.relu(self.e2(h))
        mu = self.mean(h)
        logvar = self.logvar(h).clamp(-8, 8)
        return mu, logvar, s_feat

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, s_feat, z):
        h = torch.relu(self.d1(torch.cat([s_feat, z], dim=1)))
        h = torch.relu(self.d2(h))
        a = torch.tanh(self.d3(h)) * self.max_action
        return a

    def forward(self, s_img, a):
        mu, logvar, s_feat = self.encode(s_img, a)
        z = self.reparam(mu, logvar)
        rec = self.decode(s_feat, z)
        return rec, mu, logvar, s_feat


# =========================
#  Training / Validation
# =========================
def train_one_epoch(model, loader, opt, scaler, epoch, total_epochs, cfg, scheduler=None):
    model.train()
    device = cfg.device
    running = 0.0

    # KL 退火：从 0 -> beta_kl_max
    beta = min(cfg.beta_kl_max, (epoch + 1) / max(1, cfg.warmup_epochs)) * cfg.beta_kl_max

    for i, (s_img, a) in enumerate(loader):
        s_img = s_img.to(device, non_blocking=True)
        a     = a.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            rec, mu, logvar, _ = model(s_img, a)
            rec_loss = F.mse_loss(rec, a)
            kl = 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))
            loss = rec_loss + beta * kl

        scaler.scale(loss).backward()
        if cfg.grad_clip is not None:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running += loss.item()

    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    device = cfg.device
    total = 0.0
    for s_img, a in loader:
        s_img = s_img.to(device, non_blocking=True)
        a     = a.to(device, non_blocking=True)
        rec, mu, logvar, _ = model(s_img, a)
        rec_loss = F.mse_loss(rec, a)
        kl = 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))
        loss = rec_loss + cfg.beta_kl_max * kl
        total += loss.item()
    return total / max(1, len(loader))


class VAEManifoldScorer:
    """
    使用训练好的 VAE 计算“动作到流形”的相似度奖励。
    - 读入 action_mean/std 和 sigma（训练阶段保存）
    - 采样围绕后验 z（而不是先验），更贴近流形
    """
    def __init__(self, vae: VAE, device, k=8, img_size=96, ckpt_dir=None,
                 sigma: float | None = None):
        self.vae = vae.eval()
        for p in self.vae.parameters(): p.requires_grad = False
        self.device = device
        self.k = k
        self.img_size = img_size

        assert ckpt_dir is not None, "Please provide ckpt_dir to load action_mean/std (and sigma)."
        self.act_mu  = torch.from_numpy(np.load(os.path.join(ckpt_dir, "action_mean.npy"))).to(device).float().squeeze(0)
        self.act_std = torch.from_numpy(np.load(os.path.join(ckpt_dir, "action_std.npy"))).to(device).float().squeeze(0)

        sigma_path = os.path.join(ckpt_dir, "sigma.npy")
        if sigma is None and os.path.exists(sigma_path):
            self.sigma = float(np.load(sigma_path)[0])
        elif sigma is not None:
            self.sigma = float(sigma)
        else:
            print("[VAEManifoldScorer] no sigma provided or found; default to 1.0.")
            # 回退：若没有标定文件，则设为 1.0（建议尽快用验证集标定）
            self.sigma = 1.0

    @torch.no_grad()
    def _preprocess_obs(self, obs_rgb: torch.Tensor) -> torch.Tensor:
        x = obs_rgb.float().permute(0, 3, 1, 2) / 255.0  # (B,C,H,W)
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return x

    @torch.no_grad()
    def reward(self, obs_dict: dict, action: torch.Tensor) -> torch.Tensor:
        rgb = obs_dict["rgb"].to(self.device)
        s_img = self._preprocess_obs(rgb)

        # 动作标准化（与训练一致）
        a = action.to(self.device)
        a = (a - self.act_mu) / self.act_std

        # 编码
        mu, logvar, s_feat = self.vae.encode(s_img, a)
        # 自编码重构（均值重构更稳）
        recon = self.vae.decode(s_feat, mu)
        d_rec = ((recon - a)**2).mean(dim=1)  # (B,)

        # 围绕后验采样 K 个候选，并取最近距离
        if self.k > 0:
            std = torch.exp(0.5 * logvar)                         # (B,Z)
            eps = torch.randn(self.k, a.size(0), mu.size(1), device=self.device)
            zs  = mu.unsqueeze(0) + eps * std.unsqueeze(0)        # (K,B,Z)

            # 向量化解码
            s_rep = s_feat.unsqueeze(0).expand(self.k, -1, -1).reshape(-1, s_feat.size(1))
            cand  = self.vae.decode(s_rep, zs.reshape(-1, zs.size(-1)))  # (K*B, A)
            cand  = cand.view(self.k, a.size(0), -1)                     # (K,B,A)

            d_min = ((cand - a.unsqueeze(0))**2).mean(dim=2).min(dim=0)[0]  # (B,)
            d = 0.5 * d_rec + 0.5 * d_min
        else:
            d = d_rec

        r = torch.exp(- d / (self.sigma**2))
        return r


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = TrainCfg(
        h5_path="/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/StackCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5",
        out_dir=f"/home/wzh-2004/RewardModelTest/VAE/ckpt_{timestamp}",
        device="cuda",
        latent_dim=32,
        img_size=96,
        batch_size=128,
        epochs=40,
        lr=1e-3,
        num_workers=4,
        warmup_epochs=2,
        beta_kl_max=1.0,
        early_stop_patience=6,
        max_trajs=None,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    train_loader, val_loader, in_ch, action_dim, action_mean, action_std = build_dataloaders(cfg)
    print(f"[Data] in_ch={in_ch}, action_dim={action_dim}, "
          f"train_batches={len(train_loader)}, val_batches={len(val_loader)}")

    model = VAE(in_ch=in_ch, action_dim=action_dim, latent_dim=cfg.latent_dim,
                max_action=1.0, feat_dim=256).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)
    scheduler = CosineWithWarmup(opt, warmup_steps=warmup_steps, total_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val = float("inf")
    patience = 0

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, opt, scaler, epoch, cfg.epochs, cfg, scheduler)
        val_loss = evaluate(model, val_loader, cfg)
        dt = time.time() - t0
        print(f"[Epoch {epoch+1:03d}/{cfg.epochs}] train={train_loss:.4f}  val={val_loss:.4f}  "
              f"beta={min(cfg.beta_kl_max, (epoch+1)/max(1,cfg.warmup_epochs))*cfg.beta_kl_max:.3f}  {dt:.1f}s")

        # 保存最优
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(cfg.out_dir, "vae_best.pt"))
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"[EarlyStop] no improvement for {cfg.early_stop_patience} epochs.")
                break

    # 末尾再存一份最后权重
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "vae_last.pt"))
    print(f"[Done] Best val loss: {best_val:.4f}. Models saved to {cfg.out_dir}")

    # ========= 标定 sigma（基于验证集的重构距离）=========
    model.eval()
    device = cfg.device
    d_vals = []
    with torch.no_grad():
        for s_img, a in val_loader:
            s_img = s_img.to(device, non_blocking=True)
            a     = a.to(device, non_blocking=True)
            rec, mu, logvar, s_feat = model(s_img, a)
            d_rec = ((rec - a)**2).mean(dim=1)  # (B,)
            d_vals.append(d_rec.cpu().numpy())
    if len(d_vals) > 0:
        d_all = np.concatenate(d_vals, axis=0)
        sigma = float(np.median(d_all))  # 也可以用 np.percentile(d_all, 90)
    else:
        sigma = 1.0
    np.save(os.path.join(cfg.out_dir, "sigma.npy"), np.array([sigma], dtype=np.float32))
    print(f"[Sigma] calibrated sigma={sigma:.6f} and saved to {cfg.out_dir}")


def test():
    pass


if __name__ == "__main__":
    main()
    test()
