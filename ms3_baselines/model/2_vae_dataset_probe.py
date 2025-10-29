import os, h5py, numpy as np, torch, torch.nn.functional as F
from typing import Tuple

# from VAE import VAE, VAEManifoldScorer   # 如果你的类在 VAE.py 中
# 这里按你提供的定义占位；若你已能 import，请注释掉占位并启用 import。
# ---------- 占位（如已能 from VAE import，请删掉这段） ----------
import math, torch.nn as nn
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
        x = self.net(x).flatten(1)
        return self.fc(x)

class VAE(nn.Module):
    def __init__(self, in_ch, action_dim, latent_dim, max_action=1.0, feat_dim=256):
        super().__init__()
        self.max_action = max_action
        self.s_enc = SmallCNN(in_ch=in_ch, feat_dim=feat_dim)
        self.e1 = nn.Linear(feat_dim + action_dim, 512)
        self.e2 = nn.Linear(512, 512)
        self.mean = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)
        self.d1 = nn.Linear(feat_dim + latent_dim, 512)
        self.d2 = nn.Linear(512, 512)
        self.d3 = nn.Linear(512, action_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None: nn.init.zeros_(m.bias)
    def encode(self, s_img, a):
        s_feat = self.s_enc(s_img)
        h = torch.relu(self.e1(torch.cat([s_feat, a], dim=1)))
        h = torch.relu(self.e2(h))
        mu = self.mean(h)
        logvar = self.logvar(h).clamp(-8, 8)
        return mu, logvar, s_feat
    def decode(self, s_feat, z):
        h = torch.relu(self.d1(torch.cat([s_feat, z], dim=1)))
        h = torch.relu(self.d2(h))
        a = torch.tanh(self.d3(h)) * self.max_action
        return a
# ---------- 占位结束 ----------

class VAEManifoldScorer:
    def __init__(self, vae: VAE, device, k=8, img_size=96, ckpt_dir=None, sigma=None):
        self.vae = vae.eval()
        for p in self.vae.parameters(): p.requires_grad = False
        self.device, self.k, self.img_size = device, k, img_size
        self.act_mu  = torch.from_numpy(np.load(os.path.join(ckpt_dir, "action_mean.npy"))).to(device).float().squeeze(0)
        self.act_std = torch.from_numpy(np.load(os.path.join(ckpt_dir, "action_std.npy"))).to(device).float().squeeze(0)
        sig_path = os.path.join(ckpt_dir, "sigma.npy")
        if sigma is not None: self.sigma = float(sigma)
        elif os.path.exists(sig_path): self.sigma = float(np.load(sig_path)[0])
        else: self.sigma = 1.0
    @torch.no_grad()
    def _preprocess_obs(self, nhwc_rgb: torch.Tensor) -> torch.Tensor:
        x = nhwc_rgb.float().permute(0,3,1,2) / 255.0
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return x
    @torch.no_grad()
    def reward_and_dist(self, obs_dict: dict, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb = obs_dict["rgb"].to(self.device)
        s_img = self._preprocess_obs(rgb)
        a = action.to(self.device)
        a_std = (a - self.act_mu) / (self.act_std + 1e-12)
        mu, logvar, s_feat = self.vae.encode(s_img, a_std)
        recon = self.vae.decode(s_feat, mu)
        d_rec = ((recon - a_std)**2).mean(dim=1)  # (B,)
        d = d_rec
        if self.k > 0:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(self.k, a.size(0), mu.size(1), device=self.device)
            zs  = mu.unsqueeze(0) + eps * std.unsqueeze(0)
            s_rep = s_feat.unsqueeze(0).expand(self.k, -1, -1).reshape(-1, s_feat.size(1))
            cand  = self.vae.decode(s_rep, zs.reshape(-1, zs.size(-1))).view(self.k, a.size(0), -1)
            d_min = ((cand - a_std.unsqueeze(0))**2).mean(dim=2).min(dim=0)[0]
            d = 0.5 * d_rec + 0.5 * d_min
        r = torch.exp(- d / (self.sigma**2))
        return r, d_rec

def load_random_batch(h5_path, n=512, order="base_hand", device="cuda"):
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys()); assert keys, "Empty H5."
        k0 = keys[0]
        H,W,C = f[k0]["obs"]["sensor_data"]["base_camera"]["rgb"].shape[-3:]
        A = f[k0]["actions"].shape[-1]
        all_idx = []
        for k in keys:
            T = min(
                f[k]["obs"]["sensor_data"]["base_camera"]["rgb"].shape[0],
                f[k]["obs"]["sensor_data"]["hand_camera"]["rgb"].shape[0],
                f[k]["actions"].shape[0],
            )
            for t in range(T): all_idx.append((k,t))
        sel = np.random.choice(len(all_idx), size=min(n,len(all_idx)), replace=False)
        rgbs, acts = [], []
        for s in sel:
            tk,t = all_idx[s]
            base = f[tk]["obs"]["sensor_data"]["base_camera"]["rgb"][t]
            hand = f[tk]["obs"]["sensor_data"]["hand_camera"]["rgb"][t]
            a    = f[tk]["actions"][t]
            if order=="base_hand":
                rgb = np.concatenate([base, hand], axis=-1)     # (H,W,2C)
            elif order=="hand_base":
                rgb = np.concatenate([hand, base], axis=-1)
            elif order=="base_only_dup":
                rgb = np.concatenate([base, base], axis=-1)
            elif order=="hand_only_dup":
                rgb = np.concatenate([hand, hand], axis=-1)
            else:
                raise ValueError(order)
            rgbs.append(rgb); acts.append(a)
    rgb_t = torch.from_numpy(np.stack(rgbs,0)).to(device)
    act_t = torch.from_numpy(np.stack(acts,0)).float().to(device)
    return rgb_t, act_t, (H,W,C), A

def recompute_action_stats(h5_path) -> np.ndarray:
    xs = []
    with h5py.File(h5_path,"r") as f:
        for k in f.keys():
            xs.append(f[k]["actions"][:])
    A = np.concatenate(xs, axis=0)
    return A.mean(0, keepdims=True), A.std(0, keepdims=True) + 1e-6

def run_probe(
    ckpt_dir="/home/wzh-2004/RewardModelTest/VAE/ckpt_20251015_110910",
    h5_path="/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/StackCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5",
    device="cuda",
    latent_dim=32, feat_dim=256, img_size=96, n_samples=512
):
    # 0) 载入形状信息
    rgb, act, (H,W,C), A = load_random_batch(h5_path, n=8, order="base_hand", device=device)
    in_ch = 2*C

    # 1) 对比 ckpt 的动作统计 vs H5 现算
    ckpt_mu  = np.load(os.path.join(ckpt_dir,"action_mean.npy"))
    ckpt_std = np.load(os.path.join(ckpt_dir,"action_std.npy"))
    h5_mu, h5_std = recompute_action_stats(h5_path)
    print(f"[STATS] ckpt_mu[:4]={ckpt_mu[0,:4]}, h5_mu[:4]={h5_mu[0,:4]}")
    print(f"[STATS] ckpt_std[:4]={ckpt_std[0,:4]}, h5_std[:4]={h5_std[0,:4]}")
    print(f"[STATS] ||mu_diff||={np.linalg.norm(ckpt_mu-h5_mu):.4f}, ||std_diff||={np.linalg.norm(ckpt_std-h5_std):.4f}")

    # 2) 构建/加载 VAE + scorer
    vae = VAE(in_ch=in_ch, action_dim=A, latent_dim=latent_dim, max_action=1.0, feat_dim=feat_dim).to(device)
    vae.load_state_dict(torch.load(os.path.join(ckpt_dir, "vae_best.pt"), map_location=device))
    vae.eval(); [setattr(p, "requires_grad", False) for p in vae.parameters()]
    scorer = VAEManifoldScorer(vae, device=device, k=8, img_size=img_size, ckpt_dir=ckpt_dir)
    print(f"[LOAD] sigma={scorer.sigma:.6g}, in_ch={in_ch}, action_dim={A}")

    def once(order, override_std=False, sigma=None):
        rgb, act, *_ = load_random_batch(h5_path, n=n_samples, order=order, device=device)
        if override_std:
            # 跳过标准化：仅用于检验“训练没标准化但评分做了”的可能
            act_mu_bak, act_std_bak = scorer.act_mu.clone(), scorer.act_std.clone()
            scorer.act_mu.zero_(); scorer.act_std.fill_(1.0)
        if sigma is not None:
            sig_bak = scorer.sigma; scorer.sigma = float(sigma)
        with torch.no_grad():
            # 额外：打印 z 分布（标准化后）
            a_std = (act - scorer.act_mu) / (scorer.act_std + 1e-12)
            m, s = a_std.mean(0).abs().mean().item(), a_std.std(0).mean().item()
            r, d_rec = scorer.reward_and_dist({"rgb": rgb}, act)
        print(f"[{order} | override_std={override_std} | sigma={scorer.sigma:.4g}] "
              f"r_mean={r.mean().item():.4f}, r_med={r.median().item():.4f}, "
              f"r_min={r.min().item():.4f}, r_max={r.max().item():.4f} | "
              f"d_rec_mean={d_rec.mean().item():.4f} | "
              f"z(a) mean|std={m:.3f}|{s:.3f}")
        if override_std:
            scorer.act_mu.copy_(act_mu_bak); scorer.act_std.copy_(act_std_bak)
        if sigma is not None:
            scorer.sigma = sig_bak

    # 3) 相机顺序/单路复制 尝试
    once("base_hand")
    once("hand_base")
    once("base_only_dup")
    once("hand_only_dup")

    # 4) 跳过标准化尝试
    once("base_hand", override_std=True)

    # 5) sigma 扫描（如果只是 sigma 太小，r 会随 sigma 增大显著上升；但 d_rec 仍会显示是否重构失败）
    for sig in [0.1, 0.3, 1.0, 3.0, 9.0]:
        once("base_hand", sigma=sig)

if __name__ == "__main__":
    run_probe()
