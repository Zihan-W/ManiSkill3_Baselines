import os
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from utils import make_loader, split_by_trajectory, compute_action_stats
from utils import SmallCNN

# --------- MDN 主体 ----------
class MDN(nn.Module):
    """混合密度网络：在标准化动作空间上直接输出多峰高斯混合。
    输入： state 特征 (B, state_dim)
    输出：logits (B,K), mu (B,K,A), logstd (B,K,A)
    """
    def __init__(self, state_dim: int, action_dim: int, K: int = 10, hidden: int = 512, p_drop: float = 0.1):
        super().__init__()
        self.K, self.A = K, action_dim
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),    nn.ReLU(inplace=True), nn.Dropout(p_drop),
        )
        self.logits = nn.Linear(hidden, K)
        self.mu     = nn.Linear(hidden, K * action_dim)
        self.logstd = nn.Linear(hidden, K * action_dim)

        # CHANGED: 统一管理 logσ 的上下界
        self.LOGSTD_MIN = -2.5
        self.LOGSTD_MAX =  3.0

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(s)
        logits = self.logits(h)                                # (B,K)
        mu     = self.mu(h).view(-1, self.K, self.A)           # (B,K,A)
        logstd = self.logstd(h).view(-1, self.K, self.A)       # (B,K,A)
        # CHANGED: 抬高下界，缓解尖峰
        logstd = logstd.clamp(self.LOGSTD_MIN, self.LOGSTD_MAX)
        return logits, mu, logstd


def mdn_logp(a_norm: torch.Tensor, logits: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    """计算 log p_ID(a|s)（标准化动作空间）
    a_norm: (B,A), logits:(B,K), mu/logstd:(B,K,A)
    返回: (B,)
    """
    B,K,A = mu.shape
    a = a_norm.unsqueeze(1).expand(B, K, A)  # (B,K,A)
    inv_var = torch.exp(-2.0 * logstd)
    log_comp = -0.5 * torch.sum((a - mu) ** 2 * inv_var + 2.0 * logstd + np.log(2.0 * np.pi), dim=2)  # (B,K)
    log_mix = torch.log_softmax(logits, dim=1)  # (B,K)
    return torch.logsumexp(log_mix + log_comp, dim=1)  # (B,)

@torch.no_grad()
def evaluate(feature_net: nn.Module, mdn: nn.Module, val_loader, act_mean: torch.Tensor, act_std: torch.Tensor, device) -> Tuple[float, float]:
    """验证集评估：返回 (NLL, REG) 的平均值（REG=小σ惩罚+熵正则权重项），便于 ReduceLROnPlateau 对接。"""
    feature_net.eval(); mdn.eval()
    act_mean = act_mean.to(device)
    act_std  = torch.clamp(act_std.to(device), min=1e-3)

    # CHANGED: 与训练保持一致的系数
    small_sigma_coef = 5e-4   # 可在 5e-4 ~ 1e-3 之间微调
    ent_coef         = 1e-3

    nll_total, reg_total, n = 0.0, 0.0, 0
    for img_batch, action_batch in val_loader:
        img_batch = img_batch.to(device, non_blocking=True).float()
        if img_batch.max() > 1.1 or img_batch.min() < 0.0:
            img_batch = img_batch / 255.0
        action_batch = action_batch.to(device, non_blocking=True).float()

        s = feature_net(img_batch)
        a_norm = (action_batch - act_mean) / act_std

        logits, mu, logstd = mdn(s)
        logp = mdn_logp(a_norm, logits, mu, logstd)
        nll  = -(logp.mean())

        # CHANGED: 惩罚小 σ，而不是大 σ
        small_sigma_reg = small_sigma_coef * torch.exp(-2.0 * logstd).mean()

        # CHANGED: 轻微鼓励混合权重有熵，防止只用 1 个分量
        pi  = torch.softmax(logits, dim=1)
        ent = -(pi * (pi.clamp_min(1e-12).log())).sum(dim=1).mean()
        ent_reg = - ent_coef * ent

        nll_total += nll.item()
        reg_total += (small_sigma_reg + ent_reg).item()
        n += 1

    return nll_total / max(n, 1), reg_total / max(n, 1)


# --------- 推理奖励：密度比 + MC Dropout 不确定性惩罚 ----------
class MDNReward:
    def __init__(self, feature_net: nn.Module, mdn: nn.Module, act_mean: torch.Tensor, act_std: torch.Tensor,
                 device: torch.device, mc_T: int = 5, lambda_epi: float = 0.2):
        self.f = feature_net.eval()
        self.mdn = mdn.eval()
        self.act_mean = act_mean.to(device)
        self.act_std  = torch.clamp(act_std.to(device), min=1e-3)
        self.device = device
        self.mc_T = max(1, mc_T)
        self.lambda_epi = float(lambda_epi)

    @torch.no_grad()
    def _prep(self, img, action):
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.ndim == 4 and img.shape[1] not in (1,2,3,4,6):  # 可能是 BHWC
            img = img.permute(0,3,1,2)
        img = img.float()
        if img.max() > 1.1 or img.min() < 0.0:
            img = img / 255.0
        img = img.to(self.device)
        s = self.f(img)

        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        a = action.float().to(self.device)
        a_norm = (a - self.act_mean) / self.act_std
        return s, a_norm

    @torch.no_grad()
    def compute_reward(self, img, action):
        s, a_norm = self._prep(img, action)

        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        self.mdn.apply(enable_dropout)

        logp_list = []
        for _ in range(self.mc_T):
            logits, mu, logstd = self.mdn(s)
            logp_list.append(mdn_logp(a_norm, logits, mu, logstd))  # (B,)
        logp_stack = torch.stack(logp_list, dim=0)  # (T,B)
        logp_id = logp_stack.mean(dim=0)            # (B,)
        var_epi = logp_stack.var(dim=0)             # (B,)

        A = a_norm.shape[1]
        logp_bg = -0.5 * (a_norm.pow(2).sum(dim=1) + A * np.log(2.0 * np.pi))  # (B,)

        reward = (logp_id - logp_bg) - self.lambda_epi * var_epi
        return reward.detach().cpu().numpy()

@torch.no_grad()
def debug_decompose_traj(ds, traj_indices, rewarder, device, floor=-2.5):
    mdn = rewarder.mdn
    f   = rewarder.f
    act_mean, act_std = rewarder.act_mean, rewarder.act_std
    A = act_mean.numel()
    logp_id_list, logp_bg_list, var_mc_list = [], [], []
    floor_hits = []
    for start in range(0, len(traj_indices), 512):
        idxs = traj_indices[start:start+512]
        imgs = torch.stack([ds[i][0] for i in idxs], 0).to(device).float()
        acts = torch.stack([ds[i][1] for i in idxs], 0).to(device).float()
        imgs = imgs/255.0 if imgs.max()>1.1 else imgs
        s = f(imgs)
        a_norm = (acts - act_mean) / act_std

        # MC for logp_id / var
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        mdn.apply(enable_dropout)
        logs = []
        for _ in range(max(1, rewarder.mc_T)):
            logits, mu, logstd = mdn(s)
            logp = mdn_logp(a_norm, logits, mu, logstd)
            logs.append(logp)
            # 统计 σ 是否贴下界
            floor_hits.append((logstd < floor + 0.1).float().mean().cpu())
        logp_id = torch.stack(logs, 0).mean(0)
        var_mc  = torch.stack(logs, 0).var(0)

        logp_bg = -0.5 * (a_norm.pow(2).sum(dim=1) + A*np.log(2*np.pi))
        logp_id_list.append(logp_id.cpu())
        logp_bg_list.append(logp_bg.cpu())
        var_mc_list.append(var_mc.cpu())

    import matplotlib.pyplot as plt
    logp_id = torch.cat(logp_id_list).numpy()
    logp_bg = torch.cat(logp_bg_list).numpy()
    var_mc  = torch.cat(var_mc_list).numpy()
    r = logp_id - logp_bg - rewarder.lambda_epi * var_mc
    t = np.arange(len(r))

    plt.figure(figsize=(12,6))
    plt.plot(t, r, label='reward')
    plt.plot(t, logp_id, label='logp_id', alpha=0.7)
    plt.plot(t, -logp_bg, label='-logp_bg', alpha=0.7)
    plt.plot(t, rewarder.lambda_epi*var_mc, label='λ·Var_MC', alpha=0.7)
    plt.legend(); plt.grid(True); plt.title('Decomposition'); plt.show()

    floor_ratio = torch.stack(floor_hits).mean().item()
    print(f"logstd@floor ratio (~<{floor+0.1}): {floor_ratio:.3f}")
    return r, logp_id, logp_bg, var_mc

def testing():
    from configs.cfg_parser import parse_cfg
    cfg = parse_cfg('MDN')

    # --- 路径与参数 ---
    h5_path        = cfg["paths"]["h5_dataset_path"]
    load_model_path= cfg["paths"]["load_model_path"]
    save_eval_dir  = "runs/test_mdn_reward"  # 临时保存目录
    eval_cfg       = cfg.get("eval", {})
    traj_id        = eval_cfg.get("traj_id", 0)
    batch_size     = int(eval_cfg.get("batch_size", 512))
    mc_T           = int(eval_cfg.get("mc_T", 5))
    lambda_epi     = float(eval_cfg.get("lambda_epi", 0.2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_eval_dir, exist_ok=True)

    # --- 数据集（single 模式足够；这里不需要 DataLoader）---
    ds, _ = make_loader(h5_path, mode="single", img_size=None, num_workers=0, shuffle=False)

    # --- 维度推断 & 模型构建 ---
    img0, act0 = ds[0]  # (C,H,W), (A,)
    feature_net = SmallCNN(in_ch=img0.shape[0], feat_dim=256).to(device)
    state_dim = feature_net.feat_dim
    action_dim = act0.shape[0]

    K       = getattr(getattr(cfg, "model", object()), "K", 10)
    hidden  = getattr(getattr(cfg, "model", object()), "hidden", 512)
    p_drop  = getattr(getattr(cfg, "model", object()), "dropout", 0.1)
    mdn = MDN(state_dim=state_dim, action_dim=action_dim, K=K, hidden=hidden, p_drop=p_drop).to(device)

    # --- 加载权重 ---
    sd = torch.load(load_model_path, map_location=device)
    feature_net.load_state_dict(sd['feature_net'])
    mdn.load_state_dict(sd['mdn'])
    act_mean, act_std = sd['act_mean'], sd['act_std']

    # --- 构建奖励器 ---
    rewarder = MDNReward(feature_net, mdn, act_mean, act_std, device, mc_T=mc_T, lambda_epi=lambda_epi)

    # --- 选轨迹并取索引 ---
    traj_key = f"traj_{traj_id}"
    if hasattr(ds, "get_traj_indices") and hasattr(ds, "traj_lengths") and traj_key in ds.traj_lengths:
        traj_indices = ds.get_traj_indices(traj_key)
    else:
        # 兜底：从 ds.indices 里筛
        keys = getattr(ds, "traj_lengths", {}).keys()
        if traj_key not in keys:
            # 若给的 traj_id 不存在，取第一条
            traj_key = sorted(list(keys))[0]
        traj_indices = [i for i, (k, t) in enumerate(ds.indices) if k == traj_key]
        traj_indices = sorted(traj_indices, key=lambda i: ds.indices[i][1])

    if len(traj_indices) == 0:
        print(f"[Eval] No samples found for {traj_key}.")
        return

    # --- 分批计算 reward（防 OOM） ---
    rewards_list = []
    for start in range(0, len(traj_indices), batch_size):
        idxs = traj_indices[start:start + batch_size]
        imgs = torch.stack([ds[i][0] for i in idxs], dim=0)   # (B, C, H, W)
        acts = torch.stack([ds[i][1] for i in idxs], dim=0)   # (B, A)
        rewards = rewarder.compute_reward(imgs, acts)         # (B,)
        rewards_list.append(rewards)

    rewards = np.concatenate(rewards_list, axis=0)  # (T,)
    timesteps = np.arange(len(rewards))

    # --- 可视化与保存 ---
    out_dir = os.path.join(save_eval_dir, "eval_viz")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(timesteps, rewards, marker='o', linewidth=1)
    plt.xlabel("Timestep")
    plt.ylabel("Reward (log p_ID - log p_BG - λ·Var_MC)")
    plt.title(f"Trajectory {traj_key} Rewards (K={K}, mc_T={mc_T}, λ={lambda_epi})")
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

    debug_decompose_traj(ds, traj_indices, rewarder, device)


def main():
    from configs.cfg_parser import parse_cfg
    cfg_name = 'MDN'
    cfg = parse_cfg(cfg_name) if cfg_name is not None else parse_cfg()
    # 加载路径
    h5_path = cfg["paths"]["h5_dataset_path"]
    save_ckpt_dir = cfg["paths"]["save_ckpt_dir"]
    print(f"save_ckpt_dir: {save_ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集和数据加载器
    ds, dl = make_loader(h5_path, mode="single", img_size=None, num_workers=16)
    pair_ds, pair_dl = make_loader(h5_path, mode="pair", batch_size=32, num_workers=16)

    # 划分训练集和验证集
    train_ds, val_ds = split_by_trajectory(ds, val_ratio=0.1, seed=42)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_dl   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    # 取出一条样本，用于初始化网络
    img, act = ds[0]          # 第 0 条样本 (img: (2C,H,W), act: (A,))
    feature_net = SmallCNN(in_ch=img.shape[0], feat_dim=256).to(device)
    state_dim = feature_net.feat_dim
    action_dim = act.shape[0]
    latent_dim = 32
    max_action = 1.0
    model = MDN(
        state_dim=state_dim,
        action_dim=action_dim,
        K=cfg.model.K if hasattr(cfg, "model") and hasattr(cfg.model, "K") else 10,
        hidden=cfg.model.hidden if hasattr(cfg, "model") and hasattr(cfg.model, "hidden") else 512,
        p_drop=cfg.model.dropout if hasattr(cfg, "model") and hasattr(cfg.model, "dropout") else 0.1,
    ).to(device)
    print("successfully create MDN")
    print(f"MDN parameters: state_dim={state_dim}, action_dim={action_dim}, latent_dim={latent_dim}, max_action={max_action}, device={device}")

    # 计算动作均值和标准差
    act_mean_full, act_std_full = compute_action_stats(train_ds)   # ds 是上面 make_loader 返回的 Dataset
    act_mean_full = act_mean_full.to(device)
    act_std_full  = act_std_full.to(device)
    act_mean_full.requires_grad_(False)
    act_std_full.requires_grad_(False)
    print(f"[act stats] mean[:4]={act_mean_full[:4].tolist()}, std[:4]={act_std_full[:4].tolist()}")

    # 设置优化器
    params = list(feature_net.parameters()) + list(model.parameters())
    optim = torch.optim.Adam(params, lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

    # --- Logger ---
    import wandb

    wandb.init(project=cfg['log']['wandb_project'],
               group=cfg['log']['wandb_group'],
               config=cfg)

    # ===== Train Loop (REPLACE THIS BLOCK) =====
    small_sigma_coef = 5e-4   # 惩罚 σ 过小；可在 5e-4 ~ 1e-3 间调
    ent_coef         = 1e-3   # 轻微鼓励混合权重有熵
    grad_clip        = 1.0

    epochs = 100
    best_val = float('inf')
    train_loss_history, val_loss_history = [], []

    for epoch in range(epochs):
        feature_net.train(); model.train()
        running = 0.0

        for img_batch, action_batch in train_dl:
            img_batch = img_batch.to(device, non_blocking=True).float()
            if img_batch.max() > 1.1 or img_batch.min() < 0.0:
                img_batch = img_batch / 255.0
            action_batch = action_batch.to(device, non_blocking=True).float()

            s = feature_net(img_batch)
            a_norm = (action_batch - act_mean_full) / torch.clamp(act_std_full, min=1e-3)

            logits, mu, logstd = model(s)                   # MDN 前向
            logp = mdn_logp(a_norm, logits, mu, logstd)
            nll  = -logp.mean()

            small_sigma_reg = small_sigma_coef * torch.exp(-2.0 * logstd).mean()
            pi  = torch.softmax(logits, dim=1)
            ent = -(pi * (pi.clamp_min(1e-12).log())).sum(dim=1).mean()
            loss = nll + small_sigma_reg - ent_coef * ent

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(feature_net.parameters())+list(model.parameters()), grad_clip)
            optim.step()

            running += loss.item()

        train_loss = running / max(1, len(train_dl))
        val_nll, val_reg = evaluate(feature_net, model, val_dl, act_mean_full, act_std_full, device)
        monitor_val = val_nll + val_reg

        train_loss_history.append(train_loss)
        val_loss_history.append(monitor_val)
        scheduler.step(monitor_val)

        print(f"[epoch {epoch+1}/{epochs}] train_total={train_loss:.4f} | val={monitor_val:.4f} | "
            f"val_nll={val_nll:.4f} val_reg={val_reg:.4f} | lr={optim.param_groups[0]['lr']:.2e}")
        wandb.log({
            'epoch': epoch + 1,
            'train/total_loss': train_loss,
            'val/total_loss': monitor_val,
            'val/nll': val_nll,
            'val/reg': val_reg,
            'lr': optim.param_groups[0]['lr'],
        })

        if monitor_val < best_val - 1e-3:
            best_val = monitor_val
            torch.save({
                'feature_net': feature_net.state_dict(),
                'mdn': model.state_dict(),
                'act_mean': act_mean_full.detach().cpu(),
                'act_std':  act_std_full.detach().cpu(),
            }, os.path.join(save_ckpt_dir, 'mdn_best.pt'))
            print(f"[SAVE] best checkpoint @ val={best_val:.4f}")
    # ===== End of Train Loop =====

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
    testing()  # 如需测试奖励计算，取消注释此行