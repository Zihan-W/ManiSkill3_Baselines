# probe_sa_distribution.py
from sklearn.preprocessing import StandardScaler

import os, math, numpy as np, torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# ====== 引入你已有的组件 ======
# 请保证与当前文件在同一包层级
from VAE import SmallCNN, VAE, make_loader, H5TrajectoryDataset, compute_action_stats

# ====== 配置 ======
H5_PATH    = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/StackCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"               # TODO: 换成你的h5
CKPT_PATH  = "/home/wzh-2004/ManiSkill3_Baselines/ms3_baselines/VAE/ckpt_VAE/20251028-182808/vae_best.pt"  # TODO: 换成你的权重
OUT_DIR    = "probe_out"
MAX_SAMPLES = 20000      # 采样点数上限（可调）
BATCH_SIZE  = 256
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)

def build_feature_net(ds, ckpt_path, device):
    # 取一个样本以确定通道数
    img, act = ds[0]
    in_ch = img.shape[0]
    feature_net = SmallCNN(in_ch=in_ch, feat_dim=256).to(device).eval()
    act_mean = None; act_std = None

    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        if "feature_net" in sd:
            feature_net.load_state_dict(sd["feature_net"])
        if "act_mean" in sd and "act_std" in sd:
            act_mean = sd["act_mean"].to(device).float()
            act_std  = sd["act_std"].to(device).float()
            print("[INFO] loaded act_mean/std from ckpt.")
    else:
        print("[WARN] ckpt not found. Using randomly initialized feature_net and recomputed action stats.")

    # 若未从权重得到动作统计，则现算（用完整 ds 更稳）
    if act_mean is None or act_std is None:
        act_mean, act_std = compute_action_stats(ds)
        act_mean = act_mean.to(device)
        act_std  = act_std.to(device)
    # 防止除零
    act_std = torch.clamp(act_std, min=1e-6)
    return feature_net, act_mean, act_std

def preprocess_for_gmm(X, seed=SEED):
    # 1) 双精度 + 标准化
    X = X.astype(np.float64, copy=False)
    X = StandardScaler().fit_transform(X)
    # 2) PCA 白化降维（保留到 64 维或样本数-1 的较小者）
    d_eff = min(64, X.shape[1], max(2, X.shape[0]-1))
    Z = PCA(n_components=d_eff, whiten=True, random_state=seed).fit_transform(X)
    return Z

def gmm_bic_scan(X, k_min=1, k_max=10):
    X = X.astype(np.float64, copy=False)
    bics, gmms = [], []
    for k in range(k_min, k_max+1):
        # 先试 full
        try:
            gmm = GaussianMixture(
                n_components=k, covariance_type="full",
                n_init=5, init_params="kmeans",
                reg_covar=1e-3, max_iter=500, random_state=SEED
            )
            gmm.fit(X)
        except ValueError:
            # 回退到 diag + 更强正则
            gmm = GaussianMixture(
                n_components=k, covariance_type="diag",
                n_init=5, init_params="kmeans",
                reg_covar=1e-2, max_iter=500, random_state=SEED
            )
            gmm.fit(X)
        bics.append(gmm.bic(X))
        gmms.append(gmm)
    return np.array(bics), gmms


@torch.no_grad()
def collect_phi_and_actions(ds, feature_net, act_mean, act_std, max_samples, batch_size, device):
    N = len(ds)
    # 均匀下采样索引，避免某条轨迹过拟合
    idxs = np.linspace(0, N-1, num=min(N, max_samples), dtype=int)
    loader = DataLoader(Subset(ds, idxs), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    phi_list, a_list = [], []
    for imgs, acts in loader:
        imgs = imgs.to(device, non_blocking=True).float()
        acts = acts.to(device, non_blocking=True).float()
        phi  = feature_net(imgs)               # (B, feat_dim)
        a_n  = (acts - act_mean) / act_std     # 标准化动作
        phi_list.append(phi.cpu().numpy())
        a_list.append(a_n.cpu().numpy())
    Phi = np.concatenate(phi_list, axis=0)         # (M, D_phi)
    A   = np.concatenate(a_list, axis=0)           # (M, D_a)
    X   = np.concatenate([Phi, A], axis=1)         # (M, D_phi + D_a)
    return Phi, A, X

def plot_pca_scatter(X, title, path):
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(X)
    plt.figure(figsize=(6,5))
    plt.scatter(Z[:,0], Z[:,1], s=4, alpha=0.5)
    evr = pca.explained_variance_ratio_.sum()
    plt.title(f"{title}\nPCA-2D (explained var = {evr:.2f})")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return Z, evr

def try_umap(X, title, path):
    try:
        import umap
    except Exception:
        print("[INFO] umap-learn not installed. Skip UMAP.")
        return None
    reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.1)
    U = reducer.fit_transform(X)
    plt.figure(figsize=(6,5))
    plt.scatter(U[:,0], U[:,1], s=4, alpha=0.5)
    plt.title(f"{title}\nUMAP-2D")
    plt.xlabel("U1"); plt.ylabel("U2"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return U

def plot_bic(bics, path):
    ks = np.arange(1, len(bics)+1)
    plt.figure(figsize=(6,4))
    plt.plot(ks, bics, marker="o")
    plt.xticks(ks)
    plt.xlabel("Number of components (K)")
    plt.ylabel("BIC (lower is better)")
    plt.title("GMM BIC curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def plot_cluster_on_2d(Z, labels, title, path):
    plt.figure(figsize=(6,5))
    for lab in np.unique(labels):
        m = labels == lab
        plt.scatter(Z[m,0], Z[m,1], s=6, alpha=0.6, label=f"comp {lab}")
    plt.legend(markerscale=2, fontsize=8)
    plt.title(title)
    plt.xlabel("Dim1"); plt.ylabel("Dim2"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def main():
    print(f"[INFO] Loading dataset: {H5_PATH}")
    ds, _ = make_loader(H5_PATH, mode="single", img_size=None, num_workers=8, shuffle=False)
    feature_net, act_mean, act_std = build_feature_net(ds, CKPT_PATH, DEVICE)
    print("[INFO] Extracting features & actions ...")
    Phi, A, X = collect_phi_and_actions(ds, feature_net, act_mean, act_std, MAX_SAMPLES, BATCH_SIZE, DEVICE)

    # ===== 可视化：PCA / UMAP =====
    Zpca, evr = plot_pca_scatter(X, "[phi(s), a_norm] PCA", os.path.join(OUT_DIR, "pca_scatter.png"))
    Umap = try_umap(X, "[phi(s), a_norm] UMAP", os.path.join(OUT_DIR, "umap_scatter.png"))

    # ===== 预处理给 GMM 用（标准化 + PCA白化降维）=====
    Zgmm = preprocess_for_gmm(X)

    # ===== GMM-BIC 扫描（判定多峰）=====
    print("[INFO] Fitting GMM for K=1..10 ...")
    bics, gmms = gmm_bic_scan(X, 1, 10)
    plot_bic(bics, os.path.join(OUT_DIR, "gmm_bic.png"))
    best_k = int(np.argmin(bics) + 1)
    best_gmm = gmms[best_k-1]
    labels = best_gmm.predict(X)

    # 聚类质量（可选）：用 PCA-2D 上的轮廓分数做个粗评估
    try:
        sil = silhouette_score(Zpca, labels)
    except Exception:
        sil = float("nan")

    # 聚类着色图（PCA/UMAP）
    plot_cluster_on_2d(Zpca, labels, f"GMM K={best_k} on PCA space (sil={sil:.3f})",
                       os.path.join(OUT_DIR, "clusters_on_pca.png"))
    if Umap is not None:
        plot_cluster_on_2d(Umap, labels, f"GMM K={best_k} on UMAP space",
                           os.path.join(OUT_DIR, "clusters_on_umap.png"))

    # ===== 输出结论摘要 =====
    bic_drop = bics[0] - bics.min()
    verdict = []
    if best_k == 1 and bic_drop < 10:
        verdict.append("BIC 无显著改善（K=1最佳或提升<10），整体更像“单峰近高斯”。")
    else:
        verdict.append(f"BIC 显著下降（ΔBIC≈{bic_drop:.1f}，最佳 K={best_k}），呈现“多峰”迹象。")

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write("== Probe (s,a) distribution ==\n")
        f.write(f"PCA explained variance (2D): {evr:.3f}\n")
        f.write(f"Best K by BIC: {best_k}\n")
        f.write(f"ΔBIC (K=1 vs best): {bic_drop:.1f}\n")
        f.write(f"Silhouette on PCA: {sil:.3f}\n")
        f.write("Verdict: " + verdict[0] + "\n")

    print("[DONE] Results saved to:", OUT_DIR)
    print(" - pca_scatter.png / umap_scatter.png")
    print(" - gmm_bic.png / clusters_on_*.png")
    print(" - summary.txt")

if __name__ == "__main__":
    main()
