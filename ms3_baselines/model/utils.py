import torch
import torch.nn.functional as F
import h5py

import numpy as np
import torch.nn as nn

from typing import List, Tuple, Dict, Optional
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split, Subset

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
                 traj_whitelist: Optional[List[str]] = None,
                 camera_mask: Optional[List[str]] = None):
        super().__init__()
        assert mode in ("single", "pair")
        self.h5_path = h5_path
        self.mode = mode
        self.img_size = img_size
        self._h5 = None  # lazy open
        self.camera_mask = camera_mask

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
        traj = self._h5[traj_key]

        has_base = "obs/sensor_data/base_camera/rgb" in traj
        base = traj["obs/sensor_data/base_camera/rgb"][t] if has_base else None  # (H,W,C) or None

        has_hand = "obs/sensor_data/hand_camera/rgb" in traj
        hand = traj["obs/sensor_data/hand_camera/rgb"][t] if has_hand else None  # (H,W,C) or None

        a = traj["actions"][t]  # (A,)

        qpos = traj["obs/agent/qpos"][t]
        # ensure qpos is a torch tensor (HDF5 returns numpy arrays)
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.from_numpy(qpos).float()

        # normalize camera_mask to a set of strings for exclusion
        if self.camera_mask is None:
            mask_set = set()
        elif isinstance(self.camera_mask, str):
            mask_set = {self.camera_mask}
        else:
            mask_set = set(self.camera_mask)

        imgs = []
        if has_base and "base" not in mask_set:
            base = torch.from_numpy(base).permute(2, 0, 1).float() / 255.0  # (C,H,W)
            imgs.append(base)
        if has_hand and "hand" not in mask_set:
            hand = torch.from_numpy(hand).permute(2, 0, 1).float() / 255.0  # (C,H,W)
            imgs.append(hand)

        if len(imgs) == 0:
            raise RuntimeError(f"{traj_key} at t={t} has no camera frames after applying camera_mask={self.camera_mask}")

        img = torch.cat(imgs, dim=0) if len(imgs) > 1 else imgs[0]  # (C or 2C, H, W)

        if self.img_size is not None:
            img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False).squeeze(0)

        act = torch.from_numpy(a).float()  # (A,)

        return img, act, qpos


    def __getitem__(self, idx: int):
        self._ensure_open()
        traj_key, t = self.indices[idx]
        if self.mode == "single":
            img, act, qpos = self._load_one(traj_key, t)
            return img, act, qpos
        else:
            # pair: (t, t+1) 保证是同一条轨迹
            img_t,  act_t, qpos_t = self._load_one(traj_key, t)
            img_tp1,act_tp1, qpos_tp1 = self._load_one(traj_key, t+1)
            return img_t, act_t, img_tp1, act_tp1, qpos_t, qpos_tp1

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
                traj_whitelist: Optional[List[str]] = None,
                camera_mask: Optional[List[str]] = None):
    persistent = (num_workers > 0)
    prefetch = 4 if num_workers > 0 else None
    ds = H5TrajectoryDataset(h5_path=h5_path, mode=mode, img_size=img_size, traj_whitelist=traj_whitelist, camera_mask=camera_mask)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=prefetch
    )
    return ds, dl

def compute_action_stats(dataset, batch_size=2048, num_workers=4, pin_memory=True,
                         use_cuda=None, stats_path: str | None = None):
    """
    仅计算并返回动作的均值和标准差，不做任何文件读写或路径相关操作。
    保持原签名以兼容调用方（stats_path 参数会被忽略）。
    返回 (mean, std) 均为 CPU 上的 torch.FloatTensor。
    """
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory,
                        persistent_workers=(num_workers > 0))

    n = 0
    s1 = None
    s2 = None
    with torch.no_grad():
        for _, a, _ in loader:
            if use_cuda:
                a = a.to(device, non_blocking=True)
            a = a.float()
            if s1 is None:
                s1 = a.sum(dim=0)
                s2 = (a * a).sum(dim=0)
            else:
                s1 = s1 + a.sum(dim=0)
                s2 = s2 + (a * a).sum(dim=0)
            n += a.shape[0]

    if s1 is None:
        raise RuntimeError("No data found in dataset to compute action stats")

    mean_dev = s1 / n
    var_dev = s2 / n - mean_dev ** 2
    mean = mean_dev.cpu().float()
    std = torch.sqrt(var_dev.clamp_min(1e-12)).cpu().float()

    return mean, std


def compute_qpos_stats(dataset, batch_size=2048, num_workers=4, pin_memory=True,
                       use_cuda=None, stats_path: str | None = None):
    """
    计算 dataset 中 qpos 的均值和标准差。Dataset 应返回 (img, act, qpos) 的三元组。
    返回 (mean, std) 均为 CPU 上的 torch.FloatTensor。
    """
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory,
                        persistent_workers=(num_workers > 0))

    n = 0
    s1 = None
    s2 = None
    with torch.no_grad():
        for _, _, q in loader:
            if use_cuda:
                q = q.to(device, non_blocking=True)
            q = q.float()
            if s1 is None:
                s1 = q.sum(dim=0)
                s2 = (q * q).sum(dim=0)
            else:
                s1 = s1 + q.sum(dim=0)
                s2 = s2 + (q * q).sum(dim=0)
            n += q.shape[0]

    if s1 is None:
        raise RuntimeError("No data found in dataset to compute qpos stats")

    mean_dev = s1 / n
    var_dev = s2 / n - mean_dev ** 2
    mean = mean_dev.cpu().float()
    std = torch.sqrt(var_dev.clamp_min(1e-12)).cpu().float()

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