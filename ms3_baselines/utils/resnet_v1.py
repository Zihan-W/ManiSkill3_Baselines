# resnet_encoder_torch.py
import math
from typing import Callable, List, Optional, Sequence, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Utils
# --------------------------
def _ensure_2tuple(x):
    if isinstance(x, (list, tuple)):
        assert len(x) == 2
        return x
    return (x, x)


def resize_if_needed(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """x: [B,C,H,W]; target_hw=(H,W)"""
    h, w = x.shape[-2], x.shape[-1]
    th, tw = _ensure_2tuple(target_hw)
    if (h, w) != (th, tw):
        x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
    return x


# --------------------------
# Building blocks
# --------------------------
class AddSpatialCoordinates(nn.Module):
    """将 [-1,1] 归一化的 (x,y) 坐标以通道形式拼到输入上。
    输入/输出: [B,C,H,W] -> [B,C+2,H,W]
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        device = x.device
        ys = torch.linspace(-1.0, 1.0, steps=h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, xs, ys], dim=1)


class SpatialSoftmax(nn.Module):
    """对每个通道做空间 softmax，输出各通道的期望坐标 (x,y)；输出维度为 [B, 2*C]。"""
    def __init__(self, height: int, width: int, channel: int, temperature: float = 1.0):
        super().__init__()
        self.h, self.w, self.c = height, width, channel
        self.learnable_temp = temperature == -1
        if self.learnable_temp:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("temperature", torch.tensor(float(temperature)), persistent=False)

        # 预先展开的坐标网格（展平成 H*W）
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, steps=height),
            torch.linspace(-1.0, 1.0, steps=width),
            indexing="ij"
        )
        self.register_buffer("pos_x", pos_x.reshape(1, 1, height * width), persistent=False)
        self.register_buffer("pos_y", pos_y.reshape(1, 1, height * width), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> [B,C,HW]
        b, c, h, w = x.shape
        assert (h, w) == (self.h, self.w), "Input spatial size must match SpatialSoftmax settings."
        x = x.reshape(b, c, h * w)
        attn = F.softmax(x / (self.temperature + 1e-6), dim=-1)  # [B,C,HW]
        ex = (attn * self.pos_x).sum(dim=-1)  # [B,C]
        ey = (attn * self.pos_y).sum(dim=-1)  # [B,C]
        out = torch.cat([ex, ey], dim=1)      # [B,2C]
        return out


class SpatialLearnedEmbeddings(nn.Module):
    """学习一组空间核，对每个通道做加权求和。输出 [B, C*num_features]。"""
    def __init__(self, height: int, width: int, channel: int, num_features: int = 5):
        super().__init__()
        self.h, self.w, self.c = height, width, channel
        # 参数形状 [F, C, H, W]，方便与 [B, C, H, W] 广播相乘
        self.kernel = nn.Parameter(torch.empty(num_features, channel, height, width))
        nn.init.kaiming_normal_(self.kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        b, c, h, w = x.shape
        assert (c, h, w) == (self.c, self.h, self.w)
        # [B,1,C,H,W] * [F,C,H,W] -> [B,F,C,H,W] -> sum(HW) -> [B,F,C] -> reshape [B, F*C]
        y = (x.unsqueeze(1) * self.kernel.unsqueeze(0)).sum(dim=(-1, -2))
        return y.reshape(b, -1)


def _norm_layer(norm: str, num_channels: int) -> nn.Module:
    if norm == "group":
        # 与 JAX 的 MyGroupNorm 类似，简单用 4 组
        num_groups = math.gcd(4, num_channels) or 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-5)
    elif norm == "layer":
        # 对 [B,C,H,W] 使用 LayerNorm，需要指定归一化维度 C,H,W
        return nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=1e-5)
    elif norm == "batch":
        return nn.BatchNorm2d(num_channels, eps=1e-5)
    else:
        raise ValueError(f"Unknown norm: {norm}")


def _act_fn(name: str) -> Callable:
    # Fix capitalization for activation function names (e.g., 'ReLU', 'GELU')
    # Use upper/lower logic to match PyTorch naming
    if hasattr(nn, name):
        return getattr(nn, name)()
    elif hasattr(nn, name.upper()):
        return getattr(nn, name.upper())()
    elif hasattr(nn, name.capitalize()):
        return getattr(nn, name.capitalize())()
    else:
        raise AttributeError(f"torch.nn has no activation '{name}'")


class BasicBlock(nn.Module):
    """对应 JAX 里的 ResNetBlock（非瓶颈）"""
    def __init__(self, in_ch: int, out_ch: int, stride: int, norm: str, act: str):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = _norm_layer(norm, out_ch)
        self.act   = _act_fn(act)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = _norm_layer(norm, out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
                _norm_layer(norm, out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.act(y + identity)


class Bottleneck(nn.Module):
    """对应 JAX 里的 BottleneckResNetBlock。输出通道= filters*4。"""
    expansion = 4
    def __init__(self, in_ch: int, mid_ch: int, stride: int, norm: str, act: str):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, 1, 0, bias=False)
        self.bn1   = _norm_layer(norm, mid_ch)
        self.act   = _act_fn(act)

        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn2   = _norm_layer(norm, mid_ch)

        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, 1, 0, bias=False)
        self.bn3   = _norm_layer(norm, out_ch)
        nn.init.zeros_(self.bn3.weight) if hasattr(self.bn3, "weight") else None

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
                _norm_layer(norm, out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = self.act(self.bn1(self.conv1(x)))
        y = self.act(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.act(y + identity)


class FilmConditioning2D(nn.Module):
    """简化版 FiLM：根据 cond_var -> 预测 (gamma, beta)，在通道维上做仿射变换。"""
    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, cond_var: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W], cond_var: [B,D]
        gamma_beta = self.proj(cond_var)  # [B, 2C]
        b, c, h, w = x.shape
        gamma, beta = gamma_beta[:, :c], gamma_beta[:, c:]
        gamma = gamma.view(b, c, 1, 1)
        beta  = beta.view(b, c, 1, 1)
        return x * (1 + gamma) + beta


# --------------------------
# ResNet Encoder
# --------------------------
class ResNetEncoder(nn.Module):
    """
    PyTorch 版的 JAX ResNetEncoder
    关键参数：
      - stage_sizes: 每个 stage 的 block 数
      - block_cls:   BasicBlock 或 Bottleneck
      - num_filters: stem 输出通道
      - norm:        "group" / "layer" / "batch"
      - act:         "relu" / "gelu" / ...
      - add_spatial_coordinates: 是否在输入拼接 (x,y)
      - pooling_method: "avg" | "max" | "spatial_softmax" | "spatial_learned_embeddings" | "none"
      - pre_pooling: 若 True，前向直接返回卷积特征 (不做池化/瓶颈)
      - image_size:  期望输入尺寸 (H,W)，会自动 resize
      - use_film:    是否对每个 block 做 FiLM 调制（需要提供 cond_var 和 cond_dim）
    输入:
      - observations: [B,C,H,W] (0~255)
      - cond_var: [B, D]，当 use_film=True 时必需
    """
    def __init__(
        self,
        stage_sizes: Sequence[int],
        block_cls: type,
        num_filters: int = 64,
        dtype: str = "float32",
        act: str = "ReLU",
        norm: str = "group",
        add_spatial_coordinates: bool = False,
        pooling_method: str = "avg",
        use_spatial_softmax: bool = False,  # 保留兼容
        softmax_temperature: float = 1.0,
        num_spatial_blocks: int = 8,
        use_film: bool = False,
        cond_dim: Optional[int] = None,
        bottleneck_dim: Optional[int] = None,
        pre_pooling: bool = True,
        image_size: Tuple[int, int] = (128, 128),
    ):
        super().__init__()
        self.stage_sizes = list(stage_sizes)
        self.block_cls   = block_cls
        self.num_filters = num_filters
        self.norm        = norm
        self.act         = act
        self.add_spatial_coordinates = add_spatial_coordinates
        self.pooling_method = pooling_method
        self.softmax_temperature = softmax_temperature
        self.num_spatial_blocks = num_spatial_blocks
        self.use_film = use_film
        self.cond_dim = cond_dim
        self.bottleneck_dim = bottleneck_dim
        self.pre_pooling = pre_pooling
        self.image_size = _ensure_2tuple(image_size)

        in_ch = 3 + (2 if add_spatial_coordinates else 0)

        # stem
        self.conv_init = nn.Conv2d(in_ch, num_filters, 7, 2, 3, bias=False)
        self.bn_init   = _norm_layer(norm, num_filters)
        self.act_fn    = _act_fn(act)
        self.maxpool   = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stages
        stages = []
        ch = num_filters
        for i, n_blocks in enumerate(self.stage_sizes):
            blocks = []
            for j in range(n_blocks):
                stride = 2 if (i > 0 and j == 0) else 1
                if block_cls is BasicBlock:
                    blocks.append(BasicBlock(ch, ch if j>0 else ch* (2 if (i>0 and j==0) else 1), stride, norm, act))  # 这一行写法有点绕，下面更清晰
                # 更清晰写法：
                if block_cls is BasicBlock:
                    out_ch = ch if j > 0 else (ch if i == 0 else ch*2)
                    # 但因为我们下面会更新 ch，改成通用形式：
                # 用统一实现
            # 我们重写一遍以避免混乱：
        self.stages = nn.ModuleList()
        ch = num_filters
        for i, n_blocks in enumerate(self.stage_sizes):
            stage = nn.ModuleList()
            for j in range(n_blocks):
                stride = 2 if (i > 0 and j == 0) else 1
                if self.block_cls is BasicBlock:
                    out_ch = ch if i == 0 else (ch * 2 if j == 0 else ch * 2)
                    stage.append(BasicBlock(ch, out_ch, stride, norm, act))
                    ch = out_ch
                else:  # Bottleneck
                    # mid_ch = base, out_ch = mid_ch*4
                    mid = ch if i == 0 else (ch // Bottleneck.expansion * 2 if j == 0 else ch // Bottleneck.expansion)
                    out_ch = mid * Bottleneck.expansion
                    stage.append(Bottleneck(ch, mid, stride, norm, act))
                    ch = out_ch
            self.stages.append(stage)

        # FiLM（可选）：为每个 stage 的每个 block 准备一个 FiLM 模块
        if self.use_film:
            assert self.cond_dim is not None, "use_film=True 时必须提供 cond_dim"
            self.films = nn.ModuleList()
            ch_ptr = num_filters
            for i, stage in enumerate(self.stages):
                for j, blk in enumerate(stage):
                    # 取 block 输出通道
                    if isinstance(blk, BasicBlock):
                        c_out = blk.bn2.num_features
                    else:
                        c_out = blk.bn3.num_features
                    self.films.append(FilmConditioning2D(self.cond_dim, c_out))
                # 更新 ch_ptr 供下一 stage 参考，但这里我们已经从模块上拿到了
        else:
            self.films = None

        # 池化分支
        self.spatial_softmax = None
        self.spatial_learned = None
        # 预定义一个典型的最终空间大小，用于初始化池化模块时的形状。
        # 但输入可能被 resize 到 image_size，因此以 image_size 推断：
        # stem 下采样 x4（conv2d stride=2 + maxpool stride=2），每个后续 stage 的首个 block 可能 stride=2
        # 这里不严格推断，运行时再根据特征图尺寸动态创建模块更稳妥。
        self._pool_created = False  # 延迟创建，等 forward 得到真实 [H,W,C]

        # bottleneck（可选）
        if self.bottleneck_dim is not None:
            self.fc_bottleneck = nn.Sequential(
                nn.Linear(ch, self.bottleneck_dim),
                nn.LayerNorm(self.bottleneck_dim),
                nn.Tanh(),
            )
        else:
            self.fc_bottleneck = None

        # 注册 ImageNet 均值方差（做成 buffer，自动 device 对齐）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std",  std, persistent=False)

        self.add_coords = AddSpatialCoordinates() if add_spatial_coordinates else None

    def _create_pool_modules_if_needed(self, x: torch.Tensor):
        if self._pool_created or self.pooling_method in ("avg", "max", "none"):
            return
        b, c, h, w = x.shape
        if self.pooling_method == "spatial_softmax":
            self.spatial_softmax = SpatialSoftmax(h, w, c, temperature=self.softmax_temperature).to(x.device)
        elif self.pooling_method == "spatial_learned_embeddings":
            self.spatial_learned = SpatialLearnedEmbeddings(h, w, c, num_features=self.num_spatial_blocks).to(x.device)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        self._pool_created = True

    def forward(
        self,
        observations: torch.Tensor,
        train: bool = True,
        cond_var: Optional[torch.Tensor] = None,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        # 输入: [B,C,H,W] (0~255)，先 resize，再做 ImageNet 归一化
        x = observations
        x = resize_if_needed(x, self.image_size)
        x = (x / 255.0 - self.imagenet_mean) / self.imagenet_std

        if self.add_coords is not None:
            x = self.add_coords(x)

        # stem
        x = self.act_fn(self.bn_init(self.conv_init(x)))
        x = self.maxpool(x)

        # stages
        film_idx = 0
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
                if self.use_film:
                    assert cond_var is not None, "use_film=True 但未提供 cond_var"
                    x = self.films[film_idx](x, cond_var)
                    film_idx += 1

        if self.pre_pooling:
            return x.detach() if stop_gradient else x

        # 创建池化模块（需要拿到当前 feature map 的 H,W,C）
        self._create_pool_modules_if_needed(x)

        # 池化
        if self.pooling_method == "avg":
            x = x.mean(dim=(-2, -1))  # [B,C]
        elif self.pooling_method == "max":
            x = x.amax(dim=(-2, -1))  # [B,C]
        elif self.pooling_method == "spatial_softmax":
            x = self.spatial_softmax(x)  # [B,2C]
        elif self.pooling_method == "spatial_learned_embeddings":
            x = self.spatial_learned(x)  # [B,C*F]
        elif self.pooling_method == "none":
            pass
        else:
            raise ValueError(f"Unknown pooling_method: {self.pooling_method}")

        # bottleneck
        if self.fc_bottleneck is not None:
            x = self.fc_bottleneck(x)
        return x


class PreTrainedResNetEncoder(nn.Module):
    """包装已有的 encoder（通常输出未池化特征），在此处做池化与可选 bottleneck。"""
    def __init__(
        self,
        pretrained_encoder: nn.Module,
        pooling_method: str = "avg",
        softmax_temperature: float = 1.0,
        num_spatial_blocks: int = 8,
        bottleneck_dim: Optional[int] = None,
    ):
        super().__init__()
        self.pretrained_encoder = pretrained_encoder
        self.pooling_method = pooling_method
        self.softmax_temperature = softmax_temperature
        self.num_spatial_blocks = num_spatial_blocks
        self.bottleneck_dim = bottleneck_dim

        self._pool_created = False
        self.spatial_softmax = None
        self.spatial_learned = None

        if self.bottleneck_dim is not None:
            # 注意: 这里需要知道 encoder 池化后的通道数，运行时才能确定；简单起见延迟到 forward 再建
            self._bottleneck = None  # lazy

    def _create_pool_modules_if_needed(self, x: torch.Tensor):
        if self._pool_created or self.pooling_method in ("avg", "max", "none"):
            return
        b, c, h, w = x.shape
        if self.pooling_method == "spatial_softmax":
            self.spatial_softmax = SpatialSoftmax(h, w, c, temperature=self.softmax_temperature).to(x.device)
        elif self.pooling_method == "spatial_learned_embeddings":
            self.spatial_learned = SpatialLearnedEmbeddings(h, w, c, num_features=self.num_spatial_blocks).to(x.device)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        self._pool_created = True

    def forward(self, observations: torch.Tensor, encode: bool = True, train: bool = True) -> torch.Tensor:
        x = observations
        if encode:
            x = self.pretrained_encoder(x)  # 期望 [B,C,H,W]

        self._create_pool_modules_if_needed(x)

        if self.pooling_method == "avg":
            x = x.mean(dim=(-2, -1))
        elif self.pooling_method == "max":
            x = x.amax(dim=(-2, -1))
        elif self.pooling_method == "spatial_softmax":
            x = self.spatial_softmax(x)
        elif self.pooling_method == "spatial_learned_embeddings":
            x = self.spatial_learned(x)
        elif self.pooling_method == "none":
            pass
        else:
            raise ValueError(f"Unknown pooling_method: {self.pooling_method}")

        if self.bottleneck_dim is not None:
            if not hasattr(self, "_bottleneck") or self._bottleneck is None:
                in_dim = x.shape[1]
                self._bottleneck = nn.Sequential(
                    nn.Linear(in_dim, self.bottleneck_dim),
                    nn.LayerNorm(self.bottleneck_dim),
                    nn.Tanh(),
                ).to(x.device)
            x = self._bottleneck(x)
        return x


# --------------------------
# Preset configs (factories)
# --------------------------
def _resnetv1_10(**kw):
    return ResNetEncoder(stage_sizes=(1,1,1,1), block_cls=BasicBlock, **kw)

def _resnetv1_10_frozen(**kw):
    return ResNetEncoder(stage_sizes=(1,1,1,1), block_cls=BasicBlock, pre_pooling=True, **kw)

def _resnetv1_18(**kw):
    return ResNetEncoder(stage_sizes=(2,2,2,2), block_cls=BasicBlock, **kw)

def _resnetv1_18_frozen(**kw):
    return ResNetEncoder(stage_sizes=(2,2,2,2), block_cls=BasicBlock, pre_pooling=True, **kw)

def _resnetv1_34(**kw):
    return ResNetEncoder(stage_sizes=(3,4,6,3), block_cls=BasicBlock, **kw)

def _resnetv1_50(**kw):
    return ResNetEncoder(stage_sizes=(3,4,6,3), block_cls=Bottleneck, **kw)

def _resnetv1_18_deeper(**kw):
    return ResNetEncoder(stage_sizes=(3,3,3,3), block_cls=BasicBlock, **kw)

def _resnetv1_18_deepest(**kw):
    return ResNetEncoder(stage_sizes=(4,4,4,4), block_cls=BasicBlock, **kw)

def _resnetv1_18_bridge(**kw):
    return ResNetEncoder(stage_sizes=(2,2,2,2), block_cls=BasicBlock, num_spatial_blocks=8, **kw)

def _resnetv1_34_bridge(**kw):
    return ResNetEncoder(stage_sizes=(3,4,6,3), block_cls=BasicBlock, num_spatial_blocks=8, **kw)

def _resnetv1_34_bridge_film(**kw):
    return ResNetEncoder(stage_sizes=(3,4,6,3), block_cls=BasicBlock, num_spatial_blocks=8, use_film=True, **kw)

def _resnetv1_50_bridge(**kw):
    return ResNetEncoder(stage_sizes=(3,4,6,3), block_cls=Bottleneck, num_spatial_blocks=8, **kw)

def _resnetv1_50_bridge_film(**kw):
    return ResNetEncoder(stage_sizes=(3,4,6,3), block_cls=Bottleneck, num_spatial_blocks=8, use_film=True, **kw)


resnetv1_configs: Dict[str, Callable[..., ResNetEncoder]] = {
    "resnetv1-10": _resnetv1_10,
    "resnetv1-10-frozen": _resnetv1_10_frozen,
    "resnetv1-18": _resnetv1_18,
    "resnetv1-18-frozen": _resnetv1_18_frozen,
    "resnetv1-34": _resnetv1_34,
    "resnetv1-50": _resnetv1_50,
    "resnetv1-18-deeper": _resnetv1_18_deeper,
    "resnetv1-18-deepest": _resnetv1_18_deepest,
    "resnetv1-18-bridge": _resnetv1_18_bridge,
    "resnetv1-34-bridge": _resnetv1_34_bridge,
    "resnetv1-34-bridge-film": _resnetv1_34_bridge_film,
    "resnetv1-50-bridge": _resnetv1_50_bridge,
    "resnetv1-50-bridge-film": _resnetv1_50_bridge_film,
}
