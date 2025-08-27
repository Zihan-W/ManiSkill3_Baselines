import numpy as np
import torch
import torch.nn as nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)

def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False

class FeatureExtractor(nn.Module):
    def __init__(self, sample_obs, args):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels=sample_obs["rgb"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])
        self.cam_num = in_channels // 3
        self._use_depth = False
        if 'depth' in sample_obs:
            self._use_depth = True
            in_channels += sample_obs["depth"].shape[-1]

        if args.vision_type == "RGBCNN":
            # here we use a NatureCNN architecture to process images, but any architecture is permissble here
            cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
            # to easily figure out the dimensions after flattening, we pass a test tensor
            with torch.no_grad():
                n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            extractors["rgb"] = nn.Sequential(cnn, fc)
            self.out_features += feature_size

        elif args.vision_type == "RGBDCNN":
            if self._use_depth is None:
                raise ValueError("Depth data not found in sample_obs")
            cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
            # to easily figure out the dimensions after flattening, we pass a test tensor
            with torch.no_grad():
                sample_input = torch.cat(
                    [sample_obs["rgb"], sample_obs["depth"]], dim=-1
                )
                n_flatten = cnn(sample_input.float().permute(0,3,1,2).cpu()).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            extractors["rgb"] = nn.Sequential(cnn, fc)
            self.out_features += feature_size

        elif args.vision_type in ["ResNet50", "hil_serl_resnet"]:
            if args.vision_type == "ResNet50":
                import torchvision.models as models
                backbone = models.resnet50(pretrained=True)
                backbone = nn.Sequential(*list(backbone.children())[:-1])
                freeze_model(backbone)

            elif args.vision_type == "hil_serl_resnet":
                from ms3_baselines.utils.resnet_v1 import resnetv1_configs, PreTrainedResNetEncoder
                image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])
                pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](image_size=image_size)
                freeze_model(pretrained_encoder)
                backbone = PreTrainedResNetEncoder(
                    pooling_method=args.hil_serl_resnet_pooling_method,
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                )

            # 推理并获取输出特征的维度
            env_obs = sample_obs["rgb"][..., :3]
            with torch.no_grad():
                sample_input = env_obs.float().permute(0,3,1,2).cpu()  # 添加batch维度
                output = backbone(sample_input)
                n_flatten = output.view(output.size(0), -1).size(1)

            if self.cam_num == 1:
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors["rgb"] = nn.Sequential(backbone, nn.Flatten(), fc)
                self.out_features += feature_size
            else:
                # 分别对env camera和wrist camera进行处理
                class TwoCamResNet(nn.Module):
                    def __init__(self, backbone, n_flatten, feature_size):
                        super().__init__()
                        self.backbone = backbone
                        self.fc = nn.Sequential(nn.Linear(n_flatten * 2, feature_size), nn.ReLU())
                    def forward(self, x):
                        # x shape: (B, C*2, H, W)
                        env_cam = x[..., :3, :, :]
                        wrist_cam = x[..., -3:, :, :]
                        env_feat = self.backbone(env_cam.float())
                        wrist_feat = self.backbone(wrist_cam.float())
                        combined = torch.cat([env_feat.view(env_feat.size(0), -1), wrist_feat.view(wrist_feat.size(0), -1)], dim=1)
                        return self.fc(combined)
                extractors["rgb"] = TwoCamResNet(backbone, n_flatten, feature_size)
                self.out_features += feature_size

        elif args.vision_type == "r3m50":
            class R3MExtractor(nn.Module):
                '''
                输入: x [B, C, H, W]，其中 C = 3*cam_num (+ cam_num 如果含 depth)
                行为:
                - 对每个相机的 3 通道 RGB 分别跑一个冻结的 R3M(ResNet50) 特征器 -> 2048
                - 若含 depth: 把 cam_num 路 1 通道 depth 过一个轻量 CNN -> 64
                - 将所有 RGB 特征 concat（2048*cam_num），再与 depth 特征 concat，线性投影到 feature_size
                '''
                def __init__(self, cam_num: int, use_depth: bool, feature_size: int):
                    super().__init__()
                    self.cam_num = cam_num
                    self.use_depth = use_depth

                    # 1) backbone：优先用 r3m，否则 fallback 到 torchvision resnet50
                    self.use_r3m = False
                    out_dim = 2048
                    try:
                        import r3m
                        self.backbone = r3m.load_r3m("resnet50").eval()
                        self.use_r3m = True
                    except Exception:
                        from torchvision.models import resnet50, ResNet50_Weights
                        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                        m.fc = nn.Identity()
                        self.backbone = m.eval()

                    freeze_model(self.backbone)

                    # 2) depth 编码器（可选）：把 [B, cam_num, H, W] -> 64
                    if self.use_depth:
                        self.depth_enc = nn.Sequential(
                            nn.Conv2d(self.cam_num, 32, 3, 2, 1), nn.ReLU(inplace=True),
                            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(),
                            nn.Linear(64, 64), nn.ReLU(inplace=True),
                        )
                        depth_dim = 64
                    else:
                        self.depth_enc = None
                        depth_dim = 0

                    # 3) 最终投影到 feature_size
                    self.fc = nn.Sequential(
                        nn.Linear(self.cam_num * out_dim + depth_dim, feature_size),
                        nn.ReLU(inplace=True),
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    """
                    x: [B, C, H, W], C = 3*cam_num (+ cam_num if depth)
                    返回: [B, feature_size]
                    """
                    B, C, H, W = x.shape
                    # 前 3*cam_num 是 RGB，各 3 通道一个相机
                    rgb_feats = []
                    for i in range(self.cam_num):
                        xi = x[:, i*3:(i+1)*3, :, :]            # [B,3,H,W]
                        with torch.no_grad():
                            fi = self.backbone(xi)              # [B,2048] (R3M 或 ResNet50)
                        rgb_feats.append(fi)
                    f_rgb = torch.cat(rgb_feats, dim=1)         # [B, 2048*cam_num]

                    # depth 放在 RGB 后面：逐相机 1 通道 -> [B, cam_num, H, W]
                    if self.use_depth and (C >= 3*self.cam_num + self.cam_num):
                        d = x[:, 3*self.cam_num: 3*self.cam_num + self.cam_num, :, :]  # [B,cam_num,H,W]
                        f_d = self.depth_enc(d)                                        # [B,64]
                        f = torch.cat([f_rgb, f_d], dim=1)
                    else:
                        f = f_rgb

                    return self.fc(f)                                                # [B, feature_size]

            # 将提取器挂到你的 dict 里，保持统一接口
            extractors["rgb"] = R3MExtractor(self.cam_num, self._use_depth, feature_size)
            self.out_features += feature_size

        elif args.vision_type == "ResNet50_2":
            # TODO: still buggy
            import torch.nn.functional as F
            from torchvision.models import resnet50, ResNet50_Weights

            class ResNetExtractor(nn.Module):
                """
                输入: x [B, C, H, W], C = 3*cam_num (+ cam_num 若含 depth)
                - 每个相机 3 通道 RGB 分别过冻结 ResNet50(conv 到 avgpool) -> 2048
                - 若含 depth: [B, cam_num, H, W] 过轻量 CNN -> 64
                - concat 后线性映射到 feature_size
                输出: [B, feature_size]
                """
                def __init__(self, cam_num: int, use_depth: bool, feature_size: int):
                    super().__init__()
                    self.cam_num = cam_num
                    self.use_depth = use_depth

                    # 预训练 resnet50 去掉 fc
                    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                    m.fc = nn.Identity()
                    self.backbone = nn.Sequential(
                        m.conv1, m.bn1, m.relu, m.maxpool,
                        m.layer1, m.layer2, m.layer3, m.layer4,
                        m.avgpool,   # -> [B,2048,1,1]
                    ).eval()
                    freeze_model(self.backbone)
                    self.out_dim = 2048

                    # ImageNet 归一化
                    self.register_buffer("im_mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1), persistent=False)
                    self.register_buffer("im_std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1), persistent=False)

                    # depth 编码器（可选）：把 [B, cam_num, H, W] -> 64
                    if self.use_depth:
                        self.depth_enc = nn.Sequential(
                            nn.Conv2d(self.cam_num, 32, 3, 2, 1), nn.ReLU(inplace=True),
                            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                            nn.Linear(64, 64), nn.ReLU(inplace=True),
                        )
                        depth_dim = 64
                    else:
                        self.depth_enc = None
                        depth_dim = 0

                    self.fc = nn.Sequential(
                        nn.Linear(self.cam_num * self.out_dim + depth_dim, feature_size),
                        nn.ReLU(inplace=True),
                    )

                def _encode_rgb(self, x3: torch.Tensor) -> torch.Tensor:
                    # x3: [B,3,H,W] in [0,1] or [0,255]
                    if x3.max() > 1.5:
                        x3 = x3 / 255.0
                    # ResNet 任何尺寸都行，但 224 更稳
                    x3 = F.interpolate(x3, size=(224,224), mode="bilinear", align_corners=False)
                    x3 = (x3 - self.im_mean) / self.im_std
                    with torch.no_grad():
                        f = self.backbone(x3)                 # [B,2048,1,1]
                    return f.flatten(1)                       # [B,2048]

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # x: [B, C, H, W], C = 3*cam_num (+ cam_num if depth)
                    B, C, H, W = x.shape

                    # 只取前 3*cam_num 作为 RGB
                    rgb_all = x[:, :3*self.cam_num, :, :]     # [B,3*cam_num,H,W]
                    rgb_feats = []
                    for i in range(self.cam_num):
                        xi = rgb_all[:, 3*i:3*(i+1), :, :]    # [B,3,H,W]
                        fi = self._encode_rgb(xi)             # [B,2048]
                        rgb_feats.append(fi)
                    f_rgb = torch.cat(rgb_feats, dim=1)       # [B, 2048*cam_num]

                    # depth 在 RGB 之后：形状 [B, cam_num, H, W]
                    if self.use_depth and (C >= 3*self.cam_num + self.cam_num):
                        d = x[:, 3*self.cam_num:3*self.cam_num + self.cam_num, :, :]
                        f_d = self.depth_enc(d)               # [B,64]
                        f = torch.cat([f_rgb, f_d], dim=1)
                    else:
                        f = f_rgb

                    return self.fc(f)                         # [B, feature_size]

            extractors["rgb"] = ResNetExtractor(self.cam_num, self._use_depth, feature_size)
            self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            state_fc = nn.Sequential(
                layer_init(nn.Linear(state_size, 128)),
                nn.LayerNorm(128),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(128, 64)),
                )
            extractors["state"] = state_fc
            self.out_features += 64

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                if self._use_depth is not None and self._use_depth:
                    rgb = observations["rgb"].float() / 255.0
                    depth = observations["depth"]
                    obs = torch.cat(
                        [rgb, depth], axis=-1
                    )
                    obs = obs.float().permute(0,3,1,2)
                else:
                    obs = obs.float().permute(0,3,1,2)
                    obs = obs / 255.0

            # 不单独处理深度图像
            if key == "depth":
                continue

            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)