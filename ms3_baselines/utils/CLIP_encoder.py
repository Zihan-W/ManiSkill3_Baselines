import os
import pdb
import clip
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as TF

class CLIPValueHead(nn.Module):
    def __init__(self, instruction, sample_obs, cam_mode=1, use_state=True, hidden=512, dist=False, n_quant=51):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        for p in self.model.parameters(): p.requires_grad = False  # 冻结

        self.txt = instruction
        self.txt_tok = clip.tokenize([self.txt]).to(self.device)
        with torch.no_grad():
            self.fx = self.model.encode_text(self.txt_tok).float()  # (1,D)

        self.use_state = True
        state_embed_dim = 0
        if self.use_state:
            state_embed_dim = 32
            state_dim = sample_obs['state'].shape[-1] if 'state' in sample_obs else 0
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, 64), nn.ReLU(),
                nn.Linear(64, state_embed_dim), nn.ReLU()
            )
        else:
            state_embed_dim = state_dim

        self.cam_mode = cam_mode
        '''
        1: RGB + Double V
        2: RGB + Wrist RGB， 两个视角下分别评估value，再加权融合
        3: RGB + Wrist RGB + Double V
        4: RGB + Wrist RGB + Self-Attention
        '''
        if self.cam_mode == 1:
            in_dim = 3*self.model.visual.output_dim + state_embed_dim + 1  # img_f, txt_f, similarity
        elif self.cam_mode == 2:
            in_dim = (1 * self.model.visual.output_dim) + state_embed_dim + 1
        elif self.cam_mode == 3:
            in_dim = 2 * self.model.visual.output_dim + state_embed_dim + 1 + 1
        elif self.cam_mode == 4:
            in_dim = self.model.visual.output_dim + state_embed_dim + 1 + 1
            self.attention = nn.MultiheadAttention(embed_dim=self.model.visual.output_dim, num_heads=4, batch_first=True)
        if dist:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden//2),
                nn.ReLU(),
                nn.Linear(hidden//2, n_quant)
            )
            self.net2 = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden//2),
                nn.ReLU(),
                nn.Linear(hidden//2, n_quant)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden//2),
                nn.ReLU(),
                nn.Linear(hidden//2, 1)
            )
            self.net2 = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden//2),
                nn.ReLU(),
                nn.Linear(hidden//2, 1)
            )
        self.dist = dist


    def forward(self, z):
        return self.net(z)

    def value(self, obs_batch):
        if self.cam_mode == 1:
            img = self._to_clip(obs_batch)                        # [N,3,224,224]

            with torch.no_grad():
                fI = self.model.encode_image(img).float()         # [N,D]

            # 1) 归一化后的 CLIP logits（不再调用 self.model(img, ...) 重复编码）
            fIn = fI / (fI.norm(dim=-1, keepdim=True) + 1e-12)    # [N,D]
            fxn = self.fx / (self.fx.norm(dim=-1, keepdim=True) + 1e-12)  # [1,D]
            sim = self.model.logit_scale.exp() * (fIn @ fxn.t())  # [N,1]

            # 2) 按 3D+state+1 构建 z，匹配你当前 Linear(in_features=1537)
            fx = self.fx.expand(fI.size(0), -1)                   # [N,D]
            state = obs_batch['state'].to(self.device).float()  # [N, S]
            state_emb = self.state_mlp(state)                   # [N, 128]
            z = torch.cat([fI, fx, fI * fx, sim, state_emb], dim=-1)         # [N, 3D+state+1 = 1537]

            v1 = self.net(z).squeeze(-1)                           # [N]
            v2 = self.net2(z).squeeze(-1)                         # [N]
            # v = torch.min(v1, v2)
            v = 0.5 * v1 + 0.5 * v2
            return v
        elif self.cam_mode == 2:
            # img1是场景相机，img2是腕部相机
            img1, img2 = self._2imgs_to_clip(obs_batch)
            with torch.no_grad():
                fI1 = self.model.encode_image(img1).float()
                fI2 = self.model.encode_image(img2).float()  # [N,D]
            # 1) 归一化后的 CLIP logits（不再调用 self.model(img, ...) 重复编码）
            fIn1 = fI1 / (fI1.norm(dim=-1, keepdim=True) + 1e-12)  # [N,D]
            fIn2 = fI2 / (fI2.norm(dim=-1, keepdim=True) + 1e-12)  # [N,D]
            fxn = self.fx / (self.fx.norm(dim=-1, keepdim=True) + 1e-12)  # [1,D]
            sim1 = self.model.logit_scale.exp() * (fIn1 @ fxn.t())  # [N,1]
            sim2 = self.model.logit_scale.exp() * (fIn2 @ fxn.t())  # [N,1]
            # 2) 按 3D+state+1 构建 z，匹配你当前 Linear(in_features=1537)
            fx = self.fx.expand(fI1.size(0), -1)                   # [N,D]
            state = obs_batch['state'].to(self.device).float()  # [N, S]
            state_emb = self.state_mlp(state)                   # [N, 128]
            z1 = torch.cat([fI1, sim1, state_emb], dim=-1)         # [N, 3D+state+1 = 1537]
            z2 = torch.cat([fI2, sim2, state_emb], dim=-1)
            v1 = self.net(z1).squeeze(-1)                           # [N]
            v2 = self.net2(z2).squeeze(-1)                         # [N]
            alpha = 0.3  # 权重系数，调整两个视角的影响力
            v = alpha * v1 + (1 - alpha) * v2
            return v
        elif self.cam_mode == 3:
            # img1是场景相机，img2是腕部相机
            img1, img2 = self._2imgs_to_clip(obs_batch)
            with torch.no_grad():
                fI1 = self.model.encode_image(img1).float()
                fI2 = self.model.encode_image(img2).float()  # [N,D]
            # 1) 归一化后的 CLIP logits（不再调用 self.model(img, ...) 重复编码）
            fIn1 = fI1 / (fI1.norm(dim=-1, keepdim=True) + 1e-12)  # [N,D]
            fIn2 = fI2 / (fI2.norm(dim=-1, keepdim=True) + 1e-12)  # [N,D]
            fxn = self.fx / (self.fx.norm(dim=-1, keepdim=True) + 1e-12)  # [1,D]
            sim1 = self.model.logit_scale.exp() * (fIn1 @ fxn.t())  # [N,1]
            sim2 = self.model.logit_scale.exp() * (fIn2 @ fxn.t())  # [N,1]
            # 2) 按 3D+state+1 构建 z，匹配你当前 Linear(in_features=1537)
            fx = self.fx.expand(fI1.size(0), -1)                   # [N,D]
            state = obs_batch['state'].to(self.device).float()  # [N, S]
            state_emb = self.state_mlp(state)                   # [N, 128]
            z = torch.cat([fI1, fI2, sim1, sim2, state_emb], dim=-1)         # [N, 3D+state+1 = 1537]
            v1 = self.net(z).squeeze(-1)                           # [N]
            v2 = self.net2(z).squeeze(-1)                         # [N]
            alpha = 0.5
            v = alpha * v1 + (1 - alpha) * v2
            return v
        elif self.cam_mode == 4:
            # img1是场景相机，img2是腕部相机
            img1, img2 = self._2imgs_to_clip(obs_batch)
            with torch.no_grad():
                fI1 = self.model.encode_image(img1).float()
                fI2 = self.model.encode_image(img2).float()  # [N,D]

            # 对 CLIP logits 进行归一化处理
            fIn1 = fI1 / (fI1.norm(dim=-1, keepdim=True) + 1e-12)  # [N, D]
            fIn2 = fI2 / (fI2.norm(dim=-1, keepdim=True) + 1e-12)  # [N, D]
            fxn = self.fx / (self.fx.norm(dim=-1, keepdim=True) + 1e-12)  # [1, D]
            sim1 = self.model.logit_scale.exp() * (fIn1 @ fxn.t())  # [N, 1]
            sim2 = self.model.logit_scale.exp() * (fIn2 @ fxn.t())  # [N, 1]
            # 通过 self-attention 融合两个相机的特征
            img_features = torch.stack([fI1, fI2], dim=1)  # [N, 2, D]，堆叠两个相机的特征
            attn_output, attn_weights = self.attention(img_features, img_features, img_features)  # [N, 2, D], [N, 2, 2]
            # 选择加权的特征进行融合
            fused_features = torch.sum(attn_output, dim=1)  # [N, D]，通过 attention 加权融合两个相机的特征
            state = obs_batch['state'].to(self.device).float()  # [N, S]
            state_emb = self.state_mlp(state)  # [N, 128]
            z = torch.cat([fused_features, sim1, sim2, state_emb], dim=-1)  # [N, D + 128 + D] = [N, 1537]
            v1 = self.net(z).squeeze(-1)  # [N]
            v2 = self.net2(z).squeeze(-1)  # [N]
            import ipdb;ipdb.set_trace()
            alpha = 0.5
            v = alpha * v1 + (1 - alpha) * v2
            return v
        elif self.cam_mode == 5:
            img1, img2 = self._to_clip(obs_batch)
            with torch.no_grad():
                fI1 = self.model.encode_image(img1).float()
                fI2 = self.model.encode_image(img2).float()  # [N,D]
                fx = self.fx.expand(fI1.size(0), -1)
                logits_img1, logits_img2 = self.model(image_input, text_inputs)
                img1_probs = logits_img1.cpu().numpy()
                img2_probs = logits_img2.cpu().numpy()

    def _to_clip(self, obs_dict: torch.Tensor) -> torch.Tensor:
        imgs = []
        # obs: [B,H,W,C]，取前三通道为主视图RGB
        obs_nhwc = obs_dict['rgb']
        x = obs_nhwc.to(self.device).float()
        if x.max() > 1.5: x = x / 255.0
        x = x[..., :3].permute(0,3,1,2)                  # -> [B,3,H,W]
        for i in range(x.size(0)):
            xi = x[i].detach().cpu()                    # [3,H,W] 或 [H,W,3]
            if xi.dim() == 4: xi = xi[0]
            if xi.size(0) != 3: xi = xi.permute(2,0,1)  # 保证 [3,H,W]
            if xi.max() <= 1.0: xi = (xi * 255).byte()
            pil = TF.to_pil_image(xi)                   # Tensor -> PIL
            imgs.append(self.preprocess(pil))           # -> [3,224,224] tensor
        img = torch.stack(imgs, dim=0).to(self.device)  # [B,3,224,224]
        return img

    def _2imgs_to_clip(self, obs_dict: torch.Tensor) -> torch.Tensor:
        _img1 = []
        _img2 = []
        # obs: [B,H,W,C]，取前三通道为主视图RGB
        obs_nhwc = obs_dict['rgb']
        x = obs_nhwc.to(self.device).float()
        if x.max() > 1.5: x = x / 255.0
        x1 = x[..., :3].permute(0,3,1,2)                  # -> [B,3,H,W]
        x2 = x[..., 3:6].permute(0,3,1,2)                  # -> [B,3,H,W]
        for i in range(x1.size(0)):
            xi = x1[i].detach().cpu()                    # [3,H,W] 或 [H,W,3]
            if xi.dim() == 4: xi = xi[0]
            if xi.size(0) != 3: xi = xi.permute(2,0,1)  # 保证 [3,H,W]
            if xi.max() <= 1.0: xi = (xi * 255).byte()
            pil1 = TF.to_pil_image(xi)                   # Tensor -> PIL
            _img1.append(self.preprocess(pil1))           # -> [3,224,224] tensor
        for i in range(x2.size(0)):
            xi = x2[i].detach().cpu()                    # [3,H,W] 或 [H,W,3]
            if xi.dim() == 4: xi = xi[0]
            if xi.size(0) != 3: xi = xi.permute(2,0,1)  # 保证 [3,H,W]
            if xi.max() <= 1.0: xi = (xi * 255).byte()
            pil = TF.to_pil_image(xi)                   # Tensor -> PIL
            _img2.append(self.preprocess(pil))           # -> [3,224,224] tensor
        img1 = torch.stack(_img1, dim=0).to(self.device)  # [B,3,224,224]
        img2 = torch.stack(_img2, dim=0).to(self.device)  # [B,3,224,224]
        return img1, img2


class CLIPActionHead(CLIPValueHead):
    def __init__(self, instruction, action_dim, sample_obs, cam_mode=1, use_state=True, hidden=512, dist=False, n_quant=51):
        super().__init__(instruction, sample_obs, cam_mode, use_state, hidden, dist, n_quant)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        for p in self.model.parameters(): p.requires_grad = False  # 冻结

        self.txt = instruction
        self.txt_tok = clip.tokenize([self.txt]).to(self.device)
        with torch.no_grad():
            self.fx = self.model.encode_text(self.txt_tok).float()  # (1,D)

        self.use_state = True
        state_embed_dim = 0
        if self.use_state:
            state_embed_dim = 32
            state_dim = sample_obs['state'].shape[-1] if 'state' in sample_obs else 0
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, 64), nn.ReLU(),
                nn.Linear(64, state_embed_dim), nn.ReLU()
            )
        else:
            state_embed_dim = state_dim

        self.cam_mode = cam_mode
        self.action_dim = action_dim

        if self.cam_mode == 1:
            in_dim = 2*self.model.visual.output_dim + state_embed_dim + 2  # img_f, txt_f, similarity
        if dist:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden//2),
                nn.ReLU(),
                nn.Linear(hidden//2, n_quant)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden//2),
                nn.ReLU(),
                nn.Linear(hidden//2, action_dim)
            )
        self.dist = dist


    def forward(self, obs_batch):                     # 直接当作 actor_mean(obs)
        return self.get_action(obs_batch)

    def get_action(self, obs_batch):
        # img1是场景相机，img2是腕部相机
        img1, img2 = self._2imgs_to_clip(obs_batch)
        with torch.no_grad():
            fI1 = self.model.encode_image(img1).float()
            fI2 = self.model.encode_image(img2).float()  # [N,D]
        # 1) 归一化后的 CLIP logits（不再调用 self.model(img, ...) 重复编码）
        fIn1 = fI1 / (fI1.norm(dim=-1, keepdim=True) + 1e-12)  # [N,D]
        fIn2 = fI2 / (fI2.norm(dim=-1, keepdim=True) + 1e-12)  # [N,D]
        fxn = self.fx / (self.fx.norm(dim=-1, keepdim=True) + 1e-12)  # [1,D]
        sim1 = self.model.logit_scale.exp() * (fIn1 @ fxn.t())  # [N,1]
        sim2 = self.model.logit_scale.exp() * (fIn2 @ fxn.t())  # [N,1]
        # 2) 按 3D+state+1 构建 z，匹配你当前 Linear(in_features=1537)
        fx = self.fx.expand(fI1.size(0), -1)                   # [N,D]
        state = obs_batch['state'].to(self.device).float()  # [N, S]
        state_emb = self.state_mlp(state)                   # [N, 128]
        z = torch.cat([fI1, fI2, sim1, sim2, state_emb], dim=-1)         # [N, 5D+state+1 = 1537]
        mean = self.net(z)                           # [N]
        return mean


if __name__ == "__main__":
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Prepare the inputs
    image_path = "fig_test/stage3.png"
    image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    print("image_input shape:", image_input.shape)
    text_inputs = clip.tokenize(["Align the peg with the side-facing hole and insert smoothly until fully seated; then release the peg without collision"]).to(device)
    # Calculate features
    with torch.no_grad():
        img_f = model.encode_image(image_input).float()   # (1,D)
        txt_f = model.encode_text(text_inputs).float()     # (1,D)

        # 保险起见再做一次 L2 归一化（有些实现已归一化，但重复一次不影响）
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

        # 余弦相似度 ∈ [-1, 1]
        cos_sim = (img_f @ txt_f.T).item()
    print("cosine similarity:", cos_sim)