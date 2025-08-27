import torch
import torch.nn as nn
from typing import Optional, Tuple

Hidden = Tuple[torch.Tensor, torch.Tensor]  # (h, c)


class LSTMBase(nn.Module):
    """
    通用 LSTM 基类（time-major 接口）。
    - 输入 x_seq: [T, B, in_dim]
    - 可选 masks: [T, B]，1 表示该 env 在该步 done（会把 h/c 清零）
    - 维护/管理 (h, c) 隐藏状态
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        layernorm: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,  # time-major: [T, B, *]
        )
        self.ln = nn.LayerNorm(hidden_size) if layernorm else None

    # --------- hidden state helpers ---------
    def initial_state(self, batch_size: int, device=None) -> Hidden:
        h = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        return (h, c)

    @staticmethod
    def mask_state(hc: Hidden, done_b: torch.Tensor) -> Hidden:
        """
        根据 done 掩码清零隐藏状态。
        done_b: [B] 或 [B,]，1/True 表示 episode 结束。
        """
        h, c = hc
        m = (1.0 - done_b.float()).view(1, -1, 1)  # -> [1, B, 1]
        return (h * m, c * m)

    @staticmethod
    def detach_state(hc: Hidden) -> Hidden:
        h, c = hc
        return (h.detach(), c.detach())

    # --------- forward ---------
    def forward(
        self,
        x_seq: torch.Tensor,           # [T, B, in_dim] 或 [B, in_dim]（单步）
        hc: Optional[Hidden] = None,   # (h, c) 或 None
        masks: Optional[torch.Tensor] = None,  # [T, B] 或 None
        return_sequence: bool = True,
    ) -> Tuple[torch.Tensor, Hidden]:
        """
        返回:
          y_seq: [T, B, H]（或单步时 [B, H]）
          hc_out: (h, c)
        """
        single_step = False
        if x_seq.dim() == 2:  # [B, in_dim] -> [1, B, in_dim]
            x_seq = x_seq.unsqueeze(0)
            single_step = True

        T, B, _ = x_seq.shape
        if hc is None:
            hc = self.initial_state(B, device=x_seq.device)
        h, c = hc

        outputs = []
        for t in range(T):
            if masks is not None:
                h, c = self.mask_state((h, c), masks[t])  # masks[t]: [B]
            y_t, (h, c) = self.lstm(x_seq[t].unsqueeze(0), (h, c))  # y_t: [1, B, H]
            y_t = y_t.squeeze(0)  # [B, H]
            if self.ln is not None:
                y_t = self.ln(y_t)
            outputs.append(y_t)

        y_seq = torch.stack(outputs, dim=0)  # [T, B, H]
        if single_step and not return_sequence:
            return y_seq[-1], (h, c)
        return (y_seq if return_sequence else y_seq[-1]), (h, c)


class LSTMCritic(LSTMBase):
    """
    继承自 LSTMBase 的 Critic：
    feature -> pre-MLP -> LSTM -> V(s)
    """
    def __init__(
        self,
        feature_dim: int,          # 来自 feature_net 的特征维度
        pre_mlp_dim: int = 256,
        lstm_hidden: int = 256,
        num_layers: int = 1,
        layernorm_lstm: bool = False,
    ):
        super().__init__(
            input_size=pre_mlp_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            layernorm=layernorm_lstm,
        )
        self.pre_mlp = nn.Sequential(
            nn.Linear(feature_dim, pre_mlp_dim),
            nn.ReLU(inplace=True),
        )
        self.v_head = nn.Linear(lstm_hidden, 1)

        # 初始化（可选，稳定一些）
        for m in self.pre_mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.v_head.bias)

    # 序列前向：feat_seq [T,B,F] / 单步 [B,F]
    def forward(
        self,
        feat_seq: torch.Tensor,
        hc: Optional[Hidden] = None,
        masks: Optional[torch.Tensor] = None,
        return_sequence: bool = True,
    ) -> Tuple[torch.Tensor, Hidden]:
        # 适配单步/多步
        if feat_seq.dim() == 2:
            x = self.pre_mlp(feat_seq)  # [B, pre_mlp_dim]
            y, hc = super().forward(x, hc=hc, masks=None, return_sequence=False)
            v = self.v_head(y)  # [B, 1]
            return v, hc
        else:
            T, B, F = feat_seq.shape
            x = self.pre_mlp(feat_seq.view(T * B, F)).view(T, B, -1)  # [T,B,pre_mlp_dim]
            y_seq, hc = super().forward(x, hc=hc, masks=masks, return_sequence=True)  # [T,B,H]
            v_seq = self.v_head(y_seq)  # [T,B,1]
            if return_sequence:
                return v_seq, hc
            return v_seq[-1], hc  # 只取最后一步

    # 便捷单步接口：给 PPO rollout 用
    def value_step(
        self,
        feat_t: torch.Tensor,  # [B, F]
        hc: Optional[Hidden],
        done_b: Optional[torch.Tensor] = None,  # [B]
    ) -> Tuple[torch.Tensor, Hidden]:
        if hc is None:
            hc = self.initial_state(batch_size=feat_t.shape[0], device=feat_t.device)
        if done_b is not None:
            hc = self.mask_state(hc, done_b)  # reset hidden for finished envs
        v_t, hc = self.forward(feat_t, hc=hc, masks=None, return_sequence=False)  # [B,1]
        return v_t.squeeze(-1), hc  # -> [B]

    @torch.no_grad()
    def clone_as_target(self) -> "LSTMCritic":
        """
        创建一个冻结的 target critic（用于你之前的 target value 策略）。
        """
        import copy
        tgt = copy.deepcopy(self).eval()
        for p in tgt.parameters():
            p.requires_grad = False
        return tgt
