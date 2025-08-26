import torch
import os

def save_actor_ckpt(actor, run_name, tag):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
        },
        f"runs/{run_name}/checkpoints/{tag}.pt",
    )

def save_critic_ckpt(critic, run_name, tag):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    torch.save(
        {
            "critic": critic.state_dict(),
        },
        f"runs/{run_name}/checkpoints/{tag}.pt",
    )

def load_actor_from_ckpt(self, ckpt_path, load_logstd=True, freeze_feature=False):
    """只把 feature_net / actor_mean (/ actor_logstd) 从 BC ckpt 加载进来。"""
    dev = next(self.parameters()).device
    state = torch.load(ckpt_path, map_location=dev)

    # 兼容 {'actor': state_dict} 或 直接 state_dict 两种保存方式
    sd = state.get("actor", state)
    # 兼容 DDP 的 'module.' 前缀
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # 只挑 actor 相关键
    filtered = {}
    for k, v in sd.items():
        if k.startswith("feature_net.") or k.startswith("actor_mean."):
            filtered[k] = v
        elif load_logstd and (k == "actor_logstd"):
            filtered[k] = v

    # 实际加载（strict=False 跳过 critic 等不匹配的键）
    missing, unexpected = self.load_state_dict(filtered, strict=False)
    # print("[actor load] missing:", missing)
    # print("[actor load] unexpected:", unexpected)

    if freeze_feature:
        for p in self.feature_net.parameters():
            p.requires_grad = False
        print("[actor load] feature_net is frozen.")