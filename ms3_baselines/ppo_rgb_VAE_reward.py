# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# 从/home/wzh-2004/RewardModelTest/VAE/VAE.py加载
import sys
sys.path.append("/home/wzh-2004/RewardModelTest/VAE")
from VAE import VAE, VAEManifoldScorer

@dataclass
class Args:

    vae_ckpt: Optional[str] = "/home/wzh-2004/RewardModelTest/VAE/ckpt_20251015_110910"
    vae_reward_coef: float = 1   # λ, 与环境奖励加权
    vae_sigma: float = 0.5         # 高斯核宽度，越小越“尖锐”
    vae_k: int = 8                 # 采样K个z近似最短流形距离
    vae_img_size: int = 96         # 训练VAE时的输入尺寸, 用于resize
    vae_action_scale: float = 1.0  # 如VAE训练时把动作缩放到[-1,1]，这里也保持一致


    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill_Baselines"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    # wandb_group: str = "VAE_reward"
    wandb_group: str = "DEBUG"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "all"
    """the environment rendering mode"""

    # Algorithm specific arguments
    env_id: str = "StackCube-v1"
    """the id of the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    total_timesteps: int = 20000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 265
    """the number of parallel environments"""
    num_eval_envs: int = 80
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 16
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 100
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels=sample_obs["rgb"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])

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

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

class Agent(nn.Module):
    def __init__(self, envs, sample_obs):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()
        latent_size = self.feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5)
    def get_features(self, x):
        return self.feature_net(x)
    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # env_kwargs = dict(obs_mode="rgb", render_mode=args.render_mode, sim_backend="physx_cuda")
    train_env_kwargs = dict(obs_mode="rgb", render_mode="none", sim_backend="physx_cuda")
    eval_env_kwargs  = dict(obs_mode="rgb", render_mode=args.render_mode, sim_backend="physx_cuda")

    if args.control_mode is not None:
        # env_kwargs["control_mode"] = args.control_mode
        train_env_kwargs["control_mode"] = args.control_mode
        eval_env_kwargs["control_mode"] = args.control_mode

    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **eval_env_kwargs)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **train_env_kwargs)

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None

    #####
    @torch.no_grad()
    def _vae_debug_once(vae, vae_scorer, obs_rgb, actions, device, sigma_boost_if_tiny=True,
                        target_r_mean=0.2, max_tries=4):
        """
        计算一次 d_rec / r 的统计信息；如 r_mean 极小，逐步把 sigma *= 3 做在线放大。
        返回 (d_rec, r_cur, sigma_now) 供记录。
        """
        s_img = vae_scorer._preprocess_obs(obs_rgb.to(device))
        a_std = (actions.to(device) - vae_scorer.act_mu) / vae_scorer.act_std

        mu, logvar, s_feat = vae.encode(s_img, a_std)
        recon = vae.decode(s_feat, mu)
        d_rec = ((recon - a_std) ** 2).mean(dim=1)  # (B,)

        def r_from_sigma(sig):
            sig = float(sig)
            sig = max(sig, 1e-12)
            return torch.exp(- d_rec / (sig ** 2))

        r_cur = r_from_sigma(vae_scorer.sigma)
        print(f"[VAE DEBUG] sigma={float(vae_scorer.sigma):.6g} | "
            f"d_rec mean/med/max={d_rec.mean().item():.4e}/"
            f"{d_rec.median().item():.4e}/{d_rec.max().item():.4e} | "
            f"r mean/med/min={r_cur.mean().item():.4f}/"
            f"{r_cur.median().item():.4f}/{r_cur.min().item():.4f}")

        tries = 0
        if sigma_boost_if_tiny:
            while r_cur.mean().item() < target_r_mean and tries < max_tries:
                vae_scorer.sigma *= 3.0
                r_cur = r_from_sigma(vae_scorer.sigma)
                tries += 1
                print(f"[VAE AUTO-SIGMA] trial#{tries}: sigma->{float(vae_scorer.sigma):.6g}, "
                    f"r_mean={r_cur.mean().item():.4f}")
        return d_rec, r_cur, float(vae_scorer.sigma)

    if not args.evaluate:
        print("Running training")
        #if args.track:
        if True:
            import wandb
            config = vars(args)
            # normalized_dense
            # sparse
            config["env_cfg"] = dict(**train_env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**eval_env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            print("TRACKING HERE")
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)

    next_obs = {k: v.to(device, non_blocking=True) for k, v in next_obs.items()}  # 立刻迁到GPU

    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    eps_returns = torch.zeros(args.num_envs, dtype=torch.float, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    agent = Agent(envs, sample_obs=next_obs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # === 加载已训练的 VAE，用于奖励 ===
    vae_scorer = None
    vae_ckpt_path = f"{args.vae_ckpt}/vae_best.pt"
    if args.vae_ckpt is not None and os.path.isfile(vae_ckpt_path):
        # 推断通道数与动作维
        sample_rgb = next_obs["rgb"]  # (N,H,W,C)
        in_ch = sample_rgb.shape[-1]  # 若你训练时是两路相机拼接到C维，这里就等于 6；单路就是 3
        action_dim = int(np.prod(envs.single_action_space.shape))

        vae = VAE(in_ch=in_ch, action_dim=action_dim, latent_dim=32, max_action=1.0, feat_dim=256).to(device)
        vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device))
        vae_scorer = VAEManifoldScorer(
            vae=vae, device=device, k=args.vae_k,
            img_size=args.vae_img_size, ckpt_dir=args.vae_ckpt,
        )
        print(f"[VAE] loaded from {args.vae_ckpt}, in_ch={in_ch}, action_dim={action_dim}, sigma={vae_scorer.sigma}")
        # === DEBUG BLOCK 1: VAE 元数据 ===
        try:
            print(f"[VAE] act_mu shape={tuple(vae_scorer.act_mu.shape)}, "
                f"act_std shape={tuple(vae_scorer.act_std.shape)}, "
                f"std min/mean/max={vae_scorer.act_std.min().item():.3e}/"
                f"{vae_scorer.act_std.mean().item():.3e}/{vae_scorer.act_std.max().item():.3e}")
            print(f"[VAE] sigma(initial) = {float(vae_scorer.sigma):.6g}, k = {vae_scorer.k}, img_size = {vae_scorer.img_size}")
        except Exception as e:
            print("[VAE DEBUG] failed to print VAE meta:", repr(e))
    else:
        print("[VAE] no vae_ckpt provided or file not found; skip VAE reward.")

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    cumulative_times = defaultdict(float)

    for iteration in range(1, args.num_iterations + 1):
        # === DEBUG BLOCK 4: 观测图像尺寸/通道 ===
        if global_step == args.num_envs:
            rgb = obs[step]["rgb"]
            print(f"[VAE DEBUG] obs['rgb'] shape={tuple(rgb.shape)} "
                f"(expect (B,H,W,3) and will be resized to {vae_scorer.img_size}x{vae_scorer.img_size})")

        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")

                # 计算成功率
                if k == "success_once":
                    print(f"eval_metrics[k]={eval_metrics[k]}")
                    total_eval_episodes = 0
                    total_success = 0
                    for _ in eval_metrics[k]:
                        total_eval_episodes += len(_.int())
                        total_success += _.int().sum().item()   # 或 _.long().sum().item()
                    success_rate = total_success / total_eval_episodes
                    if logger is not None:
                        logger.add_scalar(f"eval/success_rate", success_rate, global_step)
                    print(f"eval_success_rate={success_rate} over {total_eval_episodes} episodes")
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break

        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        rollout_time = time.perf_counter()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_obs = {k: v.to(device, non_blocking=True) for k, v in next_obs.items()}

            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            # rewards[step] = reward.view(-1) * args.reward_scale

            # === VAE 流形奖励：用 obs[step] 和 actions[step]（即 (s_t, a_t)）===
            # === DEBUG BLOCK 2: 前几步做一次分布/自适应 ===
            if vae_scorer is not None and global_step < args.num_envs * 4:
                d_dbg, r_dbg, sigma_now = _vae_debug_once(
                    vae=vae, vae_scorer=vae_scorer,
                    obs_rgb=obs[step]["rgb"], actions=actions[step],
                    device=device, sigma_boost_if_tiny=True,  # 若 r 很小会自动放大 sigma
                    target_r_mean=0.2, max_tries=4
                )
                if logger is not None:
                    logger.add_scalar("debug/d_rec_mean", d_dbg.mean().item(), global_step)
                    logger.add_scalar("debug/d_rec_median", d_dbg.median().item(), global_step)
                    logger.add_scalar("debug/sigma_now", sigma_now, global_step)
                    logger.add_scalar("debug/r_dbg_mean", r_dbg.mean().item(), global_step)

            # === DEBUG BLOCK 3: 动作分布与标准化基准的对齐性 ===
            if vae_scorer is not None and (global_step % (args.num_envs * 32) == 0):
                a_cur = actions[step].detach()
                a_mean = a_cur.mean(dim=0)
                a_std  = a_cur.std(dim=0).clamp_min(1e-12)
                ref_mu  = vae_scorer.act_mu.squeeze(0).detach().cpu()
                ref_std = vae_scorer.act_std.squeeze(0).detach().cpu()
                # 量一个粗略的 mismatch（越小越好；>10 表明维度/尺度可能错）
                mismatch = torch.mean(torch.abs((a_mean.cpu() - ref_mu) / (ref_std + 1e-12))).item()
                print(f"[VAE DEBUG] action mean/std vs ref std | mismatch={mismatch:.3f} "
                    f"(>10 often means control_mode/scale mismatch)")
                if logger is not None:
                    logger.add_scalar("debug/action_mean_std_mismatch", mismatch, global_step)

            # === VAE 流形奖励（VAE 训练在“绝对 q”，而 PPO 输出“Δq”时的转换）===
            if vae_scorer is not None:
                with torch.no_grad():
                    # 0) 取出 PPO 的动作（Δq），以及 env 的动作界
                    a_delta = actions[step]  # (B, A) 这里是 Δq
                    act_low  = torch.as_tensor(envs.single_action_space.low,  device=device, dtype=a_delta.dtype)
                    act_high = torch.as_tensor(envs.single_action_space.high, device=device, dtype=a_delta.dtype)
                    a_delta_clip = torch.max(torch.min(a_delta, act_high), act_low)

                    # 1) 从 obs['state'] 取当前 qpos（假定前 action_dim 就是 qpos；如不是，请调整切片）
                    action_dim = a_delta.shape[-1]
                    assert "state" in obs[step], "需要 obs['state'] 来做 Δq -> 绝对 q 的转换"
                    qpos_cur = obs[step]["state"][..., :action_dim]  # (B, A)

                    # 2) 组装“绝对 q”作为 VAE 的 action
                    a_abs = qpos_cur + a_delta_clip

                    # 3) 可选：按你 VAE 数据集的绝对关节界做裁剪（如果你有绝对 q 的物理上下界）
                    # 若没有绝对界，通常不强制裁剪也行，因为标准化会把它带回训练分布；保留注释作为提示。
                    # joint_low_abs, joint_high_abs = <加载/写死你的绝对关节上下界张量，shape=(A,)>
                    # a_abs = torch.max(torch.min(a_abs, joint_high_abs), joint_low_abs)

                    # 4) 奖励：用“绝对 q”的动作打分（与训练时一致）
                    r_mani_abs = vae_scorer.reward(obs[step], a_abs)

                    # 5) 诊断：看看如果误用 Δq 去打分，r 有多差（只记录不参与训练）
                    r_mani_delta = vae_scorer.reward(obs[step], a_delta_clip)

                    # 6) 选择用于训练的奖励（这里用“绝对 q”的 r”）
                    r_mani = r_mani_abs

                    # 7) 调试日志（前几步打印，持续写入 TB/W&B）
                    #   统计 Δq 超界比例（提示策略是否经常超出 env 的 delta 界）
                    over_low  = (a_delta < act_low).float()
                    over_high = (a_delta > act_high).float()
                    frac_oob  = (over_low + over_high).mean().item()
                    max_viol  = torch.max(torch.max((act_low - a_delta).clamp_min(0.0).abs()),
                                        torch.max((a_delta - act_high).clamp_min(0.0).abs())).item()

                    if logger is not None:
                        logger.add_scalar("debug/frac_action_oob_delta", frac_oob, global_step)
                        logger.add_scalar("debug/r_mani_abs_mean",   r_mani_abs.mean().item(),   global_step)
                        logger.add_scalar("debug/r_mani_delta_mean", r_mani_delta.mean().item(), global_step)

                    if global_step < args.num_envs * 8:
                        print(f"[DELTA->ABS DEBUG] oob_frac(delta)={frac_oob:.3f}, max_violation(delta)={max_viol:.3f} | "
                            f"r_abs_mean={r_mani_abs.mean().item():.4f}, r_delta_mean={r_mani_delta.mean().item():.4f}")
            else:
                r_mani = 0.0

            if logger is not None:
                logger.add_scalar("debug/reward", reward.mean().item(), global_step)
                logger.add_scalar("debug/r_mani", args.vae_reward_coef * r_mani.mean().item(), global_step)
            # 合并奖励
            reward = torch.zeros_like(reward)
            rewards[step] = (reward.view(-1) + args.vae_reward_coef * r_mani) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"]).view(-1)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("charts/newvalue", newvalue.mean().item(), global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)

        logger.add_scalar("debug/vae_reward", r_mani.mean().item(), global_step)

    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if logger is not None: logger.close()