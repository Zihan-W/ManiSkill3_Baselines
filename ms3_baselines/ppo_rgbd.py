# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional
import copy

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

from ms3_baselines.utils.feature_extractor import FeatureExtractor, layer_init, freeze_model
from ms3_baselines.utils.pretrain import load_actor_from_ckpt

from ms3_baselines.utils.CLIP_encoder import CLIPActionHead

@dataclass
class Args:
    robot_uids: str = "panda_wristcam"
    use_depth = False
    use_pretrain = False
    actor_ckpt_path = ""
    vision_type = "RGBCNN"  # RGBCNN, RGBDCNN, ResNet50, r3m50(still buggy)， hil_serl_resnet
    # 如果选择了hil_serl_resnet，则可进一步选择pooling_method
    hil_serl_resnet_pooling_method = "spatial_learned_embeddings"   # spatial_learned_embeddings, spatial_softmax, avg, max, none

    target_value_tau: float = 0.01         # critic target value网络的软更新系数
    target_value_interval: int = 1         # 每多少个 PPO 迭代做一次软更新critic target value网络

    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill_Baselines"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
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
    # env_id: str = "PickCube-v1"
    env_id: str = "StackCube-v1"
    """the id of the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    actor_learning_rate: float = 2e-4
    critic_learning_rate: float = 3e-4
    feature_net_learning_rate: float = 2e-4
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 100
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 100
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.8
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.05
    """coefficient of the entropy"""
    vf_coef: float = 0.7
    """coefficient of the value function"""
    max_grad_norm: float = 100.0
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.3
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
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

class Agent(nn.Module):
    def __init__(self, envs, sample_obs, args):
        super().__init__()
        # feature for critic
        self.actor_feature_net = None
        self.critic_feature_net = FeatureExtractor(sample_obs=sample_obs, args=args)
        # self.actor_feature_net = FeatureExtractor(sample_obs=sample_obs, args=args)
        latent_size = self.critic_feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.critic_target = copy.deepcopy(self.critic).eval()
        freeze_model(self.critic_target)

        instruction = "a red cube and a green cube"
        action_dim = int(np.prod(envs.unwrapped.single_action_space.shape))
        self.actor_mean = CLIPActionHead(instruction, action_dim, sample_obs, use_state=True)
        # self.actor_mean = nn.Sequential(
        #     layer_init(nn.Linear(latent_size, 512)),
        #     nn.ReLU(inplace=True),
        #     layer_init(nn.Linear(512, 512)),
        #     nn.ReLU(inplace=True),
        #     layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01*np.sqrt(2)),
        # )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

        if args.vision_type in ["r3m50", "ResNet50", "hil_serl_resnet"]:
            for p in self.critic_feature_net.extractors["rgb"].backbone.parameters():
                p.requires_grad = False  # R3M/ResNet骨干

        if self.actor_feature_net is None:
            for p in self.actor_mean.model.parameters():
                p.requires_grad = False  # CLIP整模型，再次冻结，以防万一
        else:
            if args.vision_type in ["r3m50", "ResNet50", "hil_serl_resnet"]:
                for p in self.actor_feature_net.extractors["rgb"].backbone.parameters():
                    p.requires_grad = False  # R3M/ResNet骨干

    def get_features(self, x):
        return self.critic_feature_net(x)
    def get_value(self, x):
        x = self.critic_feature_net(x)
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        if self.actor_feature_net is not None:
            x = self.actor_feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        if self.actor_feature_net is not None:
            actor_feature = self.actor_feature_net(x)
        else:
            actor_feature = x
        action_mean = self.actor_mean(actor_feature)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        x = self.critic_feature_net(x)
        value = self.critic(x)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

    # target value 的前向
    @torch.no_grad()
    def get_target_value(self, x):
        xf = self.critic_feature_net(x)               # 不回传梯度
        v = self.critic_target(xf)
        return v
    # 软更新target value网络
    @torch.no_grad()
    def update_target(self, tau=0.01):
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1 - tau).add_(tau * p.data)

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

class utils_class():
    def __init__(self,device):
        self.reward_scaling = None
        self.device = device

    def normalize_advantages(self, advantages):
        """
        对优势进行标准化
        advantages: 计算出的优势值
        return: 标准化后的优势值
        """
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    class RunningMeanStdTorch:
        """Torch 版，支持向量化统计（每个 env 一条），Welford 累计。"""
        def __init__(self, shape, eps=1e-5, device="cpu", dtype=torch.float32):
            self.device = torch.device(device)
            self.dtype = dtype
            self.eps = eps
            self.n = torch.zeros((), device=self.device, dtype=torch.float64)
            self.mean = torch.zeros(shape, device=self.device, dtype=self.dtype)
            self.M2   = torch.zeros(shape, device=self.device, dtype=self.dtype)

        @torch.no_grad()
        def update(self, x):
            # x: [num_envs]
            x = x.to(self.device, self.dtype)
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

        @property
        def var(self):
            denom = torch.clamp(self.n, min=1.0)
            return self.M2 / denom

        @property
        def std(self):
            return torch.sqrt(self.var + self.eps)

    class RewardScaling:
        """
        Per-env 折扣累计 R_t = gamma * R_{t-1} + r_t
        然后按 running mean/std 做归一化（零均值 + 除以 std）。
        """
        def __init__(self, shape, gamma, device="cpu"):
            self.shape = shape          # (num_envs,)
            self.gamma = gamma
            self.device = device
            self.running_ms = utils_class.RunningMeanStdTorch(shape=shape, device=device)
            self.R = torch.zeros(shape, device=device)

        @torch.no_grad()
        def __call__(self, r_t, done_mask=None):
            """
            r_t: [num_envs] torch.Tensor 或能转成 tensor 的数据
            done_mask: [num_envs] 布尔/0-1 张量，episode 结束处为 1
            """
            r_t = torch.as_tensor(r_t, device=self.device, dtype=torch.float32).view(-1)

            # 对已 done 的 env，把累计 R 清零（在进入本步之前重置）
            if done_mask is not None:
                done_mask = done_mask.to(self.device).view(-1).bool()
                self.R[done_mask] = 0.0

            # 折扣累计
            self.R = self.gamma * self.R + r_t

            # 更新统计并做零均值/除方差
            self.running_ms.update(self.R)
            r_scaled = (r_t - self.running_ms.mean) / (self.running_ms.std + 1e-8)
            return r_scaled

        @torch.no_grad()
        def reset(self, mask=None):
            """
            可选：只对 mask==True 的 env 清零 R（例如手工 reset 的情况）。
            如果 mask is None，则全部清零。
            """
            if mask is None:
                self.R.zero_()
            else:
                mask = mask.to(self.device).view(-1).bool()
                self.R[mask] = 0.0

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

    _utils = utils_class(device)
    _utils.reward_scaling = _utils.RewardScaling(shape=(args.num_envs,), gamma=args.gamma, device=device)

    # env setup
    if args.use_depth:
        obs_mode = "rgbd"
    else:
        obs_mode = "rgb"
    env_kwargs = dict(obs_mode=obs_mode, render_mode=args.render_mode, sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=args.use_depth, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=args.use_depth, state=args.include_state)

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
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
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
    values_critic_target = torch.zeros((args.num_steps, args.num_envs), device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    agent = Agent(envs, sample_obs=next_obs, args=args).to(device)
    if args.use_pretrain:
        actor_ckpt_path = None
        agent.load_actor_from_ckpt(actor_ckpt_path, load_logstd=True, freeze_feature=False)

    # TODO： 修改需要训练的部分参数
    # 只把 requires_grad=True 的参数给优化器
    def filt(named):
        return [p for n,p in named if p.requires_grad]
    opt_groups = [
        {"params": filt(agent.critic.named_parameters()), "lr": args.critic_learning_rate, "weight_decay": 0.0},
        {"params": filt(agent.actor_mean.named_parameters()), "lr": args.actor_learning_rate, "weight_decay": 0.0},
        {"params": filt(agent.critic_feature_net.named_parameters()), "lr": args.feature_net_learning_rate, "weight_decay": 1e-4},
        {"params": [agent.actor_logstd], "lr": 3e-4, "weight_decay": 0.0},
    ]
    if agent.actor_feature_net is not None:
        opt_groups.append({"params": filt(agent.actor_feature_net.named_parameters()), "lr": args.feature_net_learning_rate, "weight_decay": 1e-4})

    optimizer = torch.optim.AdamW(opt_groups, betas=(0.9, 0.999), eps=1e-8)

    id2name = {id(p): n for n, p in agent.named_parameters()}
    def show_optimizer(optim):
        print(f"optimizer: {optim}")
        print("=== optimizer param groups ===")
        for gi, g in enumerate(optim.param_groups):
            names = [id2name.get(id(p), "<external>") for p in g["params"]]
            print(f"Group {gi} (lr={g.get('lr', None)}):")
            for n in names:
                print("  ", n)
    show_optimizer(optimizer)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    cumulative_times = defaultdict(float)

    for iteration in range(1, args.num_iterations + 1):
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
                # 计算成功率
                if k == "success_once":
                    total_eval_episodes = 0
                    total_success = 0
                    for _ in eval_metrics[k]:
                        total_eval_episodes += len(_.int())
                        total_success += _.int().sum().item()   # 或 _.long().sum().item()
                    success_rate = total_success / total_eval_episodes
                    if logger is not None:
                        logger.add_scalar(f"eval/success_rate", success_rate, global_step)
                    print(f"eval_success_rate={success_rate} over {total_eval_episodes} episodes")

                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
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
                values_critic_target[step] = agent.get_target_value(next_obs).flatten()   # value for critic target
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale
            # rewards[step] = _utils.reward_scaling(reward.view(-1), done_mask=next_done)           # ★ 正确调用            print(f"Step {step+1}/{args.num_steps} R mean: {reward.mean().item():.3f} std: {reward.std().item():.3f}")

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    # final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"]).view(-1)
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = \
                    agent.get_target_value(infos["final_observation"]).view(-1)   # 用 values for critic target替换原values
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            # next_value = agent.get_value(next_obs).reshape(1, -1)
            next_value = agent.get_target_value(next_obs).reshape(1, -1)  # 用 value for critic target替换原value
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    # nextvalues = values[t + 1]
                    nextvalues = values_critic_target[t + 1]   # 用 values for critic target 替换原values
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

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]   # 注意这里 values[t] 仍是当前 critic
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]   # 注意这里 values[t] 仍是当前 critic
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            # advantages = _utils.normalize_advantages(advantages)
            returns = advantages + values   # 仍然让当前 critic 去拟合 returns

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
        # 软更新 target value 网络
        if (iteration % args.target_value_interval) == 0:
            agent.update_target(tau=args.target_value_tau)
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
        if logger is not None:
            logger.add_scalar("critic/value_mean_rollout", values.mean().item(), global_step)
            logger.add_scalar("critic/value_std_rollout",  values.std().item(),  global_step)
            logger.add_scalar("critic/critic_target_value_mean_rollout", values_critic_target.mean().item(), global_step)
            logger.add_scalar("critic/critic_target_value_std_rollout",  values_critic_target.std().item(),  global_step)
            logger.add_scalar("critic/value_target_gap_L1", (values - values_critic_target).abs().mean().item(), global_step)
    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if logger is not None: logger.close()
