import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from mani_skill.utils import gym_utils
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from behavior_cloning.evaluate import evaluate
from behavior_cloning.make_env import make_eval_envs

from ms3_baselines.utils.feature_extractor import FeatureExtractor, layer_init
from ms3_baselines.utils.pretrain import load_actor_from_ckpt, save_actor_ckpt

@dataclass
class Args:
    robot: str = "panda_wristcam"

    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkillBC"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "StackCube-v1"
    """the id of the environment"""
    demo_path: str = "/home/wzh-2004/RewardModelTest/Maniskill3_Baseline/demos/StackCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = 100
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""

    # Behavior cloning specific arguments
    normalize_states: bool = False
    """if toggled, states are normalized to mean 0 and standard deviation 1"""
    lr: float = 1e-4
    """the learning rate for the actor"""
    normalize_states: bool = False
    """if toggled, states are normalized to mean 0 and standard deviation 1"""

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 1000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = 1000
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 50
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 8
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None

def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


def flatten_state_dict_with_space(state_dict: dict) -> np.ndarray:
    states = []
    for key in state_dict.keys():
        value = state_dict[key]
        if isinstance(value, (tuple, list)):
            state = None if len(value) == 0 else value
        elif isinstance(value, (bool, np.bool_, int, np.int32, np.int64)):
            # x = np.array(1) > 0 is np.bool_ instead of ndarray
            state = int(value)
        elif isinstance(value, (float, np.float32, np.float64)):
            state = np.float32(value)
        elif isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            if value.ndim > 2:
                raise AssertionError(
                    "The dimension of {} should not be more than 2.".format(key)
                )
            state = value
        else:
            raise TypeError("Unsupported type: {}".format(type(value)))
        if state is not None:
            states.append(state)
    if len(states) == 0:
        return np.empty(0)
    else:
        if isinstance(states[0], torch.Tensor):
            try:
                return torch.hstack(states)
            except:
                return torch.column_stack(states)
        else:
            try:
                return np.hstack(states)
            except:  # dirty fix for concat trajectory of states
                return np.column_stack(states)


class ManiSkillDataset(Dataset):
    def __init__(self, dataset_file: str, device: torch.device, load_count) -> None:
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]

        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.camera_data = defaultdict(list)
        self.actions = []
        self.dones = []
        self.states = []
        self.total_frames = 0
        self.device = device

        if load_count is None:
            load_count = len(self.episodes)

        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            agent = trajectory["obs"]["agent"]
            extra = trajectory["obs"]["extra"]

            state = np.hstack(
                [
                    flatten_state_dict_with_space(agent),
                    flatten_state_dict_with_space(extra),
                ]
            )
            self.states.append(state)

            # we use :-1 here to ignore the last observation as that
            # is the terminal observation which has no actions
            for camera_name, camera_data in trajectory["obs"]["sensor_data"].items():
                self.camera_data[camera_name + "_rgb"].append(camera_data["rgb"][:-1])
                # self.camera_data[camera_name + "_depth"].append(camera_data["depth"][:-1])

            self.actions.append(trajectory["actions"])
        for key in self.camera_data.keys():
            if "rgb" in key:
                # self.camera_data[key] = np.vstack(self.camera_data[key]) / 255.0
                self.camera_data[key] = np.vstack(self.camera_data[key])
            else:
                self.camera_data[key] = np.vstack(self.camera_data[key]) / 1024.0

        self.states = np.vstack(self.states)
        self.actions = np.vstack(self.actions)
        for key in self.camera_data.keys():
            assert self.camera_data[key].shape[0] == self.actions.shape[0]

    def __len__(self):
        return len(self.camera_data[list(self.camera_data.keys())[0]])

    def __getitem__(self, idx):
        out = {}
        out["action"] = (
            torch.from_numpy(self.actions[idx]).float().to(device=self.device)
        )
        out["state"] = torch.from_numpy(self.states[idx]).float().to(device=self.device)
        rgbd_data = []
        for key in sorted(self.camera_data.keys()):
            rgbd_data.append(torch.from_numpy(self.camera_data[key][idx]).float().to(device=self.device))
        out["rgb"] = torch.cat(rgbd_data, dim=-1)

        return out

# taken from here
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations

# 兼容 PPO 的Actor
class PPOCompatibleBC(nn.Module):
    def __init__(self, sample_obs, action_dim):
        super().__init__()
        device = sample_obs["rgb"].device
        self.feature_net = FeatureExtractor(sample_obs=sample_obs, vision_type="RGBCNN").to(device)
        latent_size = self.feature_net.out_features

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, action_dim), std=0.01*np.sqrt(2)),
        ).to(device)

        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim, device=device) * -0.5)

    def forward(self, obs):   # obs 必须是 dict，含 'rgb'（使用 'state'）
        feat = self.feature_net(obs)
        return self.actor_mean(feat)     # 直接输出 action_mean


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith(".h5"):
        import json

        json_file = args.demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")
            assert (
                control_mode == args.control_mode
            ), f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"
        print(f"Control mode: {control_mode}, matched.")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(args.torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    control_mode = os.path.split(args.demo_path)[1].split(".")[2]

    # env setup
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="rgb",
        render_mode="all",
        robot_uids=args.robot,
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    if args.track:
        import wandb

        config = vars(args)
        config["eval_env_cfg"] = dict(
            **env_kwargs,
            num_envs=args.num_eval_envs,
            env_id=args.env_id,
            # env_horizon=gym_utils.find_max_episode_steps_value(envs),
        )
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="BehaviorCloning",
            tags=["behavior_cloning"],
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    ds = ManiSkillDataset(
        args.demo_path,
        device=device,
        load_count=args.num_demos,
    )

    obs, _ = envs.reset(seed=args.seed)
    #   odict_keys(['state', 'rgb', 'depth'])
    #   obs['rgb'].shape   (5, 128, 128, 6)

    sampler = RandomSampler(ds)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    camera_count = len(ds.camera_data.keys()) // 2 # each camera has rgb and depth
    iter_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)

    data_loader = DataLoader(ds, batch_sampler=iter_sampler, num_workers=0)
    H, W = 128, 128                       # 跟你环境里用的分辨率一致
    sample_obs = {
        "rgb":   torch.zeros(args.batch_size, H, W, 6, device=device),  # [B,H,W,3*C]
        "state": torch.zeros(args.batch_size, ds.states.shape[1],     device=device),
    }
    actor = PPOCompatibleBC(sample_obs, envs.single_action_space.shape[0]).to(device=device)

    optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    best_eval_metrics = defaultdict(float)

    # 统计数据集动作均值/方差（一次性）
    a_mean = torch.from_numpy(ds.actions).float().mean(dim=0).to(device)  # [action_dim]
    a_std  = torch.from_numpy(ds.actions).float().std(dim=0).clamp_min(1e-6).to(device)

    # 初始化最后一层
    last = actor.actor_mean[-1]                       # Linear(512 -> action_dim)
    with torch.no_grad():
        last.bias.copy_(a_mean)                       # 用数据集均值初始化偏置
        nn.init.orthogonal_(last.weight, 0.01)        # 小权重即可，保持稳定

    for iteration, batch in enumerate(data_loader):
        if iteration == 0:
            print("rgb train min/max:", batch["rgb"].min().item(), batch["rgb"].max().item())  # 期望 0..255

        log_dict = {}

        optimizer.zero_grad()
        preds = actor(batch)    # TODO: check if input as same as actor of PPO
        loss = F.mse_loss(preds, batch["action"])
        loss.backward()
        optimizer.step()


        if iteration % args.log_freq == 0:
            print(f"Iteration {iteration}, loss: {loss.item()}")
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            writer.add_scalar("losses/total_loss", loss.item(), iteration)

        if iteration % args.eval_freq == 0:
            actor.eval()
            def sample_fn(obs):
                if isinstance(obs["rgb"], np.ndarray):
                    for k, v in obs.items():
                        obs[k] = torch.from_numpy(v).float().to(device)
                else:
                    obs["rgb"] = obs["rgb"].float().to(device)
                action = actor(obs)

                if args.sim_backend == "cpu":
                    action = action.cpu().numpy()
                return action

            with torch.no_grad():
                eval_metrics = evaluate(args.num_eval_episodes, sample_fn, envs)
            actor.train()
            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_actor_ckpt(actor, run_name, f"best_eval_{k}")
                    print(
                        f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )

        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_actor_ckpt(actor, run_name, str(iteration))
    envs.close()
    if args.track:
        wandb.finish()
