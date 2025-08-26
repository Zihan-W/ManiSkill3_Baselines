## Deployment

克隆本仓库：
```bash
  git clone https://github.com/Zihan-W/ManiSkill3_Baselines.git
```

安装依赖：
```bash
  cd ManiSkill3_Baselines
  pip install -r requirements.txt
  pip install -e .
```

登录wandb：
```bash
  wandb login --relogin
  # continue to login your wandb account
```

运行PPO baseline：

    运行ppo_rgbd.py即可，注意修改文件中的args参数
    默认actor为CLIP编码，critic为ResNet50编码
    首次运行会下载ResNet50的权重文件