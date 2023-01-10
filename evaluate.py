import os
import torch
import numpy as np
from collections import deque

import hyperparams as hps
from test import evaluate
from procgen import ProcgenEnv

from baselines import logger
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
import json
import time

from ppo_daac_idaac import algo, utils
from ppo_daac_idaac.arguments import parser
from ppo_daac_idaac.model import PPOnet, IDAACnet, \
    LinearOrderClassifier, NonlinearOrderClassifier
from ppo_daac_idaac.storage import DAACRolloutStorage, \
    IDAACRolloutStorage, RolloutStorage
from ppo_daac_idaac.envs import VecPyTorchProcgen
from contexts import contexts

args = parser.parse_args()

if os.path.exists('results/eval-{}-{}-s{}-c{}.json'.format(
    args.env_name, args.algo, args.seed, args.context)):
    exit()

if not os.path.exists('models/agent-{}-{}-s{}-c{}.pt'.format(
    args.env_name, args.algo, args.seed, args.context)):
    exit()

device = torch.device("cuda:0")

venv = ProcgenEnv(num_envs=args.num_processes, env_name=args.env_name,
                  num_levels=args.num_levels, start_level=args.start_level,
                  distribution_mode=args.distribution_mode, 
                  context_options=[
                      contexts[args.env_name][args.context] for _ in range(args.num_processes)
                  ]
                  )

venv = VecExtractDictObs(venv, "rgb")
venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
venv = VecNormalize(venv=venv, ob=False)
envs = VecPyTorchProcgen(venv, 'cpu')
obs_shape = envs.observation_space.shape

actor_critic = PPOnet(
    obs_shape,
    envs.action_space.n,
    base_kwargs={'hidden_size': args.hidden_size})

# torch.save([
#     actor_critic,
#     getattr(envs, 'ob_rms', None)
# ], os.path.join(args.save_dir, "agent{}.pt".format(log_file))) 
pt_path = 'models/agent-{}-{}-s{}-c{}.pt'.format(
    args.env_name, args.algo, args.seed, args.context)

actor_critic = torch.load(pt_path)[0]

eval_episode_rewards = evaluate(
    args, actor_critic, device, contexts[args.env_name][args.context], episodes=100)

print(eval_episode_rewards)
print(np.mean(eval_episode_rewards))
print(np.std(eval_episode_rewards))
print(np.median(eval_episode_rewards))

res = {
    'mean': np.mean(eval_episode_rewards).item(),
    'std': np.std(eval_episode_rewards).item(),
    'median': np.median(eval_episode_rewards).item(),
    'data': [each.item() for each in eval_episode_rewards]
}
with open('results/eval-{}-{}-s{}-c{}.json'.format(
    args.env_name, args.algo, args.seed, args.context), 'w') as f:
    json.dump(res, f, indent=4)