import os
import datetime
import time
import numpy as np
from collections import deque
import json

import hyperparams as hps
import torch
from procgen import ProcgenEnv
from procgen.default_context import default_context_options
from procgen_settings.context_sampler import sample_bigfish_conext


from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)

import wandb

from src import algo, utils
from src.arguments import parser
from src.model import PPOnet
from src.storage import RolloutStorage
from src.envs import VecPyTorchProcgen
from src.context_monitor import ContextMonitor


def get_env(args, env_context, device):
    penv = ProcgenEnv(num_envs=args.num_processes, env_name=args.env_name,
                      num_levels=0, start_level=args.start_level,
                      distribution_mode=args.distribution_mode,
                      context_options=env_context
                      )
    venv = VecExtractDictObs(penv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    venv = VecPyTorchProcgen(venv, device)
    venv.get_context = penv.env.get_context
    venv.set_context_to = penv.env.set_context_to
    return venv


def train(args):
    contexts_list = ['' for _ in range(args.num_processes)]
    target_env_ratio = 0.5 # the ratio of target env to all envs
    context_space = 150  # the maximum number of discrete contexts
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("\nArguments: ", args)

    use_wandb = not args.debug

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    run_name = '{}-{}-s{}-c{}'.format(int(time.time() * 1000),
                                      args.env_name, args.seed, args.context_setting)
    print("Run name: ", run_name)
    context_monitor = ContextMonitor(target_env_ratio, context_space, 'log/{}'.format(run_name))
    context_monitor_test = ContextMonitor(target_env_ratio, context_space, 'log/{}'.format(run_name), is_test=True)

    # 创建Log的文件夹，在'log/{}'.format(run_name)下
    if not os.path.exists('log/{}'.format(run_name)):
        os.makedirs('log/{}'.format(run_name))

    # wandb init
    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="ContextualProcgenNew",

            # track hyperparameters and run metadata
            config={
                "env_name": args.env_name,
                "context_id": args.context_setting,
                "seed": args.seed,
                "algo": args.algo,
                "num_processes": args.num_processes,
            },

            # set the name of the run
            name=run_name,
        )
        wandb.config.update({
            "setting_id": 1,
            "setting_des": 'softmax(loss_weight_c) * 10, target_env_ratio = 0.25'
        })
    
    envs = get_env(args, [
        sample_bigfish_conext() for _ in range(args.num_processes)
    ], device)

    test_envs = get_env(args, [
        sample_bigfish_conext(hardmode=True) for _ in range(args.num_processes)
    ], device)

    obs_shape = envs.observation_space.shape
    actor_critic = PPOnet(
        obs_shape,
        envs.action_space.n,
        base_kwargs={'hidden_size': args.hidden_size}
    )
    actor_critic.to(device)
    print("\n Actor-Critic Network: ", actor_critic)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space, target_env_ratio)

    batch_size = int(args.num_processes * args.num_steps / args.num_mini_batch)

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        context_space=context_space,
        max_grad_norm=args.max_grad_norm)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    nsteps = torch.zeros(args.num_processes)
    cur_save = 1

    episode_returns = []

    for j in range(num_updates):
        # Before algo update
        context_monitor.before_algo_step()

        actor_critic.train()

        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        for step in range(args.num_steps):
            # Before env step
            contexts_idxs = context_monitor.extent(envs.get_context())


            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                    rollouts.obs[step])
            obs, reward, done, infos = envs.step(action)

            # Assign context to envs which finished the previous episode
            for idx in (i for i, d in enumerate(done) if d):
                envs.set_context_to(idx, sample_bigfish_conext())
            

            for (index, info) in enumerate(infos):
                if 'episode' in info.keys():
                    context_monitor.add_episode_info(
                        contexts_idxs[index],
                        index < args.num_processes * target_env_ratio,
                        {
                            'episode_return': float(info['episode']['r']),
                            'episode_length': int(info['episode']['l'])
                        }
                    )

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            nsteps += 1
            nsteps[done == True] = 0
            rollouts.insert(obs, action, action_log_prob, value,
                            reward, masks, contexts_idxs)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)
        value_loss, action_loss, dist_entropy, loss_info = agent.update(
            rollouts, None)
        rollouts.after_update()

        context_monitor.add_step_info(loss_info)

        context_monitor.after_algo_step()

        # Print Logs per update
        print('total_num_steps: ', total_num_steps, 'mean_reward: ',
                      np.mean(context_monitor.episode_info_t_env['episode_return']))
      
        # Print Logs
        if j % args.log_interval == 0:
            if len(context_monitor.episode_info_t_env['episode_length']) > 1:
                episode_returns = context_monitor.episode_info_t_env['episode_return']
            print("\nTime: {}".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            print("Update {}, step {}:".format(j, total_num_steps))
            print("Last {} training episodes, mean/median reward {:.2f}/{:.2f}"
                .format(len(episode_returns), np.mean(episode_returns),
                        np.median(episode_returns)))
            # Evaluating on the test envs
            print("Evaluating on the test envs...")
            evaluation(test_envs, actor_critic, context_monitor_test)
            
def evaluation(envs, actor_critic, context_monitor, min_episodes_num=5, max_steps_num=1000):
    obs = envs.reset()
    context_monitor.before_algo_step()
    for step_i in range(max_steps_num * 2):
        contexts_idxs = context_monitor.extent(envs.get_context())
        with torch.no_grad():
            value, action, action_log_prob = actor_critic.act(obs)
        obs, reward, done, infos = envs.step(action)
        # Assign context to envs which finished the previous episode
        for idx in (i for i, d in enumerate(done) if d):
            envs.set_context_to(idx, sample_bigfish_conext(hardmode=False))
        # Record episode info
        for (index, info) in enumerate(infos):
            if 'episode' in info.keys():
                context_monitor.add_episode_info(
                    contexts_idxs[index],
                    index < args.num_processes * 0.5,
                    {
                        'episode_return': float(info['episode']['r']),
                        'episode_length': int(info['episode']['l'])
                    }
                )
        contexts_episodes_num = [
            len(each['episode_length']) for each in context_monitor.contextual_episode_info
        ]
        if np.all(np.array(contexts_episodes_num) >= min_episodes_num):
            print("case 1")
            break
        if np.all(np.array(contexts_episodes_num) >= 1) and step_i > max_steps_num:
            print("case 2")
            break
    context_monitor.after_algo_step()
    


if __name__ == "__main__":
    args = parser.parse_args()
    # args.env_name = 'bigfish'
    train(args)
