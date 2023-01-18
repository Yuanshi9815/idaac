import os
import torch
import numpy as np
from collections import deque

import hyperparams as hps
from test import evaluate
from procgen import ProcgenEnv
from procgen.default_context import default_context_options

from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
import time
import datetime

import wandb

from ppo_daac_idaac import algo, utils
from ppo_daac_idaac.arguments import parser
from ppo_daac_idaac.model import PPOnet, IDAACnet, \
    LinearOrderClassifier, NonlinearOrderClassifier
from ppo_daac_idaac.storage import DAACRolloutStorage, \
    IDAACRolloutStorage, RolloutStorage
from ppo_daac_idaac.envs import VecPyTorchProcgen
from contexts import contexts

def get_env(args, env_context, device):
    venv = ProcgenEnv(num_envs=args.num_processes, env_name=args.env_name, \
        num_levels=0, start_level=args.start_level, \
        distribution_mode=args.distribution_mode,
        context_options=env_context
        )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    envs = VecPyTorchProcgen(venv, device)
    return envs

def train(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.use_best_hps:
        args.value_epoch = hps.value_epoch[args.env_name]
        args.value_freq = hps.value_freq[args.env_name]
        args.adv_loss_coef = hps.adv_loss_coef[args.env_name]
        args.clf_hidden_size = hps.clf_hidden_size[args.env_name]
        args.order_loss_coef = hps.order_loss_coef[args.env_name]
        if args.env_name in hps.nonlin_envs:
            args.use_nonlinear_clf = True
        else:
            args.use_nonlinear_clf = False
    print("\nArguments: ", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    run_name = '{}-{}-s{}-c{}'.format(int(time.time() * 1000), args.env_name, args.seed, args.context)
    # make runs directory if it doesn't exist
    if not os.path.exists(os.path.join('runs', run_name)):
        os.makedirs(os.path.join('runs', run_name))
    print("Run name: ", run_name)

    # wandb init
    wandb.init(
        # set the wandb project where this run will be logged
        project="ContextualProcgen",
        
        # track hyperparameters and run metadata
        config={
            "env_name": args.env_name,
            "context_id": args.context,
            "seed": args.seed,
            "algo": args.algo,
            "num_processes": args.num_processes,
        },

        # set the name of the run
        name=run_name,
    )
    env_context = {
        **default_context_options[args.env_name],
        **contexts[args.env_name][args.context]
    }
    # Add env_context into wandb config
    wandb.config.update({"env_context": {"context_d_based": True}})
    context_options = [env_context for _ in range(args.num_processes)]
    envs = get_env(args, context_options, device)
    
    obs_shape = envs.observation_space.shape     
    if args.algo == 'ppo':
        actor_critic = PPOnet(
            obs_shape,
            envs.action_space.n,
            base_kwargs={'hidden_size': args.hidden_size})    
    else:           
        actor_critic = IDAACnet(
            obs_shape,
            envs.action_space.n,
            base_kwargs={'hidden_size': args.hidden_size})    
    actor_critic.to(device)
    print("\n Actor-Critic Network: ", actor_critic)
    
    if args.algo == 'idaac':
        if args.use_nonlinear_clf:
            order_classifier = NonlinearOrderClassifier(emb_size=args.hidden_size, \
                hidden_size=args.clf_hidden_size).to(device)       
        else:
            order_classifier = LinearOrderClassifier(emb_size=args.hidden_size)
        order_classifier.to(device)
        print("\n Order Classifier: ", order_classifier)

    if args.algo == 'idaac':
        rollouts = IDAACRolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space)
    elif args.algo == 'daac':
        rollouts = DAACRolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space)
    else:
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space)

    batch_size = int(args.num_processes * args.num_steps / args.num_mini_batch)

    if args.algo == 'idaac':
        agent = algo.IDAAC(
            actor_critic,
            order_classifier,
            args.clip_param,
            args.ppo_epoch,
            args.value_epoch, 
            args.value_freq,
            args.num_mini_batch,
            args.value_loss_coef,
            args.adv_loss_coef,
            args.order_loss_coef, 
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'daac':
        agent = algo.DAAC(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.value_epoch, 
            args.value_freq,
            args.num_mini_batch,
            args.value_loss_coef,
            args.adv_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    else: 
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)


    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_info = {
        'episode_reward': [],
        'episode_length': [],
    }
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes 

    nsteps = torch.zeros(args.num_processes)
    cur_save = 1
    phase_num = 0
    context_distribution_1 = torch.ones(16)
    context_distribution_2 = torch.ones(16)
    loss_weights = None
    for j in range(num_updates):
        context_distribution_1 *= 0.0
        context_distribution_2 *= 0.0
        actor_critic.train()

        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                if args.algo == 'ppo':
                    value, action, action_log_prob = actor_critic.act(rollouts.obs[step])
                else:
                    adv, value, action, action_log_prob = actor_critic.act(rollouts.obs[step])
                                        
            obs, reward, done, infos = envs.step(action)
            if loss_weights is None:
                envs.venv.venv.venv.venv.env.c_sp = 1
            else:
                # generate a random number between 0 and 15, which probability is proportional to the loss weight
                envs.venv.venv.venv.venv.env.c_sp = torch.multinomial(loss_weights, 1).item()
            # context_infos
            contexts_infos = torch.tensor([[info['context_info']] for info in infos], dtype=torch.long)
            # print(contexts_infos)

            if step == args.num_steps - 1:
                if total_num_steps > 5e6 and phase_num < 1:
                    phase_num += 1
                    context_options = [env_context for _ in range(args.num_processes)]
                    envs = get_env(args, context_options, device)
                    obs = envs.reset()
                    reward = torch.zeros((args.num_processes,1))
                    done = np.array([True for _ in range(args.num_processes)], dtype=bool)
                    infos = [{} for _ in range(args.num_processes)]

                elif total_num_steps > 10e6 and phase_num < 2:
                    phase_num += 1
                    context_options = [env_context for _ in range(args.num_processes)]
                    envs = get_env(args, context_options, device)
                    obs = envs.reset()
                    reward = torch.zeros((args.num_processes,1))
                    done = np.array([True for _ in range(args.num_processes)], dtype=bool)
                    infos = [{} for _ in range(args.num_processes)]

            if args.algo == 'idaac':
                levels = torch.LongTensor([info['level_seed'] for info in infos])
                if j == 0 and step == 0:
                    rollouts.levels[0].copy_(levels)

            for index in range(args.num_processes):
                if index < args.num_processes * 0.5:
                    context_distribution_1[contexts_infos[index]] += 1
                context_distribution_2[contexts_infos[index]] += 1

            for (index, info) in enumerate(infos):
                if 'episode' in info.keys() and index < args.num_processes * 0.5:
                    episode_info['episode_reward'].append(info['episode']['r'])
                    episode_info['episode_length'].append(info['episode']['l'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            nsteps += 1 
            nsteps[done == True] = 0
            if args.algo == 'idaac':
                rollouts.insert(obs, action, action_log_prob, value, \
                                reward, masks, adv, levels, nsteps)
            elif args.algo == 'daac':
                rollouts.insert(obs, action, action_log_prob, value, \
                                reward, masks, adv)
            else:
                rollouts.insert(obs, action, action_log_prob, value, \
                                reward, masks, contexts_infos)

        context_weights_1 = context_distribution_1 / context_distribution_1.sum()
        context_weights_2 = context_distribution_2 / context_distribution_2.sum()
        context_weights = context_weights_1 / context_weights_2
        min_weight = max(context_weights.min().item(), 0.2)
        context_weights = context_weights / min_weight

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()
        
        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        if args.algo == 'idaac':
            rollouts.before_update()
            order_acc, order_loss, clf_loss, adv_loss, value_loss, \
                action_loss, dist_entropy = agent.update(rollouts)    
        elif args.algo == 'daac':
            adv_loss, value_loss, action_loss, dist_entropy = agent.update(rollouts)    
        else:
            value_loss, action_loss, dist_entropy, loss_weights = agent.update(rollouts, context_weights)    
        rollouts.after_update()
        print('context_weights: ', context_weights)
        print('loss_weights: ', loss_weights, '\n')

        # Save Model per args.model_save_interval total_num_steps
        if total_num_steps / args.model_save_interval > cur_save:
            cur_save += 1
            if not os.path.exists(os.path.join(wandb.run.dir, 'models')):
                os.makedirs(os.path.join(wandb.run.dir, 'models'))
            torch.save(actor_critic, os.path.join(wandb.run.dir, 'models/model_{}.pt'.format(total_num_steps)))
            # wandb
            wandb.save(os.path.join(wandb.run.dir, 'models/model_{}.pt'.format(total_num_steps)))


        # Save Logs
        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            logs = {
                "time_stamp": time.time(),
                "update": j,
                "total_num_steps": total_num_steps,
                "value_loss": value_loss,
                "action_loss": action_loss,
                "dist_entropy": dist_entropy,
            }
            if len(episode_info['episode_length']) > 1:
                episode_rewards = episode_info['episode_reward']
                episode_lengths = episode_info['episode_length']
                logs = {**logs, **{
                    "mean_reward": np.mean(episode_rewards).item(),
                    "median_reward": np.median(episode_rewards).item(),
                    "min_reward": np.min(episode_rewards).item(),
                    "max_reward": np.max(episode_rewards).item(),
                    "mean_length": np.mean(episode_lengths).item(),
                    "median_length": np.median(episode_lengths).item(),
                    "min_length": np.min(episode_lengths).item(),
                    "max_length": np.max(episode_lengths).item(),
                    "episode_info": episode_info,
                }}
            wandb.log(logs)
            print("\nTime: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            print("Update {}, step {}:".format(j, total_num_steps))
            print("Last {} training episodes, mean/median reward {:.2f}/{:.2f}"\
                .format(len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards)))
            episode_info = {
                'episode_reward': [],
                'episode_length': [],
            }

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
