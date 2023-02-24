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

from acl_settings import settings


def get_env(args, env_context, device):
    venv = ProcgenEnv(num_envs=args.num_processes, env_name=args.env_name,
                      num_levels=0, start_level=args.start_level,
                      distribution_mode=args.distribution_mode,
                      context_options=env_context
                      )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    envs = VecPyTorchProcgen(venv, device)
    return envs


def train(args):
    target_env_ratio = 0.25
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

    use_wandb = not args.debug

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    run_name = '{}-{}-s{}-c{}'.format(int(time.time() * 1000),
                                      args.env_name, args.seed, args.context)
    print("Run name: ", run_name)

    env_context = settings[args.env_name]["context_options"]
    context_prod_dim = 1
    for each in settings[args.env_name]["context_shape"]:
        context_prod_dim *= each
    context_encoder = settings[args.env_name]["context_encoder"]
    context_decoder = settings[args.env_name]["context_decoder"]

    # wandb init
    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="ContextualProcgenNew",

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
        wandb.config.update({
            "setting_id": 5,
            "setting_des": 'softmax(loss_weight_c) * 10, target_env_ratio = 0.25'
        })
    context_options = [env_context for _ in range(args.num_processes)]
    envs = get_env(args, context_options, device)

    obs_shape = envs.observation_space.shape
    if args.algo == 'ppo':
        actor_critic = PPOnet(
            obs_shape,
            envs.action_space.n,
            base_kwargs={'hidden_size': args.hidden_size}
        )
    else:
        actor_critic = IDAACnet(
            obs_shape,
            envs.action_space.n,
            base_kwargs={'hidden_size': args.hidden_size})
    actor_critic.to(device)
    print("\n Actor-Critic Network: ", actor_critic)

    if args.algo == 'idaac':
        if args.use_nonlinear_clf:
            order_classifier = NonlinearOrderClassifier(emb_size=args.hidden_size,
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
                                  envs.observation_space.shape, envs.action_space, target_env_ratio)

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
            context_prod_dim=context_prod_dim,
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

    # set a dque to store the last 20 episode reward in 16 different contexts
    context_episode_reward = [[] for _ in range(context_prod_dim)]
    context_episode_length = [[] for _ in range(context_prod_dim)]

    loss_weights = None
    loss_weights_p = None

    aloss_quantity = torch.zeros(context_prod_dim)
    aloss_quantity_all = torch.zeros(context_prod_dim)
    vloss_quantity = torch.zeros(context_prod_dim)
    vloss_quantity_all = torch.zeros(context_prod_dim)

    context_distribution_1 = torch.ones(context_prod_dim)
    context_distribution_2 = torch.ones(context_prod_dim)

    for j in range(num_updates):
        context_episode_reward = [each[-5:] for each in context_episode_reward]
        context_episode_length = [each[-5:] for each in context_episode_length]

        aloss_quantity *= 0.8
        aloss_quantity_all *= 0.8
        vloss_quantity *= 0.8
        vloss_quantity_all *= 0.8

        context_distribution_1 *= 0.8
        context_distribution_2 *= 0.8

        actor_critic.train()

        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                if args.algo == 'ppo':
                    value, action, action_log_prob = actor_critic.act(
                        rollouts.obs[step])
                else:
                    adv, value, action, action_log_prob = actor_critic.act(
                        rollouts.obs[step])

            obs, reward, done, infos = envs.step(action)

            # the idxs of the episode that is done
            if loss_weights is not None and loss_weights_p is not None:
                distribution_weights = torch.softmax(
                        (loss_weights*context_prod_dim)*10, dim=0)
                # distribution_weights = loss_weights_p
                for idx in (i for i, d in enumerate(done) if d and i >= args.num_processes * target_env_ratio):
                    # assign a random context to the episode that is done
                    random_context = torch.multinomial(
                        distribution_weights, 1).item()
                    envs.venv.venv.venv.venv.env.set_context_to(idx, context_decoder(random_context))

            # if loss_weights is None:
            #     envs.venv.venv.venv.venv.env.c_sp = -1
            # else:
            #     # generate a random number between 0 and 15, which probability is proportional to the loss weight
            #     try:
            #         distribution_weights = torch.softmax(
            #             (loss_weights*context_prod_dim)*10, dim=0)
            #         # distribution_weights = loss_weights_p / fram_propotion
            #         # distribution_weights = distribution_weights / torch.sum(distribution_weights)
            #         # distribution_weights = torch.softmax((distribution_weights*16-1)*2, dim=0)
            #         envs.venv.venv.venv.venv.env.c_sp = torch.multinomial(
            #             distribution_weights, 1).item()
            #     except Exception as e:
            #         print(e)
            #         envs.venv.venv.venv.venv.env.c_sp = -1
            contexts_infos = torch.tensor(
                [[context_encoder(info['episode_context'])] for info in infos],
                dtype=torch.long)
            if args.algo == 'idaac':
                levels = torch.LongTensor(
                    [info['level_seed'] for info in infos])
                if j == 0 and step == 0:
                    rollouts.levels[0].copy_(levels)

            indexs = torch.arange(args.num_processes)
            indexs_constant_d = torch.arange(
                int(args.num_processes * target_env_ratio))
            context_distribution_1 += torch.bincount(
                contexts_infos[indexs_constant_d].view(-1), minlength=context_prod_dim)
            context_distribution_2 += torch.bincount(
                contexts_infos[indexs].view(-1), minlength=context_prod_dim)

            for (index, info) in enumerate(infos):
                if 'episode' in info.keys():
                    if index < args.num_processes * target_env_ratio:
                        episode_info['episode_reward'].append(
                            info['episode']['r'])
                        episode_info['episode_length'].append(
                            info['episode']['l'])
                    context_episode_reward[contexts_infos[index].item()].append(
                        info['episode']['r'])
                    context_episode_length[contexts_infos[index].item()].append(
                        info['episode']['l'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            nsteps += 1
            nsteps[done == True] = 0
            if args.algo == 'idaac':
                rollouts.insert(obs, action, action_log_prob, value,
                                reward, masks, adv, levels, nsteps)
            elif args.algo == 'daac':
                rollouts.insert(obs, action, action_log_prob, value,
                                reward, masks, adv)
            else:
                rollouts.insert(obs, action, action_log_prob, value,
                                reward, masks, contexts_infos)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        if args.algo == 'idaac':
            rollouts.before_update()
            order_acc, order_loss, clf_loss, adv_loss, value_loss, \
                action_loss, dist_entropy = agent.update(rollouts)
        elif args.algo == 'daac':
            adv_loss, value_loss, action_loss, dist_entropy = agent.update(
                rollouts)
        else:
            value_loss, action_loss, dist_entropy, loss_info = agent.update(
                rollouts, None)
        rollouts.after_update()
        

        context_aloss = loss_info['context_aloss']
        context_vloss = loss_info['context_vloss']
        context_aloss_all = loss_info['context_aloss_all']
        context_vloss_all = loss_info['context_vloss_all']

        aloss_quantity += context_aloss
        vloss_quantity += context_vloss
        aloss_quantity_all += context_aloss_all
        vloss_quantity_all += context_vloss_all

        loss_quantity = aloss_quantity + vloss_quantity * args.value_loss_coef
        loss_quantity_all = aloss_quantity_all + vloss_quantity_all * args.value_loss_coef

        loss_weights = loss_quantity / loss_quantity.sum()
        loss_weights_all = loss_quantity_all / loss_quantity_all.sum()

        fram_propotion = context_distribution_1 / context_distribution_1.sum()
        fram_propotion_all = context_distribution_2 / context_distribution_2.sum()
        loss_weights_p = loss_quantity_all / fram_propotion_all
        loss_weights_p = loss_weights_p / loss_weights_p.sum()

        log_cur_update = {
            'total_num_steps': total_num_steps,
            'frame_dist_c': [item.item() for item in fram_propotion],
            'frame_dist_a': [item.item() for item in fram_propotion_all],
            'ls_weights_c': [item.item() for item in loss_weights],
            'ls_weights_a': [item.item() for item in loss_weights_all],
            'al_weights_c': [item.item() for item in context_aloss / context_aloss.sum()],
            'vl_weights_c': [item.item() for item in context_vloss/ context_vloss.sum()],
            'al_weights_a': [item.item() for item in context_aloss_all / context_aloss_all.sum()],
            'vl_weights_a': [item.item() for item in context_vloss_all / context_vloss_all.sum()],
            'ls_weights_p': [item.item() for item in loss_weights_p],
            'contexts_rwd': [np.mean(each) if len(each) > 0 else 0.0 for each in context_episode_reward],
            'contexts_len': [np.mean(each) if len(each) > 0 else 0.0 for each in context_episode_length],
        }

        def show_in_terminal(log_data, context_shape):
            print(context_shape)
            dim_1, dim_2 = context_shape
            red, yellow, green, blue, defualt = '\033[91m', '\033[93m', '\033[92m', '\033[94m', '\033[0m'
            pre_block = ' ' * ((14 - dim_1*2)//2)
            post_block = ' ' * (14 - dim_1*2 - (14 - dim_1*2)//2)
            lines = [*(pre_block for _ in range(dim_2)), ' ']
            for i in range(dim_2):
                for (key, values) in log_data.items():
                    if key in ['contexts_len', 'total_num_steps']:
                        continue
                    if key == 'contexts_rwd':
                        for j in range(dim_1):
                            color = red if (values[i*dim_1+j] > 8)else yellow if (
                                values[i*dim_1+j] > 5) else green if (values[i*dim_1+j] > 2) else blue
                            cur_block = color + '██' + defualt
                            lines[i] += cur_block
                    else:
                        for j in range(dim_1):
                            color = red if (values[i*dim_1+j] > 1/4) else yellow if (
                                values[i*dim_1+j] > 1/8) else green if (values[i*dim_1+j] > 1/context_prod_dim) else blue
                            cur_block = color + '██' + defualt
                            lines[i] += cur_block
                    lines[i] += post_block + pre_block
            for (key, values) in log_data.items():
                if key in ['contexts_len', 'total_num_steps']:
                    continue
                # print the key string in 12 chars
                lines[4] += key[:14].ljust(14)
            for line in lines:
                print(line)

        for key, value in log_cur_update.items():
            if key == 'total_num_steps':
                print('total_num_steps: ', value, 'mean_reward: ',
                      np.mean(episode_info['episode_reward']))
            elif key == 'contexts_len':
                print(key, ['{:>05.0f}'.format(item) for item in value])
            elif key == 'contexts_rwd':
                print(key, ['{:>05.2f}'.format(item) for item in value])
            else:
                print(key, ['{:>05.3f}'.format(item) for item in value])

        show_in_terminal(log_cur_update, settings[args.env_name]["context_shape"])

        # Save Model per args.model_save_interval total_num_steps
        if total_num_steps / args.model_save_interval > cur_save:
            cur_save += 1
            if use_wandb:
                if not os.path.exists(os.path.join(wandb.run.dir, 'models')):
                    os.makedirs(os.path.join(wandb.run.dir, 'models'))
                torch.save(actor_critic, os.path.join(wandb.run.dir,
                           'models/model_{}.pt'.format(total_num_steps)))
                wandb.save(os.path.join(wandb.run.dir,
                           'models/model_{}.pt'.format(total_num_steps)))

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
            if use_wandb:
                wandb.log(logs)
            print("\nTime: {}".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            print("Update {}, step {}:".format(j, total_num_steps))
            print("Last {} training episodes, mean/median reward {:.2f}/{:.2f}"
                  .format(len(episode_rewards), np.mean(episode_rewards),
                          np.median(episode_rewards)))
            episode_info = {
                'episode_reward': [],
                'episode_length': [],
            }

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
