import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import torch.optim as optim
import numpy as np

class PPG():
    """
    PPG
    """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 context_space,
                 lr=None,
                 eps=None,
                 max_grad_norm=None, 
                 ppg_auxiliary_epoch=8,
                 kl_coef=1):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.context_space = context_space
        self.ppg_auxiliary_epoch = ppg_auxiliary_epoch
        self.kl_coef = kl_coef

    def update_auxiliary(self, rollouts, episode_i, auxiliary_interval_n):
        for e in range(self.ppg_auxiliary_epoch):
            data_generator = rollouts.feed_forward_generator(None, self.num_mini_batch)
            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                    old_action_log_probs_batch, adv_targ, context_idx, loss_mask = sample
                
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    obs_batch, actions_batch)
                
                aux_loss = (values - return_batch).pow(2)
                kl_loss = kl_divergence(
                    Categorical(action_log_probs),
                    Categorical(old_action_log_probs_batch)
                ).unsqueeze(1).float()

                sum_loss = aux_loss + kl_loss * self.kl_coef

                # update loss
                self.optimizer.zero_grad()
                (sum_loss.mean()).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                        self.max_grad_norm)
                self.optimizer.step()

        

    def update(self, rollouts, episode_i, auxiliary_interval_n):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        ploss_t = torch.zeros(self.context_space)
        vloss_t = torch.zeros(self.context_space)
        entrp_t = torch.zeros(self.context_space)
        tloss_t = torch.zeros(self.context_space)
        steps_t = torch.zeros(self.context_space)
        ploss_c = torch.zeros(self.context_space)
        vloss_c = torch.zeros(self.context_space)
        entrp_c = torch.zeros(self.context_space)
        tloss_c = torch.zeros(self.context_space)
        steps_c = torch.zeros(self.context_space)


        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                    old_action_log_probs_batch, adv_targ, context_idx, loss_mask = sample
                
                context_idx = context_idx.to('cpu').squeeze(1)

                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    obs_batch, actions_batch)
                    
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2)

                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                                value_losses_clipped)

                # 将loss mask转换为两层的tensor，一层是loss mask，一层是1-loss mask
                loss_mask = loss_mask.view(-1, 1)
                loss_mask = torch.cat((loss_mask, 1-loss_mask), dim=1)

                # loss mask对应的是target，也就是plow_t, vloss_t, entrp_t
                # 而1-loss mask对应的是contextal的，也就是ploss_c, vloss_c, entrp_c
                action_loss_separated = action_loss.view(-1, 1) * loss_mask
                value_loss_separated = value_loss.view(-1, 1) * loss_mask
                dist_entropy_separated = dist_entropy.view(-1, 1) * loss_mask

                # 将loss mask对应的loss加到target上，将1-loss mask对应的loss加到contextal上
                ploss_t.index_add_(0, context_idx, action_loss_separated[:, 0].abs().cpu().detach())
                vloss_t.index_add_(0, context_idx, value_loss_separated[:, 0].abs().cpu().detach())
                entrp_t.index_add_(0, context_idx, dist_entropy_separated[:, 0].abs().cpu().detach())
                tloss_t.index_add_(0, context_idx, (action_loss_separated[:, 0] + value_loss_separated[:, 0] - dist_entropy_separated[:, 0] * self.entropy_coef).abs().cpu().detach())
                ploss_c.index_add_(0, context_idx, action_loss_separated[:, 1].abs().cpu().detach())
                vloss_c.index_add_(0, context_idx, value_loss_separated[:, 1].abs().cpu().detach())
                entrp_c.index_add_(0, context_idx, dist_entropy_separated[:, 1].abs().cpu().detach())
                tloss_c.index_add_(0, context_idx, (action_loss_separated[:, 1] + value_loss_separated[:, 1] - dist_entropy_separated[:, 1] * self.entropy_coef).abs().cpu().detach())

                steps_t.index_add_(0, context_idx, torch.ones_like(context_idx) * loss_mask[:, 0].sum().cpu().detach())
                steps_c.index_add_(0, context_idx, torch.ones_like(context_idx) * loss_mask[:, 1].sum().cpu().detach())

                self.optimizer.zero_grad()
                (value_loss.mean() * self.value_loss_coef + action_loss.mean() -
                    dist_entropy.mean() * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                        self.max_grad_norm)
                self.optimizer.step()  
                value_loss_epoch += value_loss.mean().item()
                action_loss_epoch += action_loss.mean().item()
                dist_entropy_epoch += dist_entropy.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        loss_info = {
            'policy_loss_t': ploss_t,
            'policy_loss_c': ploss_c,
            'value_loss_t': vloss_t,
            'value_loss_c': vloss_c,
            'entropy_t': entrp_t,
            'entropy_c': entrp_c,
            'total_loss_t': tloss_t,
            'total_loss_c': tloss_c,
            'steps_t': steps_t,
            'steps_c': steps_c,
        }
        
        if not episode_i % auxiliary_interval_n:
            auxiliary_log = self.update_auxiliary(rollouts, episode_i, auxiliary_interval_n)

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_info
