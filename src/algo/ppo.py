import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PPO():
    """
    PPO
    """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                #  context_prod_dim,
                 lr=None,
                 eps=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        # self.context_aloss = torch.zeros(context_prod_dim)
        # self.context_vloss = torch.zeros(context_prod_dim)
        # self.context_aloss_all = torch.zeros(context_prod_dim)
        # self.context_vloss_all = torch.zeros(context_prod_dim)
        
    def update(self, rollouts, context_weights):
        # self.context_aloss *= 0
        # self.context_vloss *= 0
        # self.context_aloss_all *= 0
        # self.context_vloss_all *= 0

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

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
                # self.context_aloss.index_add_(0, context_idx, (action_loss.abs() * loss_mask).view(-1).cpu().detach())
                # self.context_vloss.index_add_(0, context_idx, (value_loss.abs() * loss_mask).view(-1).cpu().detach())
                # self.context_aloss_all.index_add_(0, context_idx, (action_loss.abs()).view(-1).cpu().detach())
                # self.context_vloss_all.index_add_(0, context_idx, (value_loss.abs()).view(-1).cpu().detach())
                
                # context_weights_min = torch.min(context_weights)
                # weights = torch.log(context_weights/context_weights_min) + 1
                # weight_array = weights[context_idx].to(action_loss.device)
                # value_loss = value_loss * weight_array
                # action_loss = action_loss * weight_array

                self.optimizer.zero_grad()
                (value_loss.mean() * self.value_loss_coef + action_loss.mean() -
                    dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                        self.max_grad_norm)
                self.optimizer.step()  
                value_loss_epoch += value_loss.mean().item()
                action_loss_epoch += action_loss.mean().item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        # loss_weights = self.context_aloss + self.context_vloss * self.value_loss_coef
        # loss_weights = loss_weights / loss_weights.sum()
        # loss_weights_all = self.context_aloss_all + self.context_vloss_all * self.value_loss_coef
        # loss_weights_all = loss_weights_all / loss_weights_all.sum()

        loss_info = {
            # 'context_aloss': self.context_aloss.detach().cpu(),
            # 'context_vloss': self.context_vloss.detach().cpu(),
            # 'context_aloss_all': self.context_aloss_all.detach().cpu(),
            # 'context_vloss_all': self.context_vloss_all.detach().cpu(),
        }

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_info
