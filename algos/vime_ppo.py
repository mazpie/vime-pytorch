import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np

from collections import deque
from misc.arguments import get_args

args = get_args()
device = torch.device("cuda:0" if args.cuda else "cpu")

from misc.arguments import get_args

class VIME_PPO():
    def __init__(self,
                 actor_critic,
                 dynamics,
                 replay_pool,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 min_pool_size=1000,
                 kl_q_len=10):

        self.actor_critic = actor_critic
        self.dynamics = dynamics
        self.replay_pool = replay_pool

        self.min_pool_size= min_pool_size

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.kl_q_len = kl_q_len
        self._kl_mean = deque(maxlen=self.kl_q_len)
        self._kl_std = deque(maxlen=self.kl_q_len)
        self.kl_previous = deque(maxlen=self.kl_q_len)

    def update(self, rollouts):
        mod_rollouts = copy.deepcopy(rollouts)

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        ### Train Dynamics

        n_updates_per_sample = 500
        pool_batch_size = 10

        if self.replay_pool.size >= self.min_pool_size:
            obs_mean, obs_std, act_mean, act_std = self.replay_pool.mean_obs_act()
            _inputss = []
            _targetss = []
            for _ in range(n_updates_per_sample): # TODO: self.n_updates_per_sample):
                batch = self.replay_pool.random_batch(
                    pool_batch_size)
                obs = (batch['observations'] - obs_mean) / \
                      (obs_std + 1e-8)
                next_obs = (
                                   batch['next_observations'] - obs_mean) / (obs_std + 1e-8)
                act = (batch['actions'] - act_mean) / \
                      (act_std + 1e-8)
                _inputs = np.hstack(
                    [obs, act])
                _targets = next_obs
                _inputss.append(_inputs)
                _targetss.append(_targets)

            _inputss = torch.Tensor(_inputss).to(device)
            _targetss = torch.Tensor(_targetss).to(device)

            old_acc = 0.
            for _inputs, _targets in zip(_inputss, _targetss):
                _out = self.dynamics.pred_fn(_inputs)
                old_acc += torch.mean((_out - _targets)**2)
            old_acc /= len(_inputss)

            print(f"Old Accuracy: {old_acc}")

            for _inputs, _targets in zip(_inputss, _targetss):
                self.dynamics.train_fn(_inputs, _targets)

            new_acc = 0.
            for _inputs, _targets in zip(_inputss, _targetss):
                _out = self.dynamics.pred_fn(_inputs)
                new_acc += torch.mean((_out - _targets)**2)
            new_acc /= len(_inputss)

            print(f"New Accuracy: {new_acc}")
            ###

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


