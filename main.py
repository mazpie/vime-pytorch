import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algos.ppo import PPO
from algos.vime_ppo import VIME_PPO
from dynamics.bnn import BNN
from misc import utils
from misc.arguments import get_args
from misc.envs import make_vec_envs
from models.model import Policy
from misc.storage import RolloutStorage
from misc.evaluation import evaluate
from misc.replay_pool import ReplayPool
import torch.multiprocessing as mp


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'ppo':
        agent = PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'vime-ppo':
        if envs.action_space.__class__.__name__ == "Discrete":
            action_dim = envs.action_space.n
        elif envs.action_space.__class__.__name__ == "Box":
            action_dim = envs.action_space.shape[0]
        elif envs.__class__.__name__ == "MultiBinary":
            action_dim = envs.action_space.shape[0]
        else:
            raise NotImplementedError

        dynamics = BNN(
            n_in=envs.observation_space.shape[0] + action_dim,
            n_hidden=[32],
            n_out=envs.observation_space.shape[0],
            n_batches=args.num_mini_batch)
        dynamics.to(device)

        replay_pool = ReplayPool(
            max_pool_size=args.replay_pool_size,
            observation_shape=envs.observation_space.shape[0],
            action_dim=action_dim
        )

        agent = VIME_PPO(
            actor_critic,
            dynamics,
            replay_pool,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    else:
        raise NotImplementedError

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            if args.algo == 'vime-ppo':
                agent.replay_pool.add_sample(obs, action, reward, done)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.algo == 'vime-ppo':
            # Computing intrinsic rewards.
            # ----------------------------

            # Mean/std obs/act based on replay pool.
            obs_mean, obs_std, act_mean, act_std = agent.replay_pool.mean_obs_act()
            # Save original reward.
            second_order_update = True
            kl_batch_size = 1
            use_replay_pool = True
            n_itr_update = 1
            normalize_reward = False
            use_kl_ratio = True,
            use_kl_ratio_q = True
            eta = args.eta

            if j > 0:
                # Iterate over all paths and compute intrinsic reward by updating the
                # model on each observation, calculating the KL divergence of the new
                # params to the old ones, and undoing this operation.
                obs = rollouts.obs.reshape((args.num_steps + 1) * args.num_processes, -1) # observation should be already normalized
                act = ((rollouts.actions - act_mean) / (act_std + 1e-8))\
                    .reshape(args.num_steps* args.num_processes, -1)
                rew = rollouts.rewards
                # inputs = (o,a), target = o'

                obs_nxt = np.empty((0,obs.shape[1]))
                _inputs = np.empty((0,obs.shape[1] + act.shape[1]))
                for i in range(args.num_processes):
                    start = args.num_steps * i + i
                    end = args.num_steps * (i + 1) + i
                    obs_nxt = np.vstack([obs_nxt,obs[start + 1:end+1]])
                    _inputs = np.vstack([_inputs,np.hstack([obs[start:end], act[start - i:end - i]])])
                    _targets = obs_nxt

                _inputs = torch.Tensor(_inputs).to(device)
                _targets = torch.Tensor(_targets).to(device)

                # KL vector assumes same shape as reward.
                kl = torch.zeros(rew.shape)

                processes = []
                if args.num_processes == 1:
                    compute_intrinsic_reward(agent.dynamics, 0, _inputs, _targets, kl, args, kl_batch_size,
                                             second_order_update, n_itr_update, use_replay_pool)
                else:
                    for p in range(args.num_processes):
                        import copy
                        dynamics = copy.deepcopy(agent.dynamics)
                        p = mp.Process(target=compute_intrinsic_reward, args=(dynamics,p, _inputs, _targets, kl,
                                                           args, kl_batch_size, second_order_update,
                                                           n_itr_update, use_replay_pool))
                        p.start()
                        processes.append(p)
                    for p in processes:
                        p.join()

                # Last element in KL vector needs to be replaced by second last one
                # because the actual last observation has no next observation.
                kl[-1] = kl[-2]

                # Perform normalization of the intrinsic rewards.
                if use_kl_ratio:
                    if use_kl_ratio_q:
                        # Update kl Q
                        agent.kl_previous.append(np.median(np.hstack(kl)))
                        previous_mean_kl = np.mean(np.asarray(agent.kl_previous))
                        kl = kl / previous_mean_kl

                # Add KL as intrinsic reward to external reward
                print(f"Sum of extrinsic (normalized) rewards: {rollouts.rewards.sum()}")
                rollouts.rewards = rollouts.rewards + eta * kl
                print(f"Sum of combined (normalized) rewards: {rollouts.rewards.sum()}")

                # Discount eta TODO?
                # eta *= eta_discount

            # ----------------------------

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
            or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), # Last {}
                            np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


def compute_intrinsic_reward(dynamics, p, _inputs, _targets, kl, args, kl_batch_size, second_order_update, n_itr_update, use_replay_pool):
    for k in range(p * args.num_steps,
                   int((p * args.num_steps) + np.ceil(args.num_steps / float(kl_batch_size)))):

        # Save old params for every update.
        dynamics.save_old_params()
        start = k * kl_batch_size
        end = np.minimum(
            (k + 1) * kl_batch_size, _targets.shape[0] - 1)

        if second_order_update:
            # We do a line search over the best step sizes using
            # step_size * invH * grad
            #                 best_loss_value = np.inf
            for step_size in [0.01]:
                dynamics.save_old_params()
                loss_value = dynamics.train_update_fn(
                    _inputs[start:end], _targets[start:end], step_size)
                loss_value = loss_value.detach()
                kl_div = np.clip(loss_value, 0, 1000)
                # If using replay pool, undo updates.
                if use_replay_pool:
                    dynamics.reset_to_old_params()
        else:
            # Update model weights based on current minibatch.
            for _ in range(n_itr_update):
                dynamics.train_update_fn(
                    _inputs[start:end], _targets[start:end])
            # Calculate current minibatch KL.
            kl_div = np.clip(
                float(dynamics.f_kl_div_closed_form().detach()), 0, 1000)

        for k in range(start, end):
            index = k % args.num_steps
            kl[index][p] = kl_div

        # If using replay pool, undo updates.
        if use_replay_pool:
            dynamics.reset_to_old_params()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()