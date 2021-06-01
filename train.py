import numpy as np
import torch
import argparse
import os
import time
import json

from DRIBO import utils
from DRIBO.logger import Logger
from DRIBO.video import VideoRecorder

from DRIBO.DRIBO_sac import DRIBOSacAgent
from DRIBO import pytorch_util as ptu


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--pre_transform_image_size', default=119, type=int)

    parser.add_argument('--image_size', default=100, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='DRIBO_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--mib_seq_len', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--beta_start_value', default=1e-4, type=float)
    parser.add_argument('--beta_end_value', default=1e-3, type=float)
    parser.add_argument('--grad_clip', default=500, type=float)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    # try 0.05 or 0.1
    parser.add_argument('--critic_tau', default=0.01, type=float)
    # try to change it to 1 and retain 0.01 above
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='rssm', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-5, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--stochastic_dim', default=30, type=int)
    parser.add_argument('--deterministic_dim', default=200, type=int)
    parser.add_argument('--multi_view_skl', default=False, action='store_true')
    parser.add_argument('--kl_balance', default=False, action='store_true')
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')

    # noisy bg
    parser.add_argument('--noisy_bg', default=False, action='store_true')

    parser.add_argument('--log_interval', default=100, type=int)
    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            prev_state = None
            prev_action = None
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if args.encoder_type == 'rssm':
                    obs = utils.center_crop_image(obs, args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action, prev_action, prev_state = agent.sample_action(
                            obs, prev_action, prev_state
                        )
                    else:
                        action, prev_action, prev_state = agent.select_action(
                            obs, prev_action, prev_state
                        )
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)

        L.log('eval/' + prefix + 'eval_time', time.time()-start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

        return mean_ep_reward

    mean_ep_reward = run_eval_loop(sample_stochastically=False)
    L.dump(step)
    return mean_ep_reward


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'DRIBO_sac':
        return DRIBOSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            stochastic_size=args.stochastic_dim,
            deterministic_size=args.deterministic_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            multi_view_skl=args.multi_view_skl,
            mib_batch_size=args.batch_size,
            mib_seq_len=args.mib_seq_len,
            beta_start_value=args.beta_start_value,
            beta_end_value=args.beta_end_value,
            grad_clip=args.grad_clip,
            kl_balancing=args.kl_balance,
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main():
    args = parse_args()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    pre_transform_image_size = args.pre_transform_image_size
    # record the pre transform image size for translation
    # pre_image_size = args.pre_transform_image_size

    # resource_files = '~/packages/AdvGen/Invariant_RL/distractors/*.mp4'
    resource_files = '~/packages/AdvGen/kinetics-downloader' + \
        '/dataset/train/arranging_flowers/*.mp4'
    eval_resource_files = '~/packages/AdvGen/kinetics-downloader' + \
        '/dataset/test/*.mp4'
    img_source = 'video'
    total_frames = 1000
    if args.noisy_bg:
        from noisy_bg.envs import dmc2gym
        env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=resource_files,
            img_source=img_source,
            total_frames=total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'rssm'),
            height=pre_transform_image_size,
            width=pre_transform_image_size,
            frame_skip=args.action_repeat,
            frame_stack=args.frame_stack,
            extra='train',
        )
        eval_env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=eval_resource_files,
            img_source=img_source,
            total_frames=total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'rssm'),
            height=pre_transform_image_size,
            width=pre_transform_image_size,
            frame_skip=args.action_repeat,
            frame_stack=args.frame_stack,
            extra='eval',
        )
    else:
        import dmc2gym
        env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'rssm'),
            height=pre_transform_image_size,
            width=pre_transform_image_size,
            frame_skip=args.action_repeat
        )
        eval_env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'rssm'),
            height=pre_transform_image_size,
            width=pre_transform_image_size,
            frame_skip=args.action_repeat
        )

    env.seed(args.seed)
    eval_env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'rssm' and not args.noisy_bg:
        env = utils.FrameStack(env, k=args.frame_stack)
        eval_env = utils.FrameStack(eval_env, k=args.frame_stack)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    if args.noisy_bg:
        background = 'natural_video'
    else:
        background = 'clean'
    env_name = args.domain_name + '-' + args.task_name + '-' + background
    exp_name = env_name + '-' + ts + '-im' + str(args.image_size) + '-dim' + \
        str(args.encoder_feature_dim) + '-b' + str(args.batch_size) + '-s' \
        + str(args.seed) + '-' + args.encoder_type \
        + '-stacked_frames' + str(args.frame_stack) + \
        '-final_beta' + str(args.beta_end_value)
    args.work_dir = args.work_dir + '/' + exp_name

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ptu.device = device

    action_shape = env.action_space.shape

    if args.encoder_type == 'rssm':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (
            3*args.frame_stack,
            args.pre_transform_image_size, args.pre_transform_image_size
        )
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        path_len=1000 // args.action_repeat,
        device=device,
        image_size=args.image_size,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb, config='DRIBO')

    episode, episode_reward, done = 0, 0, True
    max_mean_ep_reward = 0
    start_time = time.time()

    for step in range(args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            mean_ep_reward = evaluate(
                eval_env, agent, video, args.num_eval_episodes, L, step, args
            )
            if args.save_model and mean_ep_reward > max_mean_ep_reward:
                max_mean_ep_reward = mean_ep_reward
                agent.save_DRIBO(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            prev_state = None
            prev_action = None
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action, prev_action, prev_state = agent.sample_action(
                    obs, prev_action, prev_state
                )

        # run training update
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()
