import argparse
from Logger import logger
import gym
import torch
import numpy as np
import utils
import policy
import os
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from visualizer import Visualizer
# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0
    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            try:
                obs, reward, done, _ = env.step(action)
            except:
                print("the exception action is:", action)
                print("actor exception weight:", policy.actor.actor.weight.view(-1).sort()[0][0:50])
                print("critic exception weight:", policy.critic_dis.critic.weight.view(-1).sort()[0][0:50])

            avg_reward += reward
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(("Evaluation over %d episodes: %f") % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="DDPG")  # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--exp_name", default="HalfCheetah-v2") #the name of one experiment, always the same as env_name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates

    parser.add_argument("--save_net", default=1000, type=int)  # save frequency
    parser.add_argument("--save_num", default=20, type=int)  # number of saved network
    parser.add_argument("--sample_times", default=100, type=int)  # Frequency of delayed policy updates

    #对于Ant：VMAX 130 VMIN -60
    #对于HalfCheetah: VMAX 1000 VMIN -10
    #对于Hopper: VMAX 600 VMIN -60
    #对于Reacher: VMAX 0 VMIN -50
    #对于Walker: VMAX 700 VMIN -50

    VMIN = -10.0
    VMAX = 1000.0

    args = parser.parse_args()

    if args.env_name == 'HalfCheetah-v2':
        VMAX = 1000.0
        VMIN = -10.0
    elif args.env_name == 'Ant-v2':
        VMAX = 200.0
        VMIN = -60.0
    elif args.env_name == 'Hopper-v2':
        VMAX = 600.0
        VMIN = -60.0
    elif args.env_name == 'Reacher-v2':
        VMAX = 0.0
        VMIN = -50.0
    elif args.env_name == 'Walker2d-v2':
        VMAX = 700.0
        VMIN = -100.0
    elif args.env_name == 'InvertedDoublePendulum-v2':
        VMAX = 1500.0
        VMIN = 0.0
    elif args.env_name == 'InvertedPendulum-v2':
        VMAX = 160.0
        VMIN = 0.0
    elif args.env_name == 'Swimmer-v2':
        VMAX = 120.0
        VMIN = 0.0
    print('env info:**********')
    print('env name is:', args.env_name)

    sample_times = args.sample_times
    print("sample_times:", sample_times)

    print('VMAX is:' + str(VMAX) + " VMIN is:" + str(VMIN))
    print('**********')

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")
    log_stat = logger(args.exp_name, str(args.seed), "result/" + args.policy_name) #record the reward

    env = gym.make(args.env_name)

    # 保存网络参数的频率
    save_net_freq = args.save_net
    # 保存过去的网络参数的总数
    net_number = args.save_num

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vis = Visualizer('Distributed TPS-CI2 DDPG_' + args.exp_name + "_" + str(args.seed))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    replay_buffer = utils.ReplayBuffer()

    total_timesteps = 0
    episode_num = 0
    done = True

    episode_reward = 0
    episode_timesteps = 0
    timesteps_since_eval = 0
    timesteps_since_log = 0
    timesteps_since_plot = 0
    count_eval_times = 0
    # episode reward arr用来记录每个episode中每步获得的奖励，然后计算累计折扣回报
    episode_reward_arr = []
    # 记录所有的累计折扣回报
    total_discounted_reward_arr = []
    timesteps_since_save_net = 0
    #记录修改VMAX的时间步
    timesteps_since_update_vmax = 0
    #是否修改VMAX
    if_change_vmax = True
    min_reward = 0.0
    max_reward = 0.0


    policy = policy.DDPG(state_dim, action_dim, max_action, net_number)

    while total_timesteps < args.max_timesteps:
        if done:
            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))
                # 在一个episode结束的时候对policy进行训练

                temp_sum = 0
                for r in reversed(episode_reward_arr):
                    temp_sum *= args.discount
                    temp_sum += r
                total_discounted_reward_arr.append(temp_sum)
                #if timesteps_since_update_vmax >= 1000:
                #    timesteps_since_update_vmax %= 1000
                #    if_change_vmax = True

                actor_loss, mean_loss, current_dist, vmax = policy.train(replay_buffer, episode_timesteps, VMIN, VMAX, sample_times, if_change_vmax, args.batch_size, args.discount, args.tau)
                #if_change_vmax = False


                vis.plot(name='VMAX', y=vmax)
                if timesteps_since_log >= 1000:
                    timesteps_since_log %= 1000
                    vis.plot(name='actor_loss', y=(actor_loss).data)
                    vis.plot(name='critic_loss', y=(mean_loss).data)
                    q_value = torch.sum(current_dist * policy.critic_dis.get_z_atoms(VMIN, VMAX), 1)
                    vis.plot(name='averaged q value', y=(q_value.mean()).data)
                    vis.plot(name='current episode discounted reward', y=temp_sum)
                    vis.plot(name='average last 10 discounted reward', y=np.mean(total_discounted_reward_arr[-10:]))

                # 保存旧的网络参数，方便后续的采样操作
                if timesteps_since_save_net >= save_net_freq:
                    timesteps_since_save_net %= save_net_freq
                    policy.critic_arr[policy.count_critic % net_number].load_state_dict(
                        policy.critic.state_dict())
                    policy.count_critic += 1

                if timesteps_since_plot >= 50000:
                    timesteps_since_plot %= 50000
                    x = np.linspace(-10, 100, 51)
                    y = current_dist[0].detach().cpu().numpy()
                    plt.bar(x, y)
                    plt.xlabel('number')
                    plt.ylabel('value')
                    dis_save_dir = 'dist_result/' + args.exp_name + "_seed" + str(args.seed)
                    if not os.path.exists(dis_save_dir):
                        os.makedirs(dis_save_dir)

                    plt.savefig(dis_save_dir + "/" + str(total_timesteps) + '.jpg')
                    plt.close()

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                count_eval_times += 1
                eval_result = evaluate_policy(policy)
                log_stat.store(EpRet=eval_result)
                log_stat.log_tabular('Epoch', episode_num)
                log_stat.log_tabular('TotalEnvInteracts', count_eval_times * args.eval_freq)
                log_stat.log_tabular('EpRet', with_min_and_max=True)

                log_stat.dump_tabular()

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            episode_reward_arr = []
            #if episode_num % 25 == 0 and episode_num > 0:
            #    VMAX += 100


        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)
        try:
            new_obs, reward, done, _ = env.step(action)
        except:
            print("the exception action is:", action)
            print("exception weight:", policy.actor.actor.weight.view(-1).sort()[0][0:50])


        # if done:done_bool=1 else done_bool=0
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward
        # record every step reward
        episode_reward_arr.append(reward)
        # Store data in replay buffer

        replay_buffer.add((obs, new_obs, action, reward, done_bool))
        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        timesteps_since_log += 1
        timesteps_since_plot += 1
        timesteps_since_save_net += 1
        timesteps_since_update_vmax += 1