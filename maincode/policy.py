import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#moving block bootstrap getting the max value
def moving_block_bootstrap_max(data):
    alpha = 0.05
    # 采样次数
    num_samples = 100

    total_length = len(data)
    block_length = int(pow(total_length, 0.33))
    num_block = int(total_length / block_length)
    # 100次采样，得到采样到的block
    idx = np.random.randint(0, num_block, size=(num_samples, num_block))
    # 每次采样都是一个episode，然后求解最大值
    temp_vmax = []
    for index in idx:
        sample_episode = []
        # 针对一次采样求出最大值
        for single_index in index:
            start_index = single_index * block_length
            end_index = (single_index + 1) * block_length
            if single_index == (num_block - 1):
                end_index = total_length
            sample_episode.append(data[start_index:end_index])
        temp_vmax.append(max(map(max, sample_episode)))
    stat = np.sort(temp_vmax)
    j = int(num_samples * alpha / 2 + (alpha + 2) / 6)
    r = num_samples * alpha / 2 + (alpha + 2) / 6 - j
    T = (1 - r) * stat[j] + r * stat[j + 1]
    return 2 * max(data) - T

def _l2_project(z_p, p, z_q):
    """Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.
    The supports z_p and z_q are specified as tensors of distinct atoms (given
    in ascending order).
    Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
    support z_q, in particular Kq need not be equal to Kp.
    Args:
      z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
      p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
      z_q: Tensor holding support to project onto, shape `[Kq]`.
    Returns:
      Projection of (z_p, p) onto support z_q under Cramer distance.
    """
    # Broadcasting of tensors is used extensively in the code below. To avoid
    # accidental broadcasting along unintended dimensions, tensors are defensively
    # reshaped to have equal number of dimensions (3) throughout and intended
    # shapes are indicated alongside tensor definitions. To reduce verbosity,
    # extra dimensions of size 1 are inserted by indexing with `None` instead of
    # `tf.expand_dims()` (e.g., `x[:, None, :]` reshapes a tensor of shape
    # `[k, l]' to one of shape `[k, 1, l]`).

    # Extract vmin and vmax and construct helper tensors from z_q
    vmin, vmax = z_q[0], z_q[-1]

    #d_pos:将vmin放在最后面，维度没有发生变化，仍然是一维
    d_pos = torch.cat([z_q, vmin[None]], 0)[1:]
    #d_neg:将vmax放在最前面，维度没有发生变化，仍然是一维
    d_neg = torch.cat([vmax[None], z_q], 0)[:-1]
    # Clip z_p to be in new support range (vmin, vmax).
    # clip经过bellman更新后的值函数区间，然后维度变成batch_size×1×Kp，kp为可取值函数的数量
    # z_p的维度为batch_size*1*Kp
    z_p = torch.clamp(z_p, vmin, vmax)[:, None, :]  # B x 1 x Kp

    # Get the distance between atom values in support.
    #z_q的维度也为一维，为设定的值函数区间
    d_pos = (d_pos - z_q)[None, :, None]  # z_q[i+1] - z_q[i]. 1 x B x 1
    d_neg = (z_q - d_neg)[None, :, None]  # z_q[i] - z_q[i-1]. 1 x B x 1
    z_q = z_q[None, :, None]  # 1 x Kq x 1

    # Ensure that we do not divide by zero, in case of atoms of identical value.
    d_neg = torch.where(d_neg > 0, 1. / d_neg, torch.zeros_like(d_neg))  # 1 x Kq x 1
    d_pos = torch.where(d_pos > 0, 1. / d_pos, torch.zeros_like(d_pos))  # 1 x Kq x 1

    delta_qp = z_p - z_q  # clip(z_p)[j] - z_q[i]. B x Kq x Kp
    d_sign = delta_qp >= 0.
    d_sign = d_sign.to(p.dtype)

    # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
    # Shape  B x Kq x Kp.
    delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]  # B x 1 x Kp.
    return torch.sum(torch.clamp(1. - delta_hat, 0., 1.) * p, 2)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_dim, 400)
        self.linear2 = nn.Linear(400 + action_dim, 300)
        self.critic = nn.Linear(300, 1)

    def forward(self, x, a):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(torch.cat([x, a], 1)))
        value = self.critic(x)
        return value


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, 400)
        self.linear2 = nn.Linear(400, 300)
        self.actor = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        action = F.tanh(self.actor(x)) * self.max_action
        return action


class Critic_dis(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic_dis, self).__init__()
        self.linear1 = nn.Linear(state_dim, 400)
        self.linear2 = nn.Linear(400 + action_dim, 300)
        self.critic = nn.Linear(300, 51)

    def forward(self, x, a):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(torch.cat([x, a], 1)))
        value = F.softmax(self.critic(x))
        #Q_value1 = torch.sum(self.get_z_atoms() * value, 1)
        return value

    def get_z_atoms(self, Vmin, Vmax):
        return torch.linspace(Vmin, Vmax, 51).to(device)

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, save_num):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        # self.critic = Critic(state_dim, action_dim).to(device)
        # self.critic_target = Critic(state_dim, action_dim).to(device)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.critic_dis = Critic_dis(state_dim, action_dim).to(device)
        self.critic_dis_target = Critic_dis(state_dim, action_dim).to(device)
        self.critic_dis_target.load_state_dict(self.critic_dis.state_dict())
        self.critic_dis_optimizer = torch.optim.Adam(self.critic_dis.parameters(), lr=2.5e-4)

        self.max_action = max_action
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)
        # 保存20个最近的网络结构,是值函数的网络
        self.critic_arr = [Critic(state_dim, action_dim).to(device) for i in range(save_num)]
        self.count_critic = 0
        self.save_num = save_num

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, Vmin, Vmax, sample_times, if_change_VMAX, batch_size=100, discount=0.99, tau=0.001):
        for i in range(iterations):
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # 更新critic网络，这个网络仅仅用来辅助估计Q值。保存这个网络的参数然后估计Q值。
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()
            # compute the current Q estimate
            current_Q = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if self.count_critic >= self.save_num and if_change_VMAX:
                min_arr = []
                max_arr = []
                for count_net in range(self.save_num):
                    past_q_value = self.critic_arr[count_net](state, action)
                    min_arr.append(torch.min(past_q_value).item())
                    max_arr.append(torch.max(past_q_value).item())
                Vmax = moving_block_bootstrap_max(max_arr)

            #compute the distribution loss
            policy_noise = 0.2
            noise_clip = 0.5
            target_action = self.actor_target(next_state)
            next_action_arr = torch.zeros(sample_times, target_action.size()[0], target_action.size()[1]).to(device)
            next_state_arr = torch.zeros(sample_times, next_state.size()[0], next_state.size()[1]).to(device)
            #在目标动作周围采样100次，然后求出平均分布
            for count_sample in range(sample_times):
                noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action_arr[count_sample] = (target_action + noise).clamp(-self.max_action, self.max_action)
                next_state_arr[count_sample] = next_state
            next_action = next_action_arr.view(-1, target_action.size()[1])
            next_state = next_state_arr.view(-1, next_state.size()[1])
            target_Q1_dist = self.critic_dis_target(next_state, next_action)#(sample_times * batchsize) * 51
            #对target_dist_arr求平均，得到平均分布
            target_Q1_dist = target_Q1_dist.view(sample_times, batch_size, 51)
            mean_dist = torch.mean(target_Q1_dist.float(), 0)

            # target_Z_atoms为51个离散点，与dist相乘可以得到真正的动作值函数
            target_Z_atoms = torch.unsqueeze(self.critic_dis_target.get_z_atoms(Vmin, Vmax), 0).repeat(batch_size, 1)  # batch_size*51
            target_Z_atoms = reward + target_Z_atoms * discount * done
            # 取较小的Q值函数作为target，得到投影后的分布
            # 得到投影后的分布，输入Q1的分布，和想要的分布z_atoms，还有应用bellman更新后得到的分布target_z_atoms
            target_projected = _l2_project(target_Z_atoms, mean_dist, self.critic_dis_target.get_z_atoms(Vmin, Vmax))
            target_project_dist = target_projected.detach()
            # Train critic
            # 根据当前网络得到值函数的分布以及值函数本身
            # current_Q1_dist, current_Q2_dist, current_Q1, current_Q2 = self.critic(state, action)
            current_Q1_dist = self.critic_dis(state, action)

            loss1 = -torch.sum(target_project_dist * torch.log(current_Q1_dist+1e-10), 1)
            # loss2 = -torch.sum(target_project_dist * torch.log(current_Q2_dist), 1)
            mean_loss = torch.mean(loss1)
            self.critic_dis_optimizer.zero_grad()
            mean_loss.backward()
            self.critic_dis_optimizer.step()

            #compute the actor loss,使用critic_dis来计算actor_loss
            actor_loss = self.critic_dis(state, self.actor(state))
            actor_loss = torch.sum(self.critic_dis.get_z_atoms(Vmin, Vmax) * actor_loss, 1)
            actor_loss = -actor_loss.mean()

            #actor_loss = -self.critic(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic_dis.parameters(), self.critic_dis_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return actor_loss, mean_loss, current_Q1_dist, Vmax

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
