import numpy as np 
import collections
import torch
import torch.nn as nn
import math 


'''
This file contains useful buidling blocks of RL algorithms:
ExperienceBuffer, Advantage estimator, N-step Q-value, Unpack batch, Moving average
PrioritizedReplayBuffer, alpha_sync, log_prob
'''
def cal_q_vals(rewards, gamma):
    '''
    Calculate q values from the end of episode 
    This is for REINFORCE algorithm

    parameters
    ----------
    rewards of an episode 

    return
    ------
    The empirical return of each state-action pair (q value) of an episode

    '''
    res = []
    sum_r = 0
    for r in reversed(rewards):
        sum_r *= gamma 
        sum_r += r 
        res.append(sum_r)

    return list(reversed(res))


def cal_nstep_qvals(net, actions, next_states, rewards, gamma, n=1):
    '''Calcuate n-step unrolling  Q-value

    parameter
    --------
    net: convert state to action value
    n: the number of unrolling steps
    states: 
    next_states
    rewards

    return 
    ------
    n-step unrolling Q-values

    '''
    Q_values = np.zeros(len(actions))
    # done_mask = [ns == True for ns in next_states]
    dis_rewards = np.zeros(len(actions))
    dis_factor = [gamma**i for i in range(n)]
    next_vals = np.zeros(len(actions))
    for i in range(len(actions)-n):
        Q_values[i] = np.sum(np.array(dis_factor)*rewards[i:i+n])
        next_vals[i] = net(next_states[i+n]).max()
    if n > 1:
        for j in range(len(actions)-n, len(actions)):
            Q_values[i] = np.sum(np.array(dis_factor)[:len(actions)-j]*rewards[j:])

    Q_values = Q_values + (gamma**n)*next_vals
    return Q_values



def cal_adv(states, actions, rewards, dones, next_states, net, gamma):
    '''
    calulate the advantage of each state-action pair. this is for A2C
    '''
    _, next_vals = net(next_states)
    Q_values = rewards + gamma*next_vals
    _, state_vals = net(states)
    adv_vals = Q_values - state_vals

    return adv_vals

def cal_nstep_adv(batch, net, gamma, n=1):
    Q_values = cal_nstep_qvals(batch, n=n)
    _, state_vals = net(states)
    adv_vals = Q_values - state_vals
    return adv_vals


def cal_adv_ref(trajectory, net_crt, states_v, device='cpu'):
    '''
    calculate advantage for training actor and reference values for training critic in PPO.

    parameter
    ---------
    trajectory: several episodes concatenated together.

    return
    ------
    advantages
    reference values

    '''
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])):
        if exp.done: 
            delta = exp.reward - val 
            last_gae = delta
        else:
            delta = exp.reward + gamma*next_val - val 
            last_gae = delta + gamma*gae_lambda*last_gae

        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
    return adv_v, ref_v 


def cal_logprob(mu, var, actions):
    '''
    calculate log prob for A2C_agnet_ca
    '''
    p1 = -((mu - actions)**2) / (2*var.clamp(min=1e-3))
    p2 = -torch.log(torch.sqrt(2*math.pi*var))
    return p1 + p2

class ExperienceBuffer:
    ''''This is the experience buffer for DQN

    parameter
    ---------
    buffer_size
    batch_size

    return
    -------
    sample(): return states, actions, rewards, dones, next_states in form 
    of numpy array.
    '''

    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for idx in indices:
            states.extend([self.buffer[idx].state])
            actions.extend([self.buffer[idx].action])
            rewards.extend([self.buffer[idx].reward])
            next_states.extend([self.buffer[idx].next_state])
            dones.extend([self.buffer[idx].done])

        states_a = np.array(states)
        actions_a = np.array(actions)
        rewards_a = np.array(rewards)
        next_states_a = np.array(next_states)
        dones_a = np.array(dones)
        return  states_a, actions_a, rewards_a,  dones_a, next_states_a

        # states_v = torch.FloatTensor(states_a)
        # actions_v = torch.FloatTensor(actions_a)
        # rewards_v = torch.FloatTensor(rewards_a)
        # next_states_v = torch.FloatTensor(next_states_a)
        # dones_v = torch.BoolTensor(dones_a)
        # return states_v, actions_v, rewards_v, dones_v, next_states_v

class ExperienceBuffer_state2emb:
    ''''This is the experience buffer for DQN

    parameter
    ---------
    buffer_size
    batch_size

    return
    -------
    sample(): return states, actions, rewards, dones, next_states in form 
    of numpy array.
    '''

    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        neighbors = []
        for idx in indices:
            states.extend([self.buffer[idx].state])
            actions.extend([self.buffer[idx].action])
            rewards.extend([self.buffer[idx].reward])
            next_states.extend([self.buffer[idx].next_state])
            dones.extend([self.buffer[idx].done])
            neighbors.extend([self.buffer[idx].neighbors])

        states_a = np.array(states)
        actions_a = np.array(actions)
        rewards_a = np.array(rewards)
        next_states_a = np.array(next_states)
        dones_a = np.array(dones)
        neighbors_a = np.array(neighbors)
        return  states_a, actions_a, rewards_a,  dones_a, next_states_a, neighbors_a


class EpisodeBuffer:
    '''A buffer stores a list of episodes. Each episode constains a list of experience

        parameters:
        ----------
        buffer size: the maximum number of episode
        batch size: the number of episode to sample

        return: 
        ------
        The experience of sampled epsidoe in form of tensor
    '''
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, episode):
        self.buffer.append(episode)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for idx in indices:
            states.extend([self.buffer[idx].state])
            actions.extend([self.buffer[idx].action])
            rewards.extend([self.buffer[idx].reward])
            next_states.extend([self.buffer[idx].next_state])
            dones.extend([self.buffer[idx].done])

        states_a = np.array(states)
        actions_a = np.array(actions)
        rewards_a = np.array(rewards)
        next_states_a = np.array(next_states)
        dones_a = np.array(dones)

        states_v = torch.FloatTensor(states_a)
        actions_v = torch.FloatTensor(actions_a)
        rewards_v = torch.FloatTensor(rewards_a)
        next_states_v = torch.FloatTensor(next_states_a)
        dones_v = torch.BoolTensor(dones_a)
        return states_v, actions_v, rewards_v, dones_v, next_states_v


def alpha_sync(net, target_net, alpha):
    """Update target net slowly

    parameters:
    -----
    alpha: step size

    return: 
    -------
    updated target net
    """
    assert isinstance(alpha, float)
    assert 0.0 < alpha <= 1.0
    state = net.state_dict()
    tgt_state = target_net.state_dict()
    for k, v in state.items():
        tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
    target_net.load_state_dict(tgt_state)
    return target_net

# class PrioritizedReplayBuffer(ExperienceReplayBuffer)

def array_to_tensor(states_a, actions_a, rewards_a,  dones_a, next_states_a):
    states_v = torch.FloatTensor(states_a)
    actions_v = torch.FloatTensor(actions_a)
    rewards_v = torch.FloatTensor(rewards_a)
    next_states_v = torch.FloatTensor(next_states_a)
    dones_v = torch.BoolTensor(dones_a)
    return states_v, actions_v, rewards_v, next_states_v, dones_v

def unpack_batch(batch, net, gamma, device='cpu'):
    '''
    convert batch into training tensors. This is used for policy gradient algorithms

    parameters:
    ----------
    batch: transitions 
    net: agent net 

    return:
    -------
    states tensore, actions tensor, target state values
    '''
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    next_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state,copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.next_state is not None:
            not_done_idx.append(idx)
            next_states.append(np.array(exp.next_state, copy=False))
    states_v = torch.FloatTensor(states).to(device)
    actions_v = torch.LongTensor(actions).to(device)
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        next_states_v = torch.FloatTensor(next_states).to(device)
        _, next_vals_v = net(next_states_v)
        next_vals_np = next_vals_v.data.cpu().numpy()[:,0]
        rewards_np[not_done_idx] += next_vals_np + next_val_gamma*next_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v 


def unpack_batch(batch, net, gamma, n=1, device='cpu'):
    ''' unpack batch of policy gradient algorithms

    paramter
    -------
    batch 
    net
    gamma 
    n: the number of unrolling steps

    return:
    states_v,
    actions_v
    adv_vals
    Q_vals
    '''
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    next_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state,copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.next_state is not None:
            not_done_idx.append(idx)
            next_states.append(np.array(exp.next_state, copy=False))

    states_v = torch.FloatTensor(states).to(device)
    actions_v = torch.LongTensor(actions).to(device)
    rewards_np = np.array(rewards, dtype=np.float32)
    Q_values = cal_nstep_qvals(batch, n=n)
    _, state_vals = net(states_v)
    adv_vals = Q_values - state_vals
    return states_v, actions_v, adv_vals


class PrioReplayBuffer:
    def __init__(self, exp_source, buf_size, prob_alpha=0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        max_prio = self.priorities.max() if self.buffer else 1.0
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

