import numpy as np 
import collections
import torch
import torch.nn as nn
import math 
import torch.nn.functional as F

'''
THis file contains neural networks models of RL algorithms: 
DQN, REINFORCE, Actor-Critic, A2C, A3C, PPO, DDPG, TD3, SAC

'''
class DSR_embedding_nn(nn.Module):
    '''
    learn state embedding and predict immediate reward
    '''
    def __init__(self, state_num, action_num, dim, dsr_matrix=None):
        super(DSR_embedding_nn, self).__init__()
        if dsr_matrix is None:
            self.embedding = nn.Embedding(state_num, dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(dsr_matrix, freeze=True)
        self.linear1 = nn.Linear(dim, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, action_num)


    def forward(self, states, sr=None):
        if sr is None:
            x = self.embedding(states).squeeze(1)
        else:
            x = sr
        x2 = self.linear1(x)
        x2 = self.relu(x2)
        y = self.linear2(x2)
        # y = torch.sigmoid(y)
        return y, x

class DSR_sr_nn(nn.Module):
    '''
    learn SR based on state feature
    '''
    def __init__(self, dim):
        super(DSR_sr_nn, self).__init__()
        self.linear1 = nn.Linear(dim, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, dim)

    def forward(self, state_emb):
        x = self.linear1(state_emb)
        # x = self.relu(x)
        y = self.linear2(x)
        return y

class State2emb_embedding_nn(nn.Module):
    def __init__(self, state_num, action_num, dim, state2emb_matrix=None):
        super(State2emb_embedding_nn, self).__init__()
        if state2emb_matrix is None:
            self.embedding = nn.Embedding(state_num, dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(state2emb_matrix, freeze=True)

    def forward(self, states):
        x = self.embedding(states).squeeze(1)
        # x = F.normalize(x, dim=0, p=2)
        cov = torch.mm(x, x.T)
        # matrix = torch.sigmoid(cov)
        return x, cov

class State2emb_q_nn(nn.Module):
    def __init__(self, action_num, dim):
        super(State2emb_q_nn, self).__init__()
        self.linear = nn.Linear(dim, 16)
        self.linear2 = nn.Linear(16, action_num)
        self.relu = nn.ReLU()

    def forward(self, states_embedding):
        y = self.linear(states_embedding)
        y = self.relu(y)
        q_values = self.linear2(y)
        q_values = torch.sigmoid(q_values)
        return q_values


class State2Emb_nn(nn.Module):
    def __init__(self, state_num, action_num, dim, state2emb_matrix=None):
        super(State2Emb_nn, self).__init__()
        if state2emb_matrix is None:
            self.embedding = nn.Embedding(state_num, dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(state2emb_matrix, freeze=True)
        self.linear = nn.Linear(dim, 16)
        self.linear2 = nn.Linear(16, action_num)
        self.relu = nn.ReLU()
        # self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, states):
        x = self.embedding(states).squeeze(1)
        cov = torch.mm(x, x.T)
        matrix = torch.sigmoid(cov)
        y = self.linear(x)
        y = self.relu(y)
        q_values = self.linear2(y)
        # q_values = torch.sigmoid(q_values)
        return q_values, matrix

class DQN_maze_nn(nn.Module):
    def __init__(self, state_num, action_num, dim, dqn_matrix=None):
        super(DQN_maze_nn, self).__init__()
        if dqn_matrix is None:
            self.embedding = nn.Embedding(state_num, dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(dqn_matrix, freeze=True)
        self.linear = nn.Linear(dim, 16)
        self.linear2 = nn.Linear(16, action_num)
        self.relu = nn.ReLU()
        # self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, states):
        x = self.embedding(states).squeeze(1)
        y = self.linear(x)
        y = self.relu(y)
        q_values = self.linear2(y)
        # q_values = torch.sigmoid(q_values)
        return q_values

class DQN_node2vec_nn(nn.Module):
    def __init__(self, state_num, action_num, dim, node2vec_matrix=None):
        super(DQN_node2vec_nn, self).__init__()
        if node2vec_matrix is None:
            self.embedding = nn.Embedding(state_num, dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(node2vec_matrix, freeze=True)
        self.linear = nn.Linear(dim, 16)
        self.linear2 = nn.Linear(16, action_num)
        self.relu = nn.ReLU()

    def forward(self, states):
        x = self.embedding(states).squeeze(1)
        y = self.linear(x)
        y = self.relu(y)
        q_values = self.linear2(y)
        q_values = torch.sigmoid(q_values)
        return q_values


class DQN_pvf_nn(nn.Module):
    def __init__(self, state_num, action_num, dim, pvf_matrix=None):
        super(DQN_pvf_nn, self).__init__()
        if pvf_matrix is None:
            self.embedding = nn.Embedding(state_num, dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(pvf_matrix, freeze=True)
        self.linear = nn.Linear(dim, 32)
        self.linear2 = nn.Linear(32, action_num)
        self.relu = nn.ReLU()

    def forward(self, states):
        x = self.embedding(states).squeeze(1)
        y = self.linear(x)
        y = self.relu(y)
        q_values = self.linear2(y)
        q_values = torch.sigmoid(q_values)
        return q_values


class DQN_conv(nn.Module):
    '''
    NN for input as 2-dimensional matrix: image
    '''
    def __init__(self, input_size, n_actions):
        super(DQN_conv, self).__init__()
        # Process the image input
        self.conv = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_size)
        # output the q value of each action
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)



class DQN_1d(nn.Module):
    '''
    DQN for input in the form of 1-dimensional vector
    '''
    def __init__(self, input_size, n_actions):
        super(DQN_1d, self).__init__()
        # Output the q value of each action
        self.net=nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
            )

    def forward(self, x):
        return self.net(x)


class PG_net(nn.Module):
    '''
    Network for policy gradient algorithm. 

    Paramters:
    ==========
    Input: state featue 
    Output: Actions probability
    ==========
    '''
    def __init__(self, input_size,  n_actions):
        super(PG_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
            )

    def forward(self, x):
        x = self.net(x)
        return x



class A2C_net(nn.Module):
    def __init__(self, input_size, n_actions):
        super(A2C_net, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
            )

        self.value_net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )

    def forward(self,x):
        action_prob = self.policy_net(x)
        state_value = self.value_net(x)
        return action_prob, state_value



class A2C_net_ca(nn.Module):
    '''
    Actot-critic net for continuous action (ca) space .

    Parameter
    ---------
    Input_size: observation dimension
    action_size: action feature diemension
    mu: return the mean of output action
    var: the variance of output action
    value: the state value
    '''
    def __init__(self, input_size, action_size):
        super(A2C_net_ca, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            )
        self.mu = nn.Sequential(
            nn.Linear(32, action_size),
            nn.Tanh(),
            )
        self.var = nn.Sequential(
            nn.Linear(32, action_size),
            nn.Softplus(),
            )
        self.value = nn.Linear(32, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)


class DDPG_actor(nn.Module):
    '''
    DDPG is designed for continuous action

    Input state and then return the action
    '''
    def __init__(self, input_size, action_size):
        super(DDPG_actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), 
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Linear(32, action_size), 
            nn.Tanh(),
            )
    def forward(self, x):
        return self.net(x)


class DDPG_critic(nn.Module):
    '''
    input the state and action and then return the q value
    '''
    def __init__(self, input_size, action_size):

        super(DDPG_critic, self).__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(input_size, 64), 
            nn.ReLU(),
            )
        self.out_net = nn.Sequential(
            nn.Linear(64 + action_size, 32), 
            nn.ReLU(), 
            nn.Linear(32, 1),
            )
    def forward(self, x, a):
        obs = self.obs_net(x)
        q_val = self.out_net(torch.cat([obs, a], dim = 1))
        return q_val


class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        # return the probabilityof actions 

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        # return the estimated state value

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


class ModelActor(nn.Module):
    '''
    The network is designed for continuous action

    parameters:
    ----------
    obs_size: observation dimension
    act_size: action vetor dimension

    return: 
    ------
    mean of action vector
    '''
    def __init__(self, obs_size,act_size):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(), 
            np.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_size),
            nn.Tanh(),
            )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        return self.mu(x)

class ModelCritic(nn.Module):
    '''
    calculate the state value

    parameter:
    ---------
    obs_size: observation dimension

    return:
    ------
    state value (scaler)
    '''
    def __init__(self, obs_size):
        super(ModelCritic, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(obs_size, 64), 
            nn.ReLU(), 
            nn.Linear(64,64), 
            nn.ReLU(),
            nn.Linear(64, 1),
            )
    def forward(self, x):
        return self.value(x)




class TD3_actor(nn.Module):
    def __init__(self, state_size, action_size ):
        super(TD3_actor, self).__init__()
        self.lin1 = nn.Linear(state_size, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, action_size)
        # self.max_action = max_action

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu()
        x = self.lin3(x)
        x = torch.tanh(x)
        return x


class TD3_critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(TD3_critic, self).__init__()
        self.net1 = nn.Sequential(
                nn.Linear(state_size+action_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        self.net2 = nn.Sequential(
                nn.Linear(state_size+action_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.net1(sa)
        q2 = self.net2(sa)
        return q1, q2




class sac_actor(nn.Module):
    def __init__(self, state_size, action_size, min_log_std, max_log_std):
        super().__init__()
        self.min = min_log_std
        self.max = max_log_std

        self.net = nn.Sequential(
                nn.Linear(state_size, 64), 
                nn.ReLU(), 
                nn.Linear(64, 64), 
                nn.ReLU(), 
                nn.Linear(64, action_size)
        )

    def forward(self, x):
        x = self.net(x)
        log_std = nn.Linear(x)
        log_std = torch.clamp(log_std, self.min, self.max)
        return x, log_std


class sac_critic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net_q = nn.Sequential(
            nn.Linear(state_size+action_size, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1),
            )
        self.net_v = nn.Sequential(
            nn.Linear(state_size, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1),
            )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q = self.net_q(x)
        v = self.net_v(state)
        return q, v

















