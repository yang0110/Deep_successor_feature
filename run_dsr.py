import numpy as np 
import gym 
import gym_minigrid
from gym_minigrid.wrappers import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from agents.dsr_agent import DSR
from agents.dsr_agent_2 import DSR_2
from pygsp import graphs
from utils import *

result_path = '../results/'
data_path = '../data/'

env = gym.make('MiniGrid-Empty-8x8-v0')
# env = gym.make('MiniGrid-Empty-16x16-v0')
# env = gym.make('MiniGrid-FourRooms-v0')
# env = gym.make('MiniGrid-MultiRoom-N6-v0')
# env = gym.make('MiniGrid-KeyCorridorS5R3-v0')
env = ReseedWrapper(env, seeds=[3])
# random seeds [0,2,3,4]
env.reset()
grid = env.grid.encode()
size = grid.shape[0]-2
state_num = size*size*4

# img = env.render('rgb_array')
# plt.imshow(img)
# plt.axis('off')
# plt.tight_layout()
# # plt.savefig(result_path+'mini_grid_key_corridors5r3'+'.png', dpi=100)
# plt.show()

alpha = 0.1
gamma = 0.9
epsilon = 0.1
epi_num = 1000

max_step = 100
emb_dim = 10
learning_rate = 0.1
buffer_size = 2000
batch_size = 64
beta = 1
loop_num = 1


dsr = DSR(env, size, state_num, beta, gamma, epsilon, epi_num, max_step, learning_rate, buffer_size, batch_size, emb_dim, dsr_matrix=None)
dsr_epi_total_reward, dsr_embedding, dsr_state_count, dsr_reward_loss, dsr_sr_loss = dsr.run()

plt.figure(figsize=(6,4))
plt.plot(moving_average(dsr_epi_total_reward, n=10), label='DSR')
plt.xlabel('episode', fontsize=12)
plt.ylabel('episode reward', fontsize=12)
plt.legend(loc=4, fontsize=12)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(dsr_reward_loss, label='DSR')
plt.xlabel('episode', fontsize=12)
plt.ylabel('reward loss', fontsize=12)
plt.legend(loc=4, fontsize=12)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(dsr_sr_loss, label='DSR')
plt.xlabel('episode', fontsize=12)
plt.ylabel('sr loss', fontsize=12)
plt.legend(loc=4, fontsize=12)
plt.show()

a = np.array(dsr_state_count).reshape((4, size*size)).mean(axis=0)
b = a/a.max()
b = b.reshape((size,size))
plt.matshow(b, cmap = plt.get_cmap('BuGn'))
plt.savefig(result_path+'state_count_heatmap_Q+'+'.png', dpi=100)
plt.show()
