import sys
import os
import root_file
import rl

gwpath = os.path.join(root_file.ROOT_DIR, 'environments/GridworldEnv')
sys.path.append(gwpath)

from dqn.brain.tf_network import NetworkManager
from dqn.brain.replay_memory import ReplayMemory
from dqn.exploration_strategy import ExplorationStrategy
from environments.GridworldEnv import gridworld_env
from environments.GridworldEnv.grid_configs import configs
from dqn.brain import q_learning
import dqn_train

config = configs.config2
M = len(config)
N = len(config[0])


class Settings:
    N_EPISODES = 1000
    EPISODE_LENGTH = 100
    MEMORY_SIZE = 400
    REPLAY_FREQUENCY = 2
    START_EPSILON = 1
    STOP_EPSILON = 0.01
    INPUT_DIM = M * N * 3
    N_ACTIONS = 4
    BATCH_SIZE = 32
    DISCOUNT_FACTOR = 0.99
    ALPHA = 1 / BATCH_SIZE / 100


class Settings2:
    N_EPISODES = 1000
    EPISODE_LENGTH = 300
    MEMORY_SIZE = 600
    REPLAY_FREQUENCY = 3
    START_EPSILON = 1
    STOP_EPSILON = 0.01
    INPUT_DIM = M * N * 3
    N_ACTIONS = 4
    BATCH_SIZE = 64
    DISCOUNT_FACTOR = 0.985
    ALPHA = 1 / BATCH_SIZE / 100


if __name__ == '__main__':
    settings = Settings2()
    env = gridworld_env.GridworldEnv(M, N, grid=configs.to_state(config))
    exploration_helper = ExplorationStrategy(settings.N_EPISODES * 0.7, settings.START_EPSILON, settings.STOP_EPSILON)
    agent = rl.Agent('test1', env)
    replay_memory = ReplayMemory(settings.MEMORY_SIZE)
    network_manager = NetworkManager(settings.INPUT_DIM, settings.N_ACTIONS)
    q_table = q_learning.QTable(16, settings.N_ACTIONS)
    dqn = dqn_train.DQNTeacher(agent, replay_memory, env, network_manager, exploration_helper,
                               dqn_train.iterative_replay, settings)
    dqn.train()
