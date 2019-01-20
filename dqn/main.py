import sys
import os
import root_file
import rl

gwpath = os.path.join(root_file.ROOT_DIR, 'environments/GridworldEnv')
sys.path.append(gwpath)

from dqn.network.tf_network import NetworkManager
# from dqn.network import NetworkManager
from dqn.memory.replay_memory import ReplayMemory
from dqn.exploration_strategy import ExplorationStrategy
from environments.GridworldEnv import gridworld_env
from environments.GridworldEnv.grid_configs import configs
from dqn import q_table, dqn_trainer


class Settings:
    CONFIG = configs.config2
    M = len(CONFIG)
    N = len(CONFIG[0])
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
    CONFIG = configs.config2
    M = len(CONFIG)
    N = len(CONFIG[0])
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


class Settings3:
    CONFIG = configs.config1
    M = len(CONFIG)
    N = len(CONFIG[0])
    N_EPISODES = 2500
    EPISODE_LENGTH = 2000
    MEMORY_SIZE = 1000
    REPLAY_FREQUENCY = 4
    START_EPSILON = 1
    STOP_EPSILON = 0.01
    INPUT_DIM = M * N * 3
    N_ACTIONS = 4
    BATCH_SIZE = 128
    DISCOUNT_FACTOR = 0.99
    ALPHA = 1 / BATCH_SIZE / 100


if __name__ == '__main__':
    settings = Settings()
    env = gridworld_env.GridworldEnv(settings.M, settings.N, grid=configs.to_state(settings.CONFIG))
    exploration_helper = ExplorationStrategy(settings.N_EPISODES * 0.7, settings.START_EPSILON, settings.STOP_EPSILON)
    agent = rl.Agent('test1', env)
    replay_memory = ReplayMemory(settings.MEMORY_SIZE)
    network_manager = NetworkManager(settings.INPUT_DIM, settings.N_ACTIONS)
    q_table = q_table.QTable(16, settings.N_ACTIONS)
    dqn = dqn_trainer.DQNTeacher(agent, replay_memory, env, network_manager, exploration_helper,
                                 dqn_trainer.batch_replay, settings)
    dqn.train()
