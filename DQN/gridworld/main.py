import sys
import os
import root_file
import rl

gwpath = os.path.join(root_file.ROOT_DIR, 'environments/GridworldEnv')
sys.path.append(gwpath)

from DQN.memory.replay_memory import ReplayMemory
from DQN.exploration_strategy import ExplorationStrategy
from environments.GridworldEnv import gridworld_env
from environments.GridworldEnv.grid_configs import configs as grid_configs
from DQN import trainer
from DQN import trainer_with_table
from DQN.q_table import QTable
from DQN.gridworld import config
from DQN.gridworld.my_network import MyNetworkManager


class Settings:
    CONFIG = grid_configs.config3
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
    CONFIG = grid_configs.config2
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
    CONFIG = grid_configs.config1
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


class Hyperparams:
    learning_rate = 1e-3
    # n_hidden_layer1 = 1024
    n_hidden_layer1 = 2048


def train():
    exploration_helper = ExplorationStrategy(settings.N_EPISODES * 0.7, settings.START_EPSILON, settings.STOP_EPSILON)
    replay_memory = ReplayMemory(settings.MEMORY_SIZE)

    # dqn = trainer.DQNTeacher(
    #     agent,
    #     replay_memory,
    #     env,
    #     network_manager,
    #     exploration_helper,
    #     settings,
    #     save_path
    # )

    q_table = QTable(settings.INPUT_DIM // 3, settings.N_ACTIONS)
    dqn = trainer_with_table.DQNTeacherWithTable(
        agent,
        replay_memory,
        env,
        exploration_helper,
        settings,
        q_table,
        save_path
    )

    dqn.train()
    # print(dqn.q_table.table)


def test():
    # network_manager.restore(save_path)
    network_manager = QTable(settings.INPUT_DIM // 3, settings.N_ACTIONS)
    rl.test_agent(env, agent, network_manager, 10, 0.5)


if __name__ == '__main__':
    is_train_mode = False
    save_path = './models'

    settings = Settings3()
    hyperparams = Hyperparams()
    hyperparams.input_dim = settings.INPUT_DIM
    hyperparams.output_dim = settings.N_ACTIONS

    env = gridworld_env.GridworldEnv(settings.M, settings.N, grid=grid_configs.to_state(settings.CONFIG))
    agent = config.GridworldAgent('test1', env)
    network_manager = MyNetworkManager(
        save_path,
        './tensorboard',
        is_train_mode,
        hyperparams
    )

    if is_train_mode:
        train()
    else:
        test()
