import rl

from environments.BreakoutEnv import breakout_env
from DQN.memory.replay_memory import ReplayMemory
from DQN.exploration_strategy import ExplorationStrategy
from DQN import trainer
from DQN.breakout import config
from DQN.breakout.my_network import MyNetworkManager
from DQN.breakout.env_config import Config1 as env_config
from datetime import datetime


class Settings:
    N_EPISODES = 2000
    EPISODE_LENGTH = 300
    MEMORY_SIZE = 800
    REPLAY_FREQUENCY = 8
    START_EPSILON = 0.9
    STOP_EPSILON = 0.01
    INPUT_DIM = env_config.height * env_config.width * 3
    N_ACTIONS = 2
    BATCH_SIZE = 64
    DISCOUNT_FACTOR = 0.98
    ALPHA = 1 / BATCH_SIZE / 100


class Hyperparams:
    learning_rate = 1e-3
    n_hidden_layer1 = 64
    n_hidden_layer2 = 16


def train():
    exploration_helper = ExplorationStrategy(settings.N_EPISODES * 0.7, settings.START_EPSILON, settings.STOP_EPSILON)
    replay_memory = ReplayMemory(settings.MEMORY_SIZE)

    dqn = trainer.DQNTeacher(
        agent,
        replay_memory,
        env,
        network_manager,
        exploration_helper,
        settings
    )

    dqn.train()


def test():
    network_manager.restore('./models/model.ckpt', )
    rl.test_agent(env, agent, network_manager, 10)


if __name__ == '__main__':
    is_train_mode = True

    settings = Settings()
    hyperparams = Hyperparams()
    hyperparams.input_dim = settings.INPUT_DIM
    hyperparams.output_dim = settings.N_ACTIONS

    env = breakout_env.BreakoutEnv()
    agent = config.BreakoutAgent('test1', env)

    str_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_path = f'./tensorboard/{str_time}'
    network_manager = MyNetworkManager(
        './models/model.ckpt',
        tensorboard_path,
        is_train_mode,
        hyperparams
    )

    if is_train_mode:
        train()
    else:
        test()
