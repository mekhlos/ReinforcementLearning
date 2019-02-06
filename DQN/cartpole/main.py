import rl
import gym

from environments import env_wrapper
from DQN.memory.replay_memory import ReplayMemory
from DQN.exploration_strategy import ExplorationStrategy
from DQN import trainer
from DQN.cartpole import config
from DQN.cartpole.my_network import MyNetworkManager


class Settings:
    N_EPISODES = 800
    EPISODE_LENGTH = 500
    MEMORY_SIZE = 800
    REPLAY_FREQUENCY = 2
    START_EPSILON = 1
    STOP_EPSILON = 0.01
    INPUT_DIM = 4
    N_ACTIONS = 2
    BATCH_SIZE = 64
    DISCOUNT_FACTOR = 0.96
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

    env = env_wrapper.EnvWrapper(gym.make('CartPole-v0'))
    agent = config.CartpoleAgent('test1', env)
    network_manager = MyNetworkManager(
        './models/model.ckpt',
        './tensorboard',
        is_train_mode,
        hyperparams
    )

    if is_train_mode:
        train()
    else:
        test()
