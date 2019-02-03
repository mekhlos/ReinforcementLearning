import gym
import rl

from REINFORCE.cartpole import my_network
from REINFORCE.cartpole import config
from REINFORCE import memory as memory_module
from REINFORCE import trainer
from environments import env_wrapper


class Settings:
    N_EPISODES = 300
    INPUT_DIM = 4
    N_ACTIONS = 2
    DISCOUNT_FACTOR = 0.96


class Hyperparameters:
    LEARNING_RATE = 1e-2


def train():
    memory = memory_module.Memory(settings.DISCOUNT_FACTOR, settings.N_ACTIONS)
    training = trainer.REINFORCETeacher(agent, memory, env, network_manager, settings)
    training.train()


def test():
    network_manager.restore('./models/model.ckpt', )
    rl.test_agent(env, agent, network_manager, 10)


if __name__ == '__main__':
    settings = Settings()
    hyperparams = Hyperparameters()

    env = env_wrapper.EnvWrapper(gym.make('CartPole-v0'))
    settings.N_ACTIONS = env.env.action_space.n

    agent = config.CartpoleAgent('test1', env)
    network_manager = my_network.MyNetworkManager(
        settings.INPUT_DIM,
        settings.N_ACTIONS,
        './models/model.ckpt',
        './tensorboard',
        True
        # hyperparams
    )

    test()
