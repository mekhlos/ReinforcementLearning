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
    learning_rate = 1e-2
    n_hidden_layer1 = 10
    n_hidden_layer2 = 2


def train():
    memory = memory_module.Memory(settings.DISCOUNT_FACTOR, settings.N_ACTIONS)
    training = trainer.REINFORCETeacher(agent, memory, env, network_manager, settings)
    training.train()


def test():
    network_manager.restore('./models/model.ckpt', )
    rl.test_agent(env, agent, network_manager, 10)


if __name__ == '__main__':
    is_train_mode = True

    settings = Settings()
    hyperparams = Hyperparameters()

    env = env_wrapper.EnvWrapper(gym.make('CartPole-v0'))
    settings.N_ACTIONS = env.env.action_space.n
    hyperparams.input_dim = settings.INPUT_DIM
    hyperparams.output_dim = settings.N_ACTIONS

    agent = config.CartpoleAgent('test1', env)
    network_manager = my_network.MyNetworkManager(
        './models/model.ckpt',
        './tensorboard',
        True,
        hyperparams
    )

    if is_train_mode:
        train()
    else:
        test()
