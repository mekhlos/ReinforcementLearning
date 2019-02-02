import rl
import gym

from environments import env_wrapper
from networks.tf_network import NetworkManager
from DQN.cartpole.my_network import MyNetwork, MyNetworkTrainer
from DQN.memory.replay_memory import ReplayMemory
from DQN.exploration_strategy import ExplorationStrategy
from DQN import q_table as q_table_module, trainer


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


def train():
    exploration_helper = ExplorationStrategy(settings.N_EPISODES * 0.7, settings.START_EPSILON, settings.STOP_EPSILON)
    agent = rl.Agent('test1', env, observe_function=env.observe_f)
    replay_memory = ReplayMemory(settings.MEMORY_SIZE)
    network_manager = NetworkManager(settings.INPUT_DIM, settings.N_ACTIONS, MyNetwork, MyNetworkTrainer)
    # q_table = q_table_module.QTable(16, settings.N_ACTIONS)
    dqn = trainer.DQNTeacher(agent, replay_memory, env, network_manager, exploration_helper,
                             trainer.batch_replay, settings)
    dqn.train()


def test():
    pass


if __name__ == '__main__':
    settings = Settings()
    env = env_wrapper.EnvWrapper(gym.make('CartPole-v0'))
