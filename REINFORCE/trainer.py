import gym
import rl
import numpy as np

from REINFORCE.cartpole import network as network_module
from REINFORCE import memory as memory_module
from environments import env_wrapper
from utils import helpers


class Settings:
    N_EPISODES = 300
    INPUT_DIM = 4
    N_ACTIONS = 2
    DISCOUNT_FACTOR = 0.96


class Hyperparameters:
    LEARNING_RATE = 1e-2


class REINFORCETeacher:
    def __init__(self, agent: rl.Agent, memory, env, network_manager, settings):
        self.agent = agent
        self.memory = memory
        self.network_manager = network_manager
        self.env = env
        self.settings = settings

        self.episode_ix = 0
        self.reward_per_episode_list = []
        self.loss_per_episode_list = []

    def learn_episode(self, episode_ix):
        s, a, r = self.memory.retrieve()

        loss = self.network_manager.learn(
            state=s,
            true_action=a,
            discounted_episode_rewards=r
        )

        mean_reward = np.mean(self.reward_per_episode_list[:-10])

        # Write TF Summaries
        network_manager.write_summaries(
            episode_ix=episode_ix,
            state=s,
            true_action=a,
            discounted_episode_rewards=r,
            mean_reward=mean_reward
        )

        return loss

    def print_summary(self, n_steps):
        mean_reward = np.mean(self.reward_per_episode_list[:-10])
        maximum_reward_recorded = np.max(self.reward_per_episode_list)

        print('==========================================')
        print(f'Episode {self.episode_ix}')
        print(f'Finished in {n_steps} steps')
        print(f'Episode loss: {self.loss_per_episode_list[-1]}')
        print(f'Episode reward: {self.reward_per_episode_list[-1]}')
        print(f'Mean Reward: {mean_reward}')
        print(f'Max reward so far: {maximum_reward_recorded}')
        print()

    def train(self):
        self.reward_per_episode_list = []
        self.loss_per_episode_list = []

        for episode_ix in range(self.settings.N_EPISODES):

            self.env.reset()
            state = self.agent.observe()
            self.env.display()

            while True:
                action_probability_distribution = self.network_manager.predict_one(state)

                action = np.random.choice(
                    range(self.settings.N_ACTIONS),
                    p=action_probability_distribution.flatten()
                )  # select action w.r.t the actions prob

                _, reward, is_terminal, info = self.env.update(action)
                new_state = self.agent.observe()

                self.memory.add(state, action, reward)

                if episode_ix % 20 == 0:
                    self.env.display()

                if is_terminal:
                    self.reward_per_episode_list.append(np.sum(self.memory.rewards))

                    loss = self.learn_episode(episode_ix)
                    self.memory.reset()

                    self.loss_per_episode_list.append(loss)
                    self.print_summary(len(self.memory))

                    break

                state = new_state

            # Save model
            if episode_ix % 100 == 0:
                self.network_manager.save(path='./models/model.ckpt')

        self.network_manager.save(path='./models/model.ckpt')


if __name__ == '__main__':
    settings = Settings()
    hyperparams = Hyperparameters()

    env = env_wrapper.EnvWrapper(gym.make('CartPole-v0'))

    agent = rl.Agent('test1', env, observe_function=env.observe_f)
    network_manager = network_module.NetworkManager(
        settings.INPUT_DIM,
        settings.N_ACTIONS,
        network_module.Network,
        network_module.NetworkTrainer,
        hyperparams
    )
    memory = memory_module.Memory(settings.DISCOUNT_FACTOR, settings.N_ACTIONS)
    settings.N_ACTIONS = env.env.action_space.n
    trainer = REINFORCETeacher(agent, memory, env, network_manager, settings)
    trainer.train()
