import numpy as np


def discount_and_normalize_rewards2(rewards, gamma):
    discounted_rewards = np.array(rewards).copy()

    for i in range(len(rewards) - 2, -1, -1):
        discounted_rewards[i] = rewards[i] + gamma * discounted_rewards[i + 1]

    return (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()


def discount_and_normalize_rewards(episode_rewards, gamma):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / std

    return discounted_episode_rewards


class Memory:
    def __init__(self, gamma, n_actions):
        self.states = []
        self.actions = []
        self.rewards = []
        self.gamma = gamma
        self.n_actions = n_actions

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add(self, state, action, reward):
        one_hot_action = np.zeros(self.n_actions)
        one_hot_action[action] = 1

        self.states.append(state)
        self.actions.append(one_hot_action)
        self.rewards.append(reward)

    def retrieve(self):
        states = np.vstack(np.array(self.states))
        actions = np.vstack(np.array(self.actions))
        rewards = discount_and_normalize_rewards(self.rewards, self.gamma)
        print(f'retrieving {states.shape} memories')

        return states, actions, rewards

    def __len__(self):
        return len(self.states)
