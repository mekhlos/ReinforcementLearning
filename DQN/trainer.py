import rl

import numpy as np


class DQNTeacher:
    def __init__(self, agent, replay_memory, env, network_manager, exploration_helper, settings, save_path):
        self.agent: rl.Agent = agent
        self.replay_memory = replay_memory
        self.env = env
        self.network_manager = network_manager
        self.exploration_helper = exploration_helper
        self.settings = settings

        self.episode_ix = 0
        self.step_ix = 0
        self.reward_per_episode_list = []
        self.loss_per_episode_list = []
        self.save_path = save_path

    def replay(self):
        state, action, reward, new_state, terminal_flag = self.replay_memory.sample(self.settings.BATCH_SIZE)
        q_values = self.network_manager.predict(state)
        terminal_ix = np.where(terminal_flag > 0)[0]
        non_terminal_ix = np.where(terminal_flag < 1)[0]
        new_q_values = self.network_manager.predict(new_state[non_terminal_ix])

        target = q_values.copy()
        target[terminal_ix, action[terminal_ix]] = reward[terminal_ix]
        target[non_terminal_ix, action[non_terminal_ix]] = \
            reward[non_terminal_ix] + self.settings.DISCOUNT_FACTOR * new_q_values.max(1)

        loss = self.network_manager.learn(state, target)

        mean_reward = np.mean(self.reward_per_episode_list[:-10])
        self.network_manager.write_summaries(self.episode_ix, state, target, mean_reward)

        return loss

    def print_summary(self, n_steps):
        mean_reward = np.mean(self.reward_per_episode_list[:-10])
        maximum_reward_recorded = np.max(self.reward_per_episode_list)

        print('==========================================')
        print(f'Episode {self.episode_ix}')
        print(f'Finished in {n_steps} steps')
        print(f'Episode loss: {self.loss_per_episode_list[-1]:.7f}')
        print(f'Episode reward: {self.reward_per_episode_list[-1]:.3f}')
        print(f'Mean Reward: {mean_reward:.3f}')
        print(f'Max reward so far: {maximum_reward_recorded:.3f}')
        print(f'Epsilon: {self.exploration_helper.epsilon:.3f}')
        print()

    def train(self, display_frequency=10):
        self.reward_per_episode_list = []
        self.loss_per_episode_list = []

        for self.episode_ix in range(self.settings.N_EPISODES):

            self.env.reset()
            state = self.agent.observe()
            self.env.display()

            self.reward_per_episode_list.append(0)
            self.loss_per_episode_list.append(0)

            for step_ix in range(self.settings.EPISODE_LENGTH):
                q_values = self.network_manager.predict_one(state)

                action = self.exploration_helper.epsilon_greedy(q_values)
                _, reward, is_terminal, _ = self.env.update(self.env.get_action_space()[action])
                new_state = self.agent.observe()

                self.replay_memory.add(state, action, reward, new_state, is_terminal)
                self.reward_per_episode_list[-1] += reward

                if len(self.replay_memory) > self.settings.BATCH_SIZE and \
                        step_ix % self.settings.REPLAY_FREQUENCY == 0:
                    loss = self.replay()
                    self.loss_per_episode_list[-1] += loss

                if self.episode_ix % display_frequency == 0:
                    self.env.display()

                if is_terminal or step_ix + 1 == self.settings.EPISODE_LENGTH:
                    self.print_summary(step_ix)

                    break

                state = new_state

            self.exploration_helper.update_epsilon2()

            # Save model
            if self.episode_ix % 100 == 0:
                self.network_manager.save(path=self.save_path)

        self.network_manager.save(path=self.save_path)


if __name__ == '__main__':
    pass
