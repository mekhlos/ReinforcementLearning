import rl
import numpy as np


class REINFORCETeacher:
    def __init__(self, agent, memory, env, network_manager, settings):
        self.agent: rl.Agent = agent
        self.memory = memory
        self.network_manager = network_manager
        self.env = env
        self.settings = settings

        self.episode_ix = 0
        self.reward_per_episode_list = []
        self.loss_per_episode_list = []

    def learn_episode(self):
        s, a, r = self.memory.retrieve()

        loss = self.network_manager.learn(
            state=s,
            true_action=a,
            discounted_episode_rewards=r
        )

        mean_reward = np.mean(self.reward_per_episode_list[:-10])

        # Write TF Summaries
        self.network_manager.write_summaries(
            global_step=self.episode_ix,
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
        print(f'Episode loss: {self.loss_per_episode_list[-1]:.7f}')
        print(f'Episode reward: {self.reward_per_episode_list[-1]:.3f}')
        print(f'Mean Reward: {mean_reward:.3f}')
        print(f'Max reward so far: {maximum_reward_recorded:.3f}')
        print()

    def train(self):
        self.reward_per_episode_list = []
        self.loss_per_episode_list = []

        for self.episode_ix in range(self.settings.N_EPISODES):

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

                if self.episode_ix % 20 == 0:
                    self.env.display()

                if is_terminal:
                    loss = self.learn_episode()

                    self.reward_per_episode_list.append(np.sum(self.memory.rewards))
                    self.loss_per_episode_list.append(loss)
                    self.print_summary(len(self.memory))

                    self.memory.reset()
                    break

                state = new_state

            # Save model
            if self.episode_ix % 100 == 0:
                self.network_manager.save(path='./models/model.ckpt')

        self.network_manager.save(path='./models/model.ckpt')


if __name__ == '__main__':
    pass
