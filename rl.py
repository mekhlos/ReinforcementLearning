class Policy:
    def __init__(self):
        pass

    def get_action(self, state):
        pass


class Agent:
    def __init__(self, agent_id, environment, policy=None):
        self.agent_id = agent_id
        self.environment = environment
        self.policy = policy

    def observe(self):
        raise NotImplementedError('Please implement me!')

    def take_action(self, action):
        self.environment.update(action)

    def is_alive(self):
        pass

    def get_next_action(self, state, network):
        raise NotImplementedError('Please implement me!')

    def come_to_life(self):
        while self.is_alive():
            state = self.observe()
            action = self.policy.get_action(state)
            self.take_action(action)


def test_agent(env, agent, network_interface, n_trials):
    env.reset()
    rewards = []

    for episode_ix in range(n_trials):
        env.reset()
        state = agent.observe()

        total_rewards = 0
        print('****************************************************')
        print(f'EPISODE {episode_ix}')

        while True:
            env.display()
            action = agent.get_next_action(state, network_interface)

            _, reward, is_terminal, info = env.update(action)
            new_state = agent.observe()

            total_rewards += reward

            if is_terminal:
                rewards.append(total_rewards)
                print('Score', total_rewards)
                break

            state = new_state

    env.close()
    print(f'Score over time: {sum(rewards) / n_trials:.3f}')


if __name__ == '__main__':
    pass
