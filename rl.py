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
        environment_state = self.environment.get_state()
        state = environment_state[[0, 1, 3]]
        return state.flatten()

    def take_action(self, action):
        self.environment.update(action)

    def is_alive(self):
        pass

    def come_to_life(self):
        while self.is_alive():
            state = self.observe()
            action = self.policy.get_action(state)
            self.take_action(action)


if __name__ == '__main__':
    pass
