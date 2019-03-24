class EnvironmentInterface:

    def display(self):
        raise NotImplementedError('Please implement me!')

    def get_action_space(self):
        raise NotImplementedError('Please implement me!')

    def update(self, action):
        raise NotImplementedError('Please implement me!')

    def reset(self):
        raise NotImplementedError('Please implement me!')

    def get_state(self):
        raise NotImplementedError('Please implement me!')

    def close(self):
        raise NotImplementedError('Please implement me!')
